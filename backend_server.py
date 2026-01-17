"""
Driver Behavior Detection Backend API Server
- 다중 사용자 지원 (세션별 버퍼 관리)
- 배치 추론으로 GPU 효율 최대화
- WebSocket으로 실시간 통신
- SQLite 기반 사용자 인증 (시연용)
- GPU 상시 대기 + 즉시 배치 처리
- 최적화: torch.compile, CUDA Streams, Pinned Memory
"""
import os
import sys

# CUDA 메모리 단편화 방지 (torch import 전에 설정)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import base64
import asyncio
import numpy as np
import torch
import time
import uuid
import sqlite3
from collections import defaultdict
from io import BytesIO
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import threading
from queue import Queue, Empty
from dataclasses import dataclass

# 모델 로드
sys.path.insert(0, '/root/Driver_monitoring')
from model import DriverBehaviorModel

# 클래스 정의
CLASS_NAMES = {
    0: "Normal",
    1: "Drowsy",
    2: "Searching",
    3: "Phone",
    4: "Assault"
}

# 전역 변수
model = None
compiled_model = None  # torch.compile 최적화 모델
device = None
preallocated_buffer = None  # GPU 메모리 사전 확보용

# GPU 정규화 상수 (CUDA 텐서)
gpu_mean = None
gpu_std = None

# CUDA Streams (비동기 처리용)
inference_stream = None
transfer_stream = None

# Pinned Memory 버퍼 (CPU-GPU 전송 최적화)
pinned_buffer = None

# 메인 이벤트 루프 (스레드간 WebSocket 통신용)
main_event_loop = None

# 다중 사용자 세션 관리
user_sessions: Dict[str, Dict] = {}
sessions_lock = threading.Lock()

# 배치 추론 설정 (GPU 최적화 - Video Swin Transformer 메모리 고려)
BATCH_SIZE = 16  # 8 -> 16 (처리량 2배 증가, OOM 방지)
FRAMES_PER_INFERENCE = 30
FRAME_BUFFER_SIZE = 60  # 버퍼 크기 (60프레임 모으고)
FRAME_SHIFT = 10  # 추론 후 시프트량 (10프레임씩 이동 = 33% 새 데이터)
BATCH_TIMEOUT = 0.1  # 50ms -> 100ms (배치 채우기 시간 증가)

# 추론 큐 (즉시 처리용)
@dataclass
class InferenceJob:
    session_id: str
    frames: List[np.ndarray]
    websocket: Optional[any] = None
    timestamp: float = 0.0

inference_queue: Queue = Queue()
results_store: Dict[str, Dict] = {}  # HTTP 폴링용 결과 저장

# 디버그용 프레임 저장
DEBUG_DIR = "/tmp/inference_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)
inference_history: List[Dict] = []  # 최근 추론 기록 (최대 100개)
SAVE_DEBUG_FRAMES = False  # 프로덕션에서는 비활성화 (성능 최적화)

# 비동기 디버그 저장용 스레드 풀
from concurrent.futures import ThreadPoolExecutor
debug_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="debug_saver")

# SQLite DB 설정
DB_PATH = '/root/users.db'

def init_database():
    """SQLite 데이터베이스 초기화"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 사용자 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password TEXT NOT NULL,
            address TEXT,
            region_name TEXT,
            region_campaign TEXT,
            region_target INTEGER DEFAULT 90,
            region_reward TEXT,
            score INTEGER DEFAULT 80,
            discount_rate INTEGER DEFAULT 0,
            created_at TEXT
        )
    ''')

    # 운전 기록 테이블 (세션 정보)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS driving_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT DEFAULT 'driving',
            total_detections INTEGER DEFAULT 0,
            normal_count INTEGER DEFAULT 0,
            drowsy_count INTEGER DEFAULT 0,
            searching_count INTEGER DEFAULT 0,
            phone_count INTEGER DEFAULT 0,
            assault_count INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # 개별 감지 기록 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS driving_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driving_log_id INTEGER NOT NULL,
            detected_at TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (driving_log_id) REFERENCES driving_logs(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized!")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_model():
    """모델 로드 및 GPU 워밍업 + 메모리 사전 확보 + 최적화"""
    global model, compiled_model, device, preallocated_buffer, gpu_mean, gpu_std
    global inference_stream, transfer_stream, pinned_buffer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ===== 최적화 1: cuDNN 벤치마크 활성화 =====
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 입력 크기 고정시 최적 알고리즘 자동 선택
        torch.backends.cudnn.deterministic = False  # 약간의 비결정성 허용 (속도↑)
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 사용 (Ada GPU 최적화)
        torch.backends.cudnn.allow_tf32 = True
        print("✅ cuDNN benchmark + TF32 enabled")

    # ===== 최적화 2: CUDA Streams 생성 =====
    if torch.cuda.is_available():
        inference_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()
        print("✅ CUDA Streams created (async transfer + inference)")

    # GPU 정규화 상수 초기화
    gpu_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    gpu_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_mem:.1f} GB")

    model = DriverBehaviorModel(num_classes=5, pretrained=False)
    checkpoint = torch.load('/root/Driver_monitoring/pytorch_model.bin',
                           map_location='cpu', weights_only=True)
    # 새 형식: {'model': state_dict} / 구 형식: state_dict 직접
    state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # ===== 최적화 3: torch.compile 적용 (선택적) =====
    # torch.compile은 첫 추론에 너무 오래 걸려서 WebSocket 타임아웃 유발
    # 프로덕션에서는 사전 워밍업 완료 후 활성화 권장
    USE_TORCH_COMPILE = False
    if USE_TORCH_COMPILE:
        print("🔧 Applying torch.compile (reduce-overhead mode)...")
        try:
            compiled_model = torch.compile(
                model,
                mode="reduce-overhead",  # 지연시간 최소화 모드
                fullgraph=False,  # 동적 shape 허용
                dynamic=False,  # 정적 shape (224x224 고정)
            )
            print("✅ torch.compile applied successfully")
        except Exception as e:
            print(f"⚠️ torch.compile failed, using original model: {e}")
            compiled_model = model
    else:
        compiled_model = model
        print("ℹ️ Using original model (torch.compile disabled)")

    # ===== 최적화 4: Pinned Memory 할당 =====
    if torch.cuda.is_available():
        # CPU→GPU 전송 최적화용 고정 메모리
        pinned_buffer = torch.empty(
            BATCH_SIZE, 3, 30, 224, 224,
            dtype=torch.float32,
            pin_memory=True
        )
        print("✅ Pinned memory buffer allocated")

    # GPU 워밍업 - 최대 배치로 더미 추론 (compiled_model 사용)
    print("🔥 Warming up compiled model with max batch...")
    with torch.no_grad():
        dummy_input = torch.randn(BATCH_SIZE, 3, 30, 224, 224, device=device)
        # torch.compile 워밍업 (첫 몇 번은 컴파일 오버헤드)
        for i in range(10):  # 충분히 워밍업
            _ = compiled_model(dummy_input)
        torch.cuda.synchronize()
    print("✅ Model warmup complete")

    # GPU 메모리 사전 확보 - 실제 추론을 여러 번 수행해서 메모리 풀 확장
    print("Pre-allocating GPU memory for max throughput...")

    # 실제 max batch 추론을 여러 번 수행하여 CUDA 메모리 풀 확장
    with torch.no_grad():
        for i in range(10):
            test_input = torch.randn(BATCH_SIZE, 3, 30, 224, 224, device=device)
            _ = compiled_model(test_input)
        torch.cuda.synchronize()  # 마지막에만 동기화

    # 메모리 상태 출력
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU Memory - Current: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Peak: {max_allocated:.2f} GB")

    # 입력 버퍼 유지 (GC 방지)
    preallocated_buffer = {
        'input': torch.zeros(BATCH_SIZE, 3, 30, 224, 224, device=device),
    }
    torch.cuda.synchronize()

    print(f"Model ready! Max batch size: {BATCH_SIZE}")
    return model

# FastAPI 앱
app = FastAPI(title="Driver Behavior Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 모델
class InferRequest(BaseModel):
    session_id: str
    image: str

class SignUpRequest(BaseModel):
    id: str
    name: str
    password: str
    address: Optional[str] = None
    region_name: Optional[str] = "전국 공통"
    region_campaign: Optional[str] = "대한민국 안전운전 챌린지"
    region_target: Optional[int] = 90
    region_reward: Optional[str] = "안전운전 인증서 발급"

class LoginRequest(BaseModel):
    id: str
    password: str

class StartDrivingRequest(BaseModel):
    user_id: str

class EndDrivingRequest(BaseModel):
    driving_log_id: int

class SaveDetectionRequest(BaseModel):
    driving_log_id: int
    class_id: int
    class_name: str
    confidence: float

# 활성 운전 세션 매핑 (session_id -> driving_log_id)
active_driving_logs: Dict[str, int] = {}

def preprocess_image(base64_image: str) -> np.ndarray:
    """Base64 이미지 디코딩 + 리사이즈 (정규화는 GPU에서)"""
    import cv2

    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]

    image_bytes = base64.b64decode(base64_image)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    if img.shape[:2] != (224, 224):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  # uint8 [224, 224, 3] - 정규화는 GPU에서

def run_batch_inference(jobs: List[InferenceJob]) -> List[Dict[str, Any]]:
    """배치 추론 실행 - 최적화 버전 (CUDA Streams + Pinned Memory + torch.compile)"""
    global compiled_model, device, gpu_mean, gpu_std, inference_stream, transfer_stream

    if not jobs:
        return []

    batch_size = len(jobs)

    # ===== 최적화: NumPy 연산 벡터화 =====
    # 모든 프레임을 한 번에 스택 (리스트 컴프리헨션 대신)
    all_frames = []
    for job in jobs:
        # [30, 224, 224, 3] → [3, 30, 224, 224]
        frames_array = np.stack(job.frames, axis=0).transpose(3, 0, 1, 2)
        all_frames.append(frames_array)

    # 단일 numpy 배열로 합치기
    batch_array = np.stack(all_frames, axis=0).astype(np.float32)  # [B, 3, 30, 224, 224]
    batch_array /= 255.0  # CPU에서 스케일링 (GPU 연산 줄이기)

    # ===== 최적화: CUDA Stream으로 비동기 전송 =====
    if transfer_stream is not None:
        with torch.cuda.stream(transfer_stream):
            # Pinned memory → GPU 비동기 전송
            input_tensor = torch.from_numpy(batch_array).pin_memory().to(device, non_blocking=True)
            # 정규화 (GPU에서)
            input_tensor = (input_tensor - gpu_mean.unsqueeze(2)) / gpu_std.unsqueeze(2)

        # 추론 스트림에서 전송 완료 대기 후 추론
        if inference_stream is not None:
            inference_stream.wait_stream(transfer_stream)
    else:
        # Fallback: 동기 전송
        input_tensor = torch.from_numpy(batch_array).to(device, dtype=torch.float32)
        input_tensor = (input_tensor - gpu_mean.unsqueeze(2)) / gpu_std.unsqueeze(2)

    # ===== 최적화: Mixed Precision + 추론 스트림 =====
    if inference_stream is not None:
        with torch.cuda.stream(inference_stream):
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):  # Mixed Precision
                    output = compiled_model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
        # 결과 동기화
        inference_stream.synchronize()
    else:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):  # Mixed Precision
                output = compiled_model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

    # ===== 결과 추출 (GPU→CPU는 한 번에) =====
    pred_classes_cpu = predicted_classes.cpu().numpy()
    probs_cpu = probabilities.cpu().numpy()

    results = []
    for i in range(batch_size):
        pred_class = int(pred_classes_cpu[i])
        confidence = float(probs_cpu[i][pred_class])
        results.append({
            "class_id": pred_class,
            "class_name": CLASS_NAMES[pred_class],
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                CLASS_NAMES[j]: round(float(probs_cpu[i][j]) * 100, 2)
                for j in range(5)
            }
        })

    return results

# GPU 상시 대기 배치 워커
def gpu_batch_worker():
    """GPU에서 상시 대기하며 요청 즉시 배치 처리"""
    print("GPU batch worker started - waiting for requests...", flush=True)

    while True:
        jobs = []
        start_time = time.time()

        # 첫 번째 작업 대기 (블로킹)
        try:
            first_job = inference_queue.get(timeout=1.0)
            jobs.append(first_job)
            print(f"📥 Got first job for session {first_job.session_id[:8]}, queue size now: {inference_queue.qsize()}", flush=True)
        except Empty:
            continue  # 타임아웃, 다시 대기

        # 배치 채우기 (BATCH_TIMEOUT 내에 더 모으기)
        while len(jobs) < BATCH_SIZE:
            elapsed = time.time() - start_time
            remaining = BATCH_TIMEOUT - elapsed

            if remaining <= 0:
                break

            try:
                job = inference_queue.get(timeout=remaining)
                jobs.append(job)
            except Empty:
                break

        # 배치 추론 실행
        print(f"🔄 Processing batch of {len(jobs)} jobs...", flush=True)
        if jobs:
            try:
                inference_start = time.time()
                results = run_batch_inference(jobs)
                inference_time = (time.time() - inference_start) * 1000
                print(f"✅ Batch inference complete in {inference_time:.0f}ms", flush=True)

                # 추론 결과 로그 출력
                for i, result in enumerate(results):
                    print(f"🎯 추론결과 [{jobs[i].session_id[:8]}] {result['class_name']} ({result['confidence']:.1f}%) | 배치:{len(jobs)} | {inference_time:.0f}ms")

                # ===== 최적화: 디버그 저장 비동기 처리 (추론 블로킹 방지) =====
                if SAVE_DEBUG_FRAMES:
                    def save_debug_frames_async(jobs_copy, results_copy, inference_time_copy):
                        """별도 스레드에서 디버그 프레임 저장"""
                        import cv2
                        for i, result in enumerate(results_copy):
                            timestamp_str = time.strftime("%H%M%S")
                            session_short = jobs_copy[i].session_id[:8]
                            frames = jobs_copy[i].frames

                            # 첫번째, 중간, 마지막 프레임 저장
                            for frame_idx in [0, 14, 29]:
                                if frame_idx < len(frames):
                                    frame = frames[frame_idx]
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    filename = f"{DEBUG_DIR}/{timestamp_str}_{session_short}_f{frame_idx}_{result['class_name']}.jpg"
                                    cv2.imwrite(filename, frame_bgr)

                            # 추론 기록 저장
                            inference_record = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "session_id": session_short,
                                "class_id": result['class_id'],
                                "class_name": result['class_name'],
                                "confidence": result['confidence'],
                                "inference_time_ms": round(inference_time_copy, 1),
                                "frame_count": len(frames),
                                "frame_shape": str(frames[0].shape) if frames else "N/A"
                            }
                            inference_history.append(inference_record)
                            if len(inference_history) > 100:
                                inference_history.pop(0)

                    # 비동기 실행 (추론 스레드 블로킹 없음)
                    debug_executor.submit(save_debug_frames_async, jobs.copy(), results.copy(), inference_time)

                # 결과 전송
                for i, job in enumerate(jobs):
                    result_data = {
                        "status": "inference_complete",
                        "session_id": job.session_id,
                        "result": results[i],
                        "batch_size": len(jobs),
                        "latency_ms": round((time.time() - job.timestamp) * 1000, 1)
                    }

                    # WebSocket으로 전송 (메인 이벤트 루프 사용)
                    if job.websocket and main_event_loop:
                        try:
                            # WebSocket 연결 상태 확인
                            if hasattr(job.websocket, 'client_state'):
                                from starlette.websockets import WebSocketState
                                if job.websocket.client_state != WebSocketState.CONNECTED:
                                    print(f"⚠️ WebSocket not connected [{job.session_id[:8]}], skipping send")
                                    results_store[job.session_id] = result_data
                                    continue

                            asyncio.run_coroutine_threadsafe(
                                job.websocket.send_json(result_data),
                                main_event_loop
                            ).result(timeout=2.0)  # 타임아웃 증가
                        except Exception as ws_error:
                            print(f"⚠️ WebSocket send failed [{job.session_id[:8]}]: {type(ws_error).__name__}")
                            # 전송 실패 시 HTTP 폴링용으로 저장
                            results_store[job.session_id] = result_data

                    # HTTP 폴링용 저장 (항상 저장)
                    results_store[job.session_id] = result_data

            except Exception as e:
                import traceback
                print(f"❌ Batch inference error: {e}", flush=True)
                traceback.print_exc()

# 프레임 수집 및 큐 추가
def add_frame_to_session(session_id: str, frame: np.ndarray, websocket=None):
    """프레임 추가 및 60프레임 버퍼에서 최신 30프레임으로 추론"""
    with sessions_lock:
        if session_id not in user_sessions:
            user_sessions[session_id] = {
                'frames': [],
                'last_active': time.time(),
                'websocket': websocket
            }

        session = user_sessions[session_id]
        session['frames'].append(frame)
        session['last_active'] = time.time()
        if websocket:
            session['websocket'] = websocket

        buffer_size = len(session['frames'])

        # 60프레임 버퍼가 차면 최신 30프레임으로 추론
        if buffer_size >= FRAME_BUFFER_SIZE:
            job = InferenceJob(
                session_id=session_id,
                frames=session['frames'][-FRAMES_PER_INFERENCE:],  # 최신 30프레임
                websocket=session.get('websocket'),
                timestamp=time.time()
            )
            inference_queue.put(job)

            # 10프레임 시프트 (추론당 33% 새 데이터)
            session['frames'] = session['frames'][FRAME_SHIFT:]

            return {
                "status": "queued",
                "buffer_size": len(session['frames']),
                "queue_size": inference_queue.qsize()
            }

        return {
            "status": "buffering",
            "buffer_size": buffer_size,
            "frames_needed": FRAMES_PER_INFERENCE - buffer_size
        }

@app.on_event("startup")
async def startup_event():
    """서버 시작"""
    global main_event_loop

    # 메인 이벤트 루프 저장 (스레드간 통신용)
    main_event_loop = asyncio.get_event_loop()

    init_database()
    load_model()

    # GPU 배치 워커 스레드 시작
    worker_thread = threading.Thread(target=gpu_batch_worker, daemon=True)
    worker_thread.start()

@app.get("/health")
async def health_check():
    with sessions_lock:
        active_sessions = len(user_sessions)
        total_frames = sum(len(s['frames']) for s in user_sessions.values())

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "active_sessions": active_sessions,
        "total_buffered_frames": total_frames,
        "inference_queue_size": inference_queue.qsize(),
        "batch_size": BATCH_SIZE,
        "batch_timeout_ms": BATCH_TIMEOUT * 1000
    }

# ==================== 디버그 API ====================

@app.get("/debug/inference")
async def debug_inference():
    """최근 추론 기록 조회"""
    import glob

    # 저장된 디버그 이미지 목록
    debug_images = sorted(glob.glob(f"{DEBUG_DIR}/*.jpg"), key=os.path.getmtime, reverse=True)[:30]
    image_files = [os.path.basename(f) for f in debug_images]

    return {
        "total_inferences": len(inference_history),
        "recent_inferences": inference_history[-20:],  # 최근 20개
        "debug_images": image_files,
        "debug_dir": DEBUG_DIR,
        "save_enabled": SAVE_DEBUG_FRAMES
    }

@app.get("/debug/frame/{filename}")
async def get_debug_frame(filename: str):
    """디버그 프레임 이미지 조회"""
    file_path = os.path.join(DEBUG_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Frame not found")

@app.delete("/debug/clear")
async def clear_debug():
    """디버그 데이터 초기화"""
    import glob
    for f in glob.glob(f"{DEBUG_DIR}/*.jpg"):
        os.remove(f)
    inference_history.clear()
    return {"status": "cleared", "message": "Debug data cleared"}

# ==================== 인증 API ====================

@app.post("/auth/signup")
async def signup(request: SignUpRequest):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT id FROM users WHERE id = ?', (request.id,))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다")

        cursor.execute('''
            INSERT INTO users (id, name, password, address, region_name, region_campaign,
                             region_target, region_reward, score, discount_rate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.id, request.name, request.password, request.address,
            request.region_name, request.region_campaign, request.region_target,
            request.region_reward, 80, 0, time.strftime('%Y-%m-%d %H:%M:%S')
        ))

        conn.commit()
        conn.close()
        return {"success": True, "message": "회원가입이 완료되었습니다"}

    except sqlite3.Error as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

@app.post("/auth/login")
async def login(request: LoginRequest):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT * FROM users WHERE id = ?', (request.id,))
        user = cursor.fetchone()
        conn.close()

        if not user or user['password'] != request.password:
            raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 틀렸습니다")

        return {
            "success": True,
            "user": {
                "id": user['id'],
                "name": user['name'],
                "score": user['score'],
                "discount_rate": user['discount_rate'],
                "region": {
                    "name": user['region_name'],
                    "campaign": user['region_campaign'],
                    "target": user['region_target'],
                    "reward": user['region_reward'],
                    "address": user['address']
                }
            }
        }

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

@app.get("/auth/user/{user_id}")
async def get_user(user_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")

    return {
        "id": user['id'],
        "name": user['name'],
        "score": user['score'],
        "discount_rate": user['discount_rate'],
        "region": {
            "name": user['region_name'],
            "campaign": user['region_campaign'],
            "target": user['region_target'],
            "reward": user['region_reward'],
            "address": user['address']
        }
    }

# ==================== 운전 기록 API ====================

@app.post("/driving/start")
async def start_driving(request: StartDrivingRequest):
    """운전 시작 - 새 운전 기록 생성"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        start_time = time.strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO driving_logs (user_id, start_time, status)
            VALUES (?, ?, 'driving')
        ''', (request.user_id, start_time))

        conn.commit()
        driving_log_id = cursor.lastrowid
        conn.close()

        return {
            "success": True,
            "driving_log_id": driving_log_id,
            "start_time": start_time
        }

    except sqlite3.Error as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

@app.post("/driving/end")
async def end_driving(request: EndDrivingRequest):
    """운전 종료 - 운전 기록 업데이트"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        end_time = time.strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            UPDATE driving_logs
            SET end_time = ?, status = 'completed'
            WHERE id = ?
        ''', (end_time, request.driving_log_id))

        conn.commit()
        conn.close()

        return {
            "success": True,
            "driving_log_id": request.driving_log_id,
            "end_time": end_time
        }

    except sqlite3.Error as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

@app.post("/driving/detection")
async def save_detection(request: SaveDetectionRequest):
    """모델 감지 결과 저장"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        detected_at = time.strftime('%Y-%m-%d %H:%M:%S')

        # 감지 기록 저장
        cursor.execute('''
            INSERT INTO driving_detections (driving_log_id, detected_at, class_id, class_name, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (request.driving_log_id, detected_at, request.class_id, request.class_name, request.confidence))

        # 운전 기록의 카운트 업데이트
        count_column = {
            0: 'normal_count',
            1: 'drowsy_count',
            2: 'searching_count',
            3: 'phone_count',
            4: 'assault_count'
        }.get(request.class_id, 'normal_count')

        cursor.execute(f'''
            UPDATE driving_logs
            SET total_detections = total_detections + 1,
                {count_column} = {count_column} + 1
            WHERE id = ?
        ''', (request.driving_log_id,))

        conn.commit()
        conn.close()

        return {"success": True, "detected_at": detected_at}

    except sqlite3.Error as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

@app.get("/driving/logs/{user_id}")
async def get_driving_logs(user_id: str):
    """사용자의 모든 운전 기록 조회"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM driving_logs
        WHERE user_id = ?
        ORDER BY start_time DESC
    ''', (user_id,))

    logs = cursor.fetchall()
    conn.close()

    return {
        "logs": [
            {
                "id": log['id'],
                "start_time": log['start_time'],
                "end_time": log['end_time'],
                "status": log['status'],
                "total_detections": log['total_detections'],
                "normal_count": log['normal_count'],
                "drowsy_count": log['drowsy_count'],
                "searching_count": log['searching_count'],
                "phone_count": log['phone_count'],
                "assault_count": log['assault_count']
            }
            for log in logs
        ]
    }

@app.get("/driving/log/{driving_log_id}")
async def get_driving_log_detail(driving_log_id: int):
    """특정 운전 기록 상세 조회 (감지 기록 포함)"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 운전 기록 조회
    cursor.execute('SELECT * FROM driving_logs WHERE id = ?', (driving_log_id,))
    log = cursor.fetchone()

    if not log:
        conn.close()
        raise HTTPException(status_code=404, detail="운전 기록을 찾을 수 없습니다")

    # 감지 기록 조회
    cursor.execute('''
        SELECT * FROM driving_detections
        WHERE driving_log_id = ?
        ORDER BY detected_at ASC
    ''', (driving_log_id,))

    detections = cursor.fetchall()
    conn.close()

    return {
        "log": {
            "id": log['id'],
            "user_id": log['user_id'],
            "start_time": log['start_time'],
            "end_time": log['end_time'],
            "status": log['status'],
            "total_detections": log['total_detections'],
            "normal_count": log['normal_count'],
            "drowsy_count": log['drowsy_count'],
            "searching_count": log['searching_count'],
            "phone_count": log['phone_count'],
            "assault_count": log['assault_count']
        },
        "detections": [
            {
                "id": det['id'],
                "detected_at": det['detected_at'],
                "class_id": det['class_id'],
                "class_name": det['class_name'],
                "confidence": det['confidence']
            }
            for det in detections
        ]
    }

# ==================== 세션/추론 API ====================

@app.post("/session/create")
async def create_session():
    session_id = str(uuid.uuid4())
    with sessions_lock:
        user_sessions[session_id] = {
            'frames': [],
            'last_active': time.time(),
            'websocket': None
        }
    return {"session_id": session_id}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    with sessions_lock:
        if session_id in user_sessions:
            del user_sessions[session_id]
            return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/infer")
async def infer(request: InferRequest):
    """프레임 추가 - 30프레임 도달시 즉시 큐에 추가"""
    try:
        frame = preprocess_image(request.image)
        result = add_frame_to_session(request.session_id, frame)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{session_id}")
async def get_result(session_id: str):
    """추론 결과 폴링 (HTTP용)"""
    if session_id in results_store:
        return results_store.pop(session_id)
    return {"status": "pending"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 실시간 통신 - keep-alive 강화"""
    await websocket.accept()
    print(f"✅ WebSocket connected: {session_id[:8]}")

    with sessions_lock:
        if session_id not in user_sessions:
            user_sessions[session_id] = {
                'frames': [],
                'last_active': time.time(),
                'websocket': websocket
            }
        else:
            user_sessions[session_id]['websocket'] = websocket

    # 서버 측 ping 태스크 (keep-alive) - RunPod 프록시 타임아웃 방지
    async def server_ping():
        """2초마다 ping 전송하여 프록시 타임아웃 방지 (즉시 시작)"""
        ping_count = 0
        try:
            # 즉시 첫 ping 전송 (연결 직후)
            await websocket.send_json({"type": "server_ping", "timestamp": time.time()})
            ping_count += 1
            print(f"🏓 Ping #{ping_count} sent [{session_id[:8]}]")

            while True:
                await asyncio.sleep(2)  # 5초 -> 2초로 단축
                try:
                    await websocket.send_json({"type": "server_ping", "timestamp": time.time()})
                    ping_count += 1
                    if ping_count % 10 == 0:  # 20초마다 로그
                        print(f"🏓 Ping #{ping_count} sent [{session_id[:8]}]")
                except Exception as e:
                    print(f"⚠️ Ping failed [{session_id[:8]}]: {e}")
                    break
        except asyncio.CancelledError:
            pass

    ping_task = asyncio.create_task(server_ping())

    try:
        while True:
            try:
                # 타임아웃 설정으로 무한 대기 방지
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                # 30초간 데이터 없으면 ping 전송
                await websocket.send_json({"type": "server_ping", "timestamp": time.time()})
                continue

            if data.get('type') == 'frame':
                try:
                    frame = preprocess_image(data['image'])
                    result = add_frame_to_session(session_id, frame, websocket)
                    await websocket.send_json(result)
                except Exception as e:
                    print(f"⚠️ Frame processing error [{session_id[:8]}]: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})

            elif data.get('type') == 'ping':
                await websocket.send_json({"type": "pong", "timestamp": time.time()})

            elif data.get('type') == 'pong':
                # 클라이언트 pong 응답 무시 (keep-alive 확인용)
                pass

    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected (client): {session_id[:8]}")
    except Exception as e:
        print(f"❌ WebSocket error [{session_id[:8]}]: {type(e).__name__}: {e}")
    finally:
        ping_task.cancel()
        with sessions_lock:
            if session_id in user_sessions:
                user_sessions[session_id]['websocket'] = None
        print(f"🔚 WebSocket cleanup done: {session_id[:8]}")

# 세션 정리
async def cleanup_sessions():
    while True:
        await asyncio.sleep(60)
        current_time = time.time()
        with sessions_lock:
            expired = [sid for sid, s in user_sessions.items() if current_time - s['last_active'] > 300]
            for sid in expired:
                del user_sessions[sid]
                if sid in results_store:
                    del results_store[sid]

@app.on_event("startup")
async def start_cleanup():
    asyncio.create_task(cleanup_sessions())

# ==================== 프론트엔드 정적 파일 서빙 ====================

FRONTEND_DIR = "/workspace/ai-2026-c-team/driver_front/dist"

# 정적 파일이 존재하면 마운트
if os.path.exists(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=f"{FRONTEND_DIR}/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """SPA fallback - 모든 경로를 index.html로"""
        # API 경로는 제외
        if full_path.startswith(("auth/", "driving/", "session/", "ws/", "health", "infer", "result/")):
            raise HTTPException(status_code=404)

        # 정적 파일 확인
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)

        # SPA fallback
        return FileResponse(f"{FRONTEND_DIR}/index.html")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,  # 프론트엔드 프록시와 일치
        ws_ping_interval=10,   # 20 -> 10초 (더 빠른 ping)
        ws_ping_timeout=20,    # 30 -> 20초 (더 빠른 감지)
        timeout_keep_alive=300 # 120 -> 300초 (keep-alive 연장)
    )
