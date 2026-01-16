# Driver Monitoring Backend

GPU 기반 운전자 모니터링 AI 추론 서버

## 프로젝트 구조

```
backend/
├── backend_server.py          # FastAPI 메인 서버
├── Driver_monitoring/
│   ├── model.py               # Video Swin-T 모델 정의
│   ├── config.json            # 모델 설정
│   └── pytorch_model.bin      # 학습된 가중치 (별도 다운로드)
├── requirements.txt           # Python 의존성
└── README.md
```

## 모델 정보

- **모델**: Video Swin Transformer (Tiny)
- **백본**: TorchVision swin3d_t (Kinetics-400 사전학습)
- **입력**: [B, 3, 30, 224, 224] - 배치, 채널, 30프레임, 224x224
- **출력**: 5개 클래스 분류
  - 0: 정상 (Normal)
  - 1: 졸음운전 (Drowsy)
  - 2: 물건찾기 (Searching)
  - 3: 휴대폰 사용 (Phone)
  - 4: 운전자 폭행 (Assault)

### 모델 성능 (config.json 참조)

| 클래스 | F1 Score |
|--------|----------|
| 정상 | 0.92 |
| 졸음운전 | 0.98 |
| 물건찾기 | 0.92 |
| 휴대폰 사용 | 0.90 |
| 운전자 폭행 | 1.00 |
| **전체 Accuracy** | **95.51%** |

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 가중치 다운로드

`pytorch_model.bin` 파일은 126MB로 GitHub에 포함되지 않습니다.
HuggingFace에서 다운로드하세요:

```bash
# HuggingFace에서 다운로드
cd Driver_monitoring
wget https://huggingface.co/your-model-repo/resolve/main/pytorch_model.bin

# 또는 팀 공유 드라이브에서 다운로드
```

### 3. 서버 실행

```bash
python backend_server.py
```

서버가 `http://0.0.0.0:8000`에서 실행됩니다.

## API 엔드포인트

### 인증

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/auth/signup` | POST | 회원가입 |
| `/auth/login` | POST | 로그인 |

### 운전 세션

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/driving/start` | POST | 운전 세션 시작 |
| `/driving/end` | POST | 운전 세션 종료 |
| `/driving/logs/{user_id}` | GET | 운전 기록 조회 |

### AI 추론

| 엔드포인트 | 프로토콜 | 설명 |
|-----------|----------|------|
| `/ws/{session_id}` | WebSocket | 실시간 프레임 전송 및 추론 결과 수신 |
| `/health` | GET | 서버 상태 확인 |

## WebSocket 프레임 전송

```javascript
// 클라이언트 예시
const ws = new WebSocket('wss://your-server/ws/session123');

// 프레임 전송 (Base64 인코딩된 JPEG)
ws.send(JSON.stringify({
  frame: base64ImageData,
  timestamp: Date.now()
}));

// 추론 결과 수신
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  // { class_id: 0, class_name: "정상", confidence: 0.95, probabilities: [...] }
};
```

## 환경 요구사항

- **GPU**: NVIDIA RTX 4000 Ada 이상 (20GB VRAM 권장)
- **Python**: 3.11+
- **PyTorch**: 2.0+
- **CUDA**: 12.x

## 배치 처리

- **배치 크기**: 16 (RTX 4000 Ada 기준)
- **프레임 버퍼**: 30프레임 (1초 @ 30fps)
- **배치 타임아웃**: 50ms
