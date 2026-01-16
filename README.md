# Driver Monitoring Backend

GPU 기반 운전자 모니터링 AI 추론 서버

## 모델 정보

- **모델**: Video Swin Transformer (Tiny)
- **입력**: [B, 3, 30, 224, 224] - 배치, 채널, 30프레임, 224x224
- **출력**: 5개 클래스 분류
  - 0: Normal (정상)
  - 1: Drowsy (졸음)
  - 2: Searching (주시태만)
  - 3: Phone (휴대폰 사용)
  - 4: Assault (폭행)

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python backend_server.py
```

## API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /health` | 서버 상태 확인 |
| `POST /auth/signup` | 회원가입 |
| `POST /auth/login` | 로그인 |
| `WS /ws/{session_id}` | WebSocket 프레임 전송 |
| `POST /driving/start` | 운전 세션 시작 |
| `POST /driving/end` | 운전 세션 종료 |

## 환경

- RunPod GPU Pod (RTX 4000 Ada 20GB)
- Python 3.11
- PyTorch 2.0+
- CUDA 12.x
