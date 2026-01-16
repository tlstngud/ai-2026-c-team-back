"""
운전자 이상행동 감지 모델

- 백본: TorchVision Video Swin-T (Kinetics-400 사전학습)
- 입력: [B, 3, 30, 224, 224] (배치, 채널, 프레임, 높이, 너비)
- 출력: 5클래스 분류 (정상, 졸음운전, 물건찾기, 휴대폰 사용, 운전자 폭행)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from typing import Dict, Optional


class DriverBehaviorModel(nn.Module):
    """
    운전자 이상행동 감지 모델

    Args:
        num_classes: 출력 클래스 수 (기본값: 5, 전체 버전)
        pretrained: Kinetics-400 사전학습 가중치 사용 여부
        freeze_backbone: 백본 파라미터 동결 여부 (전이학습 시)
    """

    # 전체 5클래스
    CLASS_NAMES = ["정상", "졸음운전", "물건찾기", "휴대폰 사용", "운전자 폭행"]

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes

        # TorchVision Video Swin-T 백본 로드
        if pretrained:
            print("Loading Kinetics-400 pretrained weights...")
            self.backbone = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        else:
            self.backbone = swin3d_t(weights=None)

        # 원본 head 교체 (Kinetics-400: 400클래스 → 5클래스)
        # swin3d_t의 head는 nn.Linear(768, 400)
        in_features = self.backbone.head.in_features  # 768
        self.backbone.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.3),  # 오버피팅 방지
            nn.Linear(in_features, num_classes),
        )

        # 백본 동결 옵션
        if freeze_backbone:
            self._freeze_backbone()

        # Head 가중치 초기화
        self._init_head()

    def _freeze_backbone(self):
        """백본 파라미터 동결 (head 제외)"""
        for name, param in self.backbone.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        print("Backbone parameters frozen (head trainable)")

    def _init_head(self):
        """Head 가중치 초기화"""
        for m in self.backbone.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: [B, C, T, H, W] 형태의 비디오 텐서
               - B: 배치 크기
               - C: 채널 (3)
               - T: 프레임 수 (30)
               - H, W: 높이, 너비 (224, 224)

        Returns:
            logits: [B, num_classes] 형태의 로짓
        """
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> Dict:
        """
        추론용 예측 (단일 샘플)

        Args:
            x: [1, 3, 30, 224, 224] 형태의 비디오 텐서

        Returns:
            {
                "class": int (0~4),
                "confidence": float (0~1),
                "class_name": str
            }
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)[0]

            class_idx = probs.argmax().item()
            confidence = probs[class_idx].item()

            return {
                "class": class_idx,
                "confidence": confidence,
                "class_name": self.CLASS_NAMES[class_idx],
            }

    def get_all_probs(self, x: torch.Tensor) -> Dict:
        """
        모든 클래스의 확률 반환

        Args:
            x: [1, 3, 30, 224, 224] 형태의 비디오 텐서

        Returns:
            {
                "predictions": [{"class": int, "class_name": str, "probability": float}, ...],
                "top_class": int,
                "top_confidence": float
            }
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)[0]

            predictions = []
            for i, prob in enumerate(probs):
                predictions.append({
                    "class": i,
                    "class_name": self.CLASS_NAMES[i],
                    "probability": prob.item(),
                })

            # 확률 내림차순 정렬
            predictions.sort(key=lambda x: x["probability"], reverse=True)

            return {
                "predictions": predictions,
                "top_class": predictions[0]["class"],
                "top_confidence": predictions[0]["probability"],
            }


def create_model(
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    checkpoint_path: Optional[str] = None,
) -> DriverBehaviorModel:
    """
    모델 생성 헬퍼 함수

    Args:
        num_classes: 출력 클래스 수
        pretrained: 사전학습 가중치 사용 여부
        freeze_backbone: 백본 동결 여부
        checkpoint_path: 체크포인트 경로 (학습된 가중치 로드)

    Returns:
        DriverBehaviorModel 인스턴스
    """
    model = DriverBehaviorModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        print("Checkpoint loaded successfully")

    return model


if __name__ == "__main__":
    # 모델 테스트
    print("=" * 60)
    print("Model Test (3 classes - Demo)")
    print("=" * 60)

    # 모델 생성
    model = DriverBehaviorModel(num_classes=5, pretrained=True)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 더미 입력으로 테스트
    dummy_input = torch.randn(2, 3, 30, 224, 224)
    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # 단일 샘플 예측 테스트
    single_input = torch.randn(1, 3, 30, 224, 224)
    prediction = model.predict(single_input)
    print(f"\nPrediction: {prediction}")

    # 모든 확률 출력 테스트
    all_probs = model.get_all_probs(single_input)
    print(f"\nAll probabilities:")
    for pred in all_probs["predictions"]:
        print(f"  {pred['class_name']}: {pred['probability']:.4f}")

    print("\nModel test passed!")
