import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import json
import os
from typing import Dict

# 1. 과일 클래스 정의
# train_fruit_model.py에서 학습할 때의 클래스 목록과 반드시 순서까지 같아야 함
FRUIT_CLASSES = ["apple", "banana", "strawberry"]
NUM_CLASSES = len(FRUIT_CLASSES)

_model = None
_FRUIT_META = None


def load_model(weights_path: str = "models/fruit_resnet.pt") -> nn.Module:
    """
    train_fruit_model.py에서 학습한 ResNet18 모델을 로딩.
    - 모델 구조는 학습 때 사용한 것과 완전히 같아야 한다.
      (models.resnet18(weights=None) + 마지막 fc를 NUM_CLASSES로 교체)
    - weights_path에는 model.state_dict()가 저장되어 있어야 한다.
    """
    global _model
    if _model is not None:
        return _model

    # 🔹 학습 스크립트와 동일한 구조로 모델 생성
    model = models.resnet18(weights=None)  # 또는 pretrained=False (버전에 따라)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    # 🔹 fine-tuning된 가중치가 있을 경우 로드
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[cv_model] Loaded fine-tuned weights from {weights_path}")
    else:
        print("[cv_model] No fine-tuned weights found, using randomly initialized ResNet18.")

    model.eval()
    _model = model
    return _model


def load_fruit_meta(path: str = "data/fruit_meta.json") -> Dict:
    """
    과일 메타 정보(한글 이름, 자라는 곳, 수확 직전 이미지 경로, 설명)를 로드.
    """
    global _FRUIT_META
    if _FRUIT_META is not None:
        return _FRUIT_META

    if not os.path.exists(path):
        print(f"[cv_model] WARNING: {path} not found. Using empty meta.")
        _FRUIT_META = {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            _FRUIT_META = json.load(f)
    return _FRUIT_META


# 3. OpenCV 전처리
def preprocess_image(image: np.ndarray, target_size=(224, 224)) -> torch.Tensor:
    """
    image: numpy array (H, W, C), 보통 RGB (Gradio에서 들어오는 형식)
    ResNet18 입력(224x224, RGB, 정규화) 형태로 변환.
    """
    # Gradio에서 들어오는 이미지는 보통 RGB라서, 굳이 BGR->RGB 변환은 필요 없음.
    # 만약 BGR 이미지가 들어오는 환경이면 아래 줄을 사용:
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img = cv2.resize(image, target_size)
    img = img.astype(np.float32) / 255.0  # 0~1 스케일

    # ResNet 입력 정규화 (ImageNet 기준)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # (H, W, C) -> (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0)  # (1, C, H, W)
    return img


# 4. 메인 추론 함수
def predict_fruit(image: np.ndarray) -> Dict:
    """
    image: numpy array (H, W, C)
    return:
        {
          "fruit_eng": "apple",
          "fruit_ko": "사과",
          "grow_type": "tree",
          "pre_harvest_image_path": "...",
          "description": "설명 텍스트"
        }
    """
    model = load_model()
    meta = load_fruit_meta()

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())

    fruit_eng = FRUIT_CLASSES[pred_idx]
    info = meta.get(fruit_eng, None)

    if info is None:
        return {
            "fruit_eng": fruit_eng,
            "fruit_ko": fruit_eng,
            "grow_type": "unknown",
            "pre_harvest_image_path": None,
            "description": f"{fruit_eng}에 대한 메타 정보가 아직 등록되지 않았습니다."
        }

    img_path = info.get("pre_harvest_image_path")
    if img_path is not None and not os.path.isabs(img_path):
        img_path = os.path.join(os.getcwd(), img_path)

    return {
        "fruit_eng": fruit_eng,
        "fruit_ko": info.get("ko_name", fruit_eng),
        "grow_type": info.get("grow_type", "unknown"),
        "pre_harvest_image_path": img_path,
        "description": info.get("description", "")
    }
