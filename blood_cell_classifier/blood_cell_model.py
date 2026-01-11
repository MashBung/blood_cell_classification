import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf
from PIL import Image
import torch.nn.functional as F
from pathlib import Path


class BloodCNN(nn.Module):
    """혈액 세포 분류 CNN 모델"""

    def __init__(self, num_classes=8):
        super().__init__()

        # Stage1
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
        )

        # Stage2
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )

        # Stage3
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )

        # Stage4
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
        )

        # Stage5
        self.stage5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class BloodCellClassifier:
    """
    혈액 세포 이미지 분류기

    사용 예시:
        >>> classifier = BloodCellClassifier('best_model.pth')
        >>> result = classifier.predict('cell_image.png')
        >>> print(result['class'], result['confidence'])
    """

    # 기본 클래스 이름 (학습 시 순서와 동일해야 함)
    DEFAULT_CLASSES = [
        "Basophil",
        "Eosinophil",
        "Erythroblast",
        "Ig",
        "Lymphocyte",
        "Monocyte",
        "Neutrophil",
        "Platelet",
    ]

    # 기본 정규화 값
    MEAN = [0.6475, 0.4899, 0.6431]
    STD = [0.2282, 0.2568, 0.0901]

    def __init__(self, model_path, class_names=None, device=None):
        """
        Args:
            model_path (str): 학습된 모델 파일 경로 (.pth)
            class_names (list, optional): 클래스 이름 리스트
            device (str, optional): 'cuda' 또는 'cpu'
        """
        self.model_path = Path(model_path)
        self.class_names = class_names or self.DEFAULT_CLASSES
        self.num_classes = len(self.class_names)

        # Device 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Transform 설정
        self.transform = tf.Compose(
            [
                tf.ToImage(),
                tf.CenterCrop(128),
                tf.ToDtype(torch.float32, scale=True),
                tf.Normalize(mean=self.MEAN, std=self.STD),
            ]
        )

        # 모델 로드
        self.model = self._load_model()

        print(f"모델 로드 완료")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")

    def _load_model(self):
        """모델 로드 및 초기화"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")

        # 모델 생성
        model = BloodCNN(num_classes=self.num_classes)

        # 가중치 로드
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model

    def predict(self, image_path, return_all_probs=False):
        """
        단일 이미지 예측

        Args:
            image_path (str): 이미지 파일 경로
            return_all_probs (bool): 모든 클래스 확률 반환 여부

        Returns:
            dict: 예측 결과
                - class: 예측된 클래스 이름
                - confidence: 신뢰도 (%)
                - class_idx: 클래스 인덱스
                - probabilities: 모든 클래스 확률 (옵션)
        """
        # 이미지 로드
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

        # 전처리
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 예측
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        # 결과 구성
        result = {
            "class": self.class_names[predicted_idx.item()],
            "confidence": confidence.item() * 100,
            "class_idx": predicted_idx.item(),
        }

        if return_all_probs:
            result["probabilities"] = {
                self.class_names[i]: prob.item() * 100
                for i, prob in enumerate(probabilities)
            }

        return result

    def predict_top_k(self, image_path, k=3):
        """
        상위 K개 예측 결과 반환

        Args:
            image_path (str): 이미지 파일 경로
            k (int): 반환할 상위 예측 개수

        Returns:
            list: 상위 K개 예측 결과
        """
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 예측
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            top_k_probs, top_k_indices = torch.topk(probabilities, k)

        # 결과 구성
        results = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            results.append(
                {
                    "class": self.class_names[idx.item()],
                    "confidence": prob.item() * 100,
                    "class_idx": idx.item(),
                }
            )

        return results

    # 편의 함수들


def load_classifier(model_path, **kwargs):
    """간단한 로더 함수"""
    return BloodCellClassifier(model_path, **kwargs)


def quick_predict(model_path, image_path):
    """한 줄로 예측"""
    classifier = BloodCellClassifier(model_path)
    return classifier.predict(image_path)
