# 종속성 설치(가상환경에 설치 권장)
```
pip install -e .
```

# 기본 예측
python cli.py ./sample_image/platelet_2181.png

# 상위 3개 예측
python cli.py ./sample_image/platelet_2181.png --top-k 3

# 혈액 세포 분류기 (Blood Cell Classifier)

딥러닝 기반 혈액 세포 이미지 자동 분류 모델


## 빠른 시작
```python
from blood_cell_classifier import BloodCellClassifier

# 모델 로드
classifier = BloodCellClassifier('best_model.pth')

# 예측
result = classifier.predict('./sample_image/platelet_2181.png')
print(f"{result['class']}: {result['confidence']:.2f}%")
```

## 주요 기능

### 1. 단일 이미지 예측
```python
result = classifier.predict('./sample_image/platelet_2181.png')
```

### 2. 상위 K개 예측
```python
top3 = classifier.predict_top_k('./sample_image/platelet_2181.png', k=3)
for res in top3:
    print(f"{res['class']}: {res['confidence']:.2f}%")
```

### 3. 모든 확률 보기
```python
print("\n=== 방법 2: 모든 확률 보기 ===")
result = classifier.predict("./sample_image/platelet_2181.png", return_all_probs=True)
print(f"예측: {result['class']} ({result['confidence']:.2f}%)")
print("\n모든 클래스 확률:")
for cls, prob in result["probabilities"].items():
    print(f"  {cls}: {prob:.2f}%")
```

## 지원 클래스

- Basophil (호염기구)
- Eosinophil (호산구)
- Erythroblast (적아구)
- Ig (미성숙 과립구)
- Lymphocyte (림프구)
- Monocyte (단핵구)
- Neutrophil (호중구)
- Platelet (혈소판)

## 성능

- 전체 샘플 정확도: 99.07%
- 추론 속도: ~50ms/image (GPU)

- 검증: 98.5%
- <img width="650" height="730" alt="image" src="https://github.com/user-attachments/assets/b891dae6-930c-4243-9f42-056cac24b6ae" />

  
- 단일 이미지
- <img width="447" height="101" alt="image" src="https://github.com/user-attachments/assets/d3232301-9c27-42d2-9320-977ab6f5a3be" />

