from blood_cell_classifier.blood_cell_model import BloodCellClassifier, quick_predict

# ============================================
# 방법 1: 기본 사용
# ============================================
print("=== 방법 1: 기본 사용 ===")
classifier = BloodCellClassifier("best_model.pth")

result = classifier.predict("./sample_image/platelet_2181.png")
print(f"예측: {result['class']}")
print(f"신뢰도: {result['confidence']:.2f}%")

# ============================================
# 방법 2: 모든 확률 보기
# ============================================
print("\n=== 방법 2: 모든 확률 보기 ===")
result = classifier.predict("./sample_image/platelet_2181.png", return_all_probs=True)
print(f"예측: {result['class']} ({result['confidence']:.2f}%)")
print("\n모든 클래스 확률:")
for cls, prob in result["probabilities"].items():
    print(f"  {cls}: {prob:.2f}%")

# ============================================
# 방법 3: 상위 3개 예측
# ============================================
print("\n=== 방법 3: 상위 3개 예측 ===")
top3 = classifier.predict_top_k("./sample_image/platelet_2181.png", k=3)
for i, pred in enumerate(top3, 1):
    print(f"{i}. {pred['class']}: {pred['confidence']:.2f}%")

# ============================================
# 방법 5: 한 줄로 예측
# ============================================
print("\n=== 방법 5: 한 줄로 예측 ===")
result = quick_predict("best_model.pth", "./sample_image/platelet_2181.png")
print(f"{result['class']}: {result['confidence']:.2f}%")
