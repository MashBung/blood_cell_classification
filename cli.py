import argparse
from blood_cell_model import BloodCellClassifier


def main():
    parser = argparse.ArgumentParser(description="혈액 세포 이미지 분류")
    parser.add_argument("image", help="이미지 파일 경로")
    parser.add_argument("--model", default="best_model.pth", help="모델 파일 경로")
    parser.add_argument("--top-k", type=int, default=1, help="상위 K개 예측")

    args = parser.parse_args()

    classifier = BloodCellClassifier(args.model)

    if args.top_k == 1:
        result = classifier.predict(args.image)
        print(f"예측: {result['class']} ({result['confidence']:.2f}%)")
    else:
        results = classifier.predict_top_k(args.image, k=args.top_k)
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['class']}: {r['confidence']:.2f}%")


if __name__ == "__main__":
    main()
