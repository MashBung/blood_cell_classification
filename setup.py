from setuptools import setup, find_packages

setup(
    name="blood-cell-classifier",
    version="1.0.0",
    description="혈액 세포 이미지 분류 모델",
    author="RYU",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
    ],
    python_requires=">=3.8",
)
