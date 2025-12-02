# 🐾 AI-Generated vs Real Animal Image Classifier

**딥러닝 기반의 AI 생성 동물 이미지 vs 실제 동물 이미지 이진 분류**

This repository implements a full pipeline to classify **AI-generated** animal images versus **real** images using modern deep learning models (ResNet50, EfficientNet-B0, ViT).
It also includes **synthetic image generation**, **feature visualization**, and **dataset processing utilities**.

---

# 🗂️ Project Structure

```plaintext
AI-generated-animal-image-detector/
│
├── main.py                     # 메인 실행 스크립트
│
├── data_loader.py              # Dataset & DataLoader 생성
├── model_builder.py            # Model / Optimizer builder
├── trainer.py                  # Training + Validation + Model saving
├── evaluate.py                 # Test evaluation + CSV 기록
│
├── visualize_features.py       # Grad-CAM/activation 기반 feature 시각화
│
├── generator/
│     ├── diffusion_generator.py    # Stable Diffusion (Turbo/XL/v1.5) 이미지 생성
│     └── gan.py                    # GAN 기반 이미지 생성 실험 모듈
├── preprocessing/
│     ├── animalize.py             # WordNet 기반 species filtering
│     └── show_data.py             # CSV 정보 시각화/분석
│
├── utils/
│     ├── metrics.py                 # Accuracy/Precision/Recall/F1/ROC-AUC
│     └── dataset_reorganize_script.py
│
├── requirements.txt                 # environment info
└── results/                         # Saved model and metric results by each model combinations
```

---

# 📁 Dataset Structure

```plaintext
dataset/
│
├── train/
│     ├── real/     # real_image_0000.png …
│     └── ai/       # ai_image_0000.png …
│
├── val/ # Same structure as 'train/'
└── test/ # Same structure as 'train/'
```

---

# ⚙️ Environment

* Python 3.10+
* PyTorch 2.7.1 (cu118)
* Torchvision 0.22.1
* timm 1.0.21
* diffusers 0.35+
* transformers 4.57+
* scikit-learn 1.7.2
* pandas 2.3.3
* tqdm 4.67.1
* python-dotenv 1.1.1
* Pillow 12.0.0
* 자세한 내용은 requirements.txt

---

```bash
export DATA_PATH=/path/to/dataset
export SAVE_PATH=/path/to/save_dir

python main.py
```
- DATA_PATH
  - train/, val/, test/ directory를 포함한 dataset 경로
- SAVE_PATH
  - 학습된 model과 결과 log가 저장될 directory

### main.py Process

1. Model × Optimizer × Hyperparameter(Learning rate, epoch) combination 자동 생성
2. If the saved model(.pth) already exists, skip 3-4.
3. Dataset load -> train -> validate
4. Model save(.pth)

---

# Dataset Details

### Real Images

* Source: Kaggle
* 8 classes: *elephant, cow, sheep, dog, cat, chicken, horse, rabbit*
* 2,000 images per class → Total 16,000
* 70/15/15 split

### AI-Generated Images

* Model: Stable Diffusion Turbo
* Same 8 classes, 2,000 images each
* Same split ratio

---

# File-by-File Explanation

---

## visualize_features.py

Visualize where the model focuses when making a decision

### Features

* Grad-CAM heatmap create
* Feature map / activation visualize
* ViT : attention analyze
* Save images(.png)

---

## gan.py

AI image generator with GAN(StyleGAN2)

### Features

* Generate 3 classes image with pretrained GAN network (dog, cat, wild)
* Using StyleGAN2

---

## trainer.py

* train → validate loop
* Early saving
* Epoch metric logging
* Uses evaluate_val_acc for best checkpoint selection

## data_loader.py

* ImageFolder 기반 로딩
* Resize(224×224) → ToTensor → Normalize
* Train/Val/Test DataLoader 생성

## model_builder.py

* Builds models: ResNet50, EfficientNet-B0, ViT
* Loads pretrained weights
* Replaces classification head → 2 outputs (real vs ai)
* Creates optimizer according to settings

## evaluate.py

* Loads saved .pth weights
* Computes metrics via utils.metrics
* Appends results to CSV
* Parses model naming structure (strict format — keep consistent)

## utils/metrics.py

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC

## utils/dataset_reorganize_script.py

* Reorganizes dataset folders
* Renames files
* Handles AI/real unclassification
* Maybe it does not useful for you (Only for my local computer)

## diffusion_generator.py

* Generates images using Stable Diffusion pipelines
* Includes optional SD 1.5 + SDXL (commented)
* Active: SD-Turbo for fast generation
* Splits train/val/test automatically

---

# Important Notes / Gotchas

* diffusion_generator.py requires:

  * GPU
  * valid HuggingFace tokens (if downloading SD models)

---

# 👨‍💻 Author

**Seung-hyeon Lee (이승현)**

---
