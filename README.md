# 🐾 AI-Generated vs Real Animal Image Classifier

**딥러닝 기반의 AI 생성 동물 이미지 vs 실제 동물 이미지 이진 분류 프로젝트**

This repository implements a full pipeline to classify **AI-generated** animal images versus **real** images using modern deep learning models (ResNet50, EfficientNet-B0, ViT).
It also includes **synthetic image generation**, **feature visualization**, and **dataset processing utilities**.

---

# 🗂️ Project Structure

```plaintext
AI-generated-animal-image-detector/
│
├── main.py                     # ⭐ 실제 메인 실행 스크립트 (train.py 아님)
├── train.py                    # (deprecated) 이전 학습 스크립트
│
├── data_loader.py              # Dataset & DataLoader 생성
├── model_builder.py            # Model / Optimizer builder
├── trainer.py                  # Training + Validation + Model saving
├── evaluate.py                 # Test evaluation + CSV 기록
│
├── visualize_features.py       # ⭐ Grad-CAM/activation 기반 feature 시각화
├── gan.py                      # ⭐ GAN 기반 이미지 생성 실험 모듈
│
├── generator/
│     └── diffusion_generator.py    # Stable Diffusion Turbo 이미지 생성
│
├── preprocessing/
│     ├── animalize.py             # WordNet 기반 species filtering
│     └── show_data.py             # CSV 정보 시각화/분석
│
├── utils/
│     ├── metrics.py                 # Accuracy/Precision/Recall/F1/ROC-AUC
│     └── dataset_reorganize_script.py
│
├── class.json                       # ImageNet ID → label 매핑
├── requirements.txt                 # ⚠️ 일부 깨진 문자 포함 — 재생성 권장
└── results/                         # 모델/로그/CSV 저장
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
├── val/
└── test/
```

AI images may also follow structure:

```
ai/<generator_name>/<class_name>/...
```

---

# ⚙️ Environment

* Python 3.10+
* PyTorch 2.7.1 (cu118)
* Torchvision 0.22.1
* timm 1.0.21
* diffusers 0.35+
* transformers 4.57+
* scikit-learn, pandas, tqdm, Pillow

⚠️ *requirements.txt contains corrupted characters — consider regenerating it with:*

```bash
pip freeze > requirements.txt
```

---

# 🚀 How to Run Training

```bash
export DATA_PATH=/path/to/dataset
export SAVE_PATH=/path/to/save_dir

python main.py
```

### main.py 수행 과정 (핵심)

1. 모델 × 옵티마이저 × 하이퍼파라미터 조합 자동 생성
2. 이미 학습된 모델(pth)이 있으면 스킵
3. 데이터셋 로딩 → 모델 생성 → 학습 → 검증
4. Best model 저장 (.pth)

---

# 🧬 Dataset Details

### 🐾 Real Images

* Source: Kaggle
* 8 classes: *elephant, cow, sheep, dog, cat, chicken, horse, rabbit*
* 2,000 images per class → Total 16,000
* 70/15/15 split

### 🤖 AI-Generated Images

* Model: Stable Diffusion Turbo
* Same 8 classes, 2,000 images each
* Same split ratio

---

# 🔍 File-by-File Explanation

---

## 🎨 visualize_features.py

**“모델이 어디를 보고 판단하는가?”** 를 시각화하는 도구

### Features

* Grad-CAM heatmap 생성
* Feature map / activation 시각화
* 특정 layer 또는 class에 대해 attention 분석
* 결과를 PNG/JPEG로 저장

### Example

```bash
python visualize_features.py \
  --model_path saved_model.pth \
  --image example.jpg \
  --output_dir ./feature_vis
```

---

## 🧬 gan.py

GAN 기반 이미지 생성 실험 모듈

### Features

* Simple GAN architecture
* AI 이미지 데이터 보강을 위한 synthetic image generation
* Diffusion 모델 대비 GAN 비교 실험 가능

### Example

```bash
python gan.py --epochs 50 --save_dir ./gan_outputs
```

---

## 🏋️ trainer.py

* train → validate loop
* Early saving
* Epoch metric logging
* Uses evaluate_val_acc for best checkpoint selection

## 📦 data_loader.py

* ImageFolder 기반 로딩
* Resize(224×224) → ToTensor → Normalize
* Train/Val/Test DataLoader 생성

## 🧠 model_builder.py

* Builds models: ResNet50, EfficientNet-B0, ViT
* Loads pretrained weights
* Replaces classification head → 2 outputs (real vs ai)
* Creates optimizer according to settings

## 🧪 evaluate.py

* Loads saved .pth weights
* Computes metrics via utils.metrics
* Appends results to CSV
* Parses model naming structure (⚠️ strict format — keep consistent)

## 📊 utils/metrics.py

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC

## 🧹 utils/dataset_reorganize_script.py

* Reorganizes dataset folders
* Renames files
* Handles AI/real unclassification

## 🧨 diffusion_generator.py

* Generates images using Stable Diffusion pipelines
* Includes optional SD 1.5 + SDXL (commented)
* Active: SD-Turbo for fast generation
* Splits train/val/test automatically

---

# ⚠️ Important Notes / Gotchas

* main.py is the *actual* training entrypoint — train.py is legacy
* requirements.txt contains corrupted characters
* diffusion_generator.py requires:

  * GPU
  * valid HuggingFace tokens (if downloading SD models)

---

# 👨‍💻 Author

**Seung-hyeon Lee (이승현)**

---