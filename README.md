# AI-Generated Animal Image and Real Animal Image Classification

**AI 생성 이미지와 실제 이미지**를 분류하는 모델을 구축하고 최적화하는 연구.  
ResNet50, EfficientNet-B0, ViT 모델을 기반으로 다양한 **Optimizer / Learning Rate / Batch Size / Epoch 수** 조합을 통해 최적의 모델 연구를 진행

---

## 📁 Project Structure

```bash
AI-generated-animal-image-detector/
│
├── train.py # main 실행 script
├── data_loader.py # Dataset && DataLoader 정의
├── model_builder.py # Model && Optimizer builder function 정의
├── trainer.py # train, validate && model save logic
├── evaluate.py # test && saved model 기반 metrics 실행 && result save
├── generator/
│     └── diffusion_generator.py # AI 이미지 생성 script
├── utils/
│     └── metrics.py # 성능 평가 지표 계산 함수
└── results/ # 성능 평가 결과 및 model 저장 위치
```

```bash
dataset/
│
├── train/
│     ├── real # Real animal images
│     │    ├── real_image_0000.png
│     │    └── ...
│     └── ai # Ai-generated animal images
│          ├── ai_image_0000.png
│          └── ...
├── val/ # Same structure as 'train/'
│     └── ...
└── test/ # Same structure as 'train/'
      └── ...
```
---

## Environment

- Python 3.10+
- torch 2.7.1+cu118
- torchvision 0.22.1+cu118
- torchaudio 2.7.1+cu118
- timm 1.0.21
- diffusers 0.35.2
- transformers 4.57.1
- scikit-learn 1.7.2
- pandas 2.3.3
- tqdm 4.67.1
- python-dotenv 1.1.1
- Pillow 12.0.0
- 자세한 내용은 requirements.txt

---

## How to run?

```bash
# 환경 변수 설정
export DATA_PATH=/path/to/dataset
export SAVE_PATH=/path/to/save_dir

# 학습 실행
python train.py
```
- DATA_PATH
  - train/, val/, test/ directory를 포함한 dataset 경로
- SAVE_PATH
  - 학습된 model과 결과 log가 저장될 directory

---

## Dataset
- Data balanced

### Real images
- Kaggle에서 수집한 animal images
- Elephant, cow, sheep, dog, cat, chicken, horse, rabbit으로 구성
- 각 species 별로 2000장 (도합 16,000장)
- train : test : val = 7 : 1.5 : 1.5 비율

### AI images
- sd_turbo(diffusion model)로 생성한 animal images
- Elephant, cow, sheep, dog, cat, chicken, horse, rabbit으로 구성
- 각 species 별로 2000장 (도합 16,000장)
- train : test : val = 7 : 1.5 : 1.5 비율

---
## File explanation
>train.py
- 메인 실행 스크립트
- 전체 실험 조합(Model × Optimizer × Hyperparameters)을 자동으로 탐색
- train_and_evaluate() 호출

>data_loader.py
- Dataset loading 및 preprocessing 정의
- train, validate, test set load
- Common preprocessing(transform):
  - Resize(224, 224)
  - ToTensor()
  - ImageNet normalization
- 지정된 batch size로 DataLoader return

>model_builder.py
- Model & Optimizer 정의
  - ResNet50(Baseline)
  - EfficientNet-B0
  - ViT Base
- Pretrained model load 후 output layer -> binary classification 수정
- 이름에 따라 Optimizer 생성

>trainer.py
- train, validate
- Model save
- 한 조합에 대한 전체 train 수행
- 각 epoch에서
  - Model train
  - Validation set 성능 평가
  - Model save

>evaluate.py
- train 완료 model에 대한 metrics 계산 script
- saved model을 불러와 test
- 계산된 metrics를 results.csv로 저장

>utils/dataset_reorganize_script.py
- Dataset 정리 용도
- Class 별로 저장되어있던 AI images, Real images를 unclassify
- Image rename

>utils/metrics.py
- 평가 지표 계산 함수
- 예측값(y_pred)과 정답(y_true)을 입력받아 주요 분류 지표를 계산
- 지표 목록
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

>generator/diffusion_generator.py
- Stable diffusion 기반의 AI 생성 이미지 dataset 생성 script
- Model load
  - sd_turbo(사용됨)
  - stable diffusion v1.5
  - stable diffusion xl base 1.0
- 8개의 class
  - elephant, cat, chicken, cow, dog, horse, rabbit, sheep
- Image generate
  - Resolution : 224 x 224
  - Prompt : 'Photo of a {class_name}, high quality, natural lighting'
- Dataset division
  - train : val : test = 7 : 1.5 : 1.5

---

## Author
> Seung-hyeon Lee, 이승현