# Diabetic Retinopathy Severity Grading
### CNN + Transfer Learning | Machine Learning Course — Phase II

---

## Project Overview

An automated deep learning system that classifies retinal fundus photographs into five diabetic retinopathy (DR) severity grades using Convolutional Neural Networks and Transfer Learning.

**Dataset:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) — 3,662 labeled retinal images from Aravind Eye Hospital, India.

**Kaggle Notebook:** https://www.kaggle.com/code/funkysaint/diabetic-retinopathy-severity-grading-v01

---

## DR Severity Grades

| Grade | Severity | Description |
|-------|----------|-------------|
| 0 | No DR | Healthy retina, no lesions |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than mild, less than severe |
| 3 | Severe | Extensive hemorrhages (4-2-1 rule) |
| 4 | Proliferative DR | Neovascularization, risk of blindness |

---

## Models

| Model | Type | Parameters |
|-------|------|------------|
| Custom CNN | Trained from scratch | ~1.83M |
| ResNet50 | Transfer Learning (ImageNet) | ~23M |
| MobileNetV2 | Transfer Learning (ImageNet) | ~3.4M |

---

## Repository Structure

```
diabetic-retinopathy-severity-grading/
├── diabetic-retinopathy-severity-grading-v01.ipynb
├── README.md
├── requirements.txt
└── figures/
    ├── fig1_sample_images.pdf
    ├── fig2_class_distribution.pdf
    ├── fig3_preprocessing.pdf
    ├── fig4a_cnn_curves.pdf
    ├── fig4b_resnet_curves.pdf
    ├── fig4c_mobilenet_curves.pdf
    ├── fig6a_cm_cnn.pdf
    ├── fig6b_cm_resnet.pdf
    ├── fig6c_cm_mobilenet.pdf
    ├── fig7a_roc_cnn.pdf
    ├── fig7b_roc_resnet.pdf
    ├── fig7c_roc_mobilenet.pdf
    ├── fig8_gradcam.pdf
    └── table4_model_comparison.pdf
```

---

## Dataset Information

- **Name:** APTOS 2019 Blindness Detection
- **Source:** Kaggle — Aravind Eye Hospital, India
- **Link:** https://www.kaggle.com/competitions/aptos2019-blindness-detection
- **Total Images:** 3,662 labeled training images
- **Format:** PNG retinal fundus photographs
- **Classes:** 5 (Grade 0–4)

| Grade | Label | Images | Percentage |
|-------|-------|--------|------------|
| 0 | No DR | 1,805 | 49.3% |
| 1 | Mild | 370 | 10.1% |
| 2 | Moderate | 999 | 27.3% |
| 3 | Severe | 193 | 5.3% |
| 4 | Proliferative DR | 295 | 8.1% |

---

## Key Features

- Ben Graham retinal contrast enhancement preprocessing
- Balanced class weights to handle dataset imbalance
- Two-phase transfer learning (freeze → fine-tune)
- Early stopping to prevent overfitting
- Grad-CAM visualizations for model interpretability
- Ensemble prediction combining all 3 models
- Quadratic Weighted Kappa (QWK) clinical evaluation metric

---

## How to Run

### Kaggle (Recommended)
1. Open the [Kaggle Notebook](https://www.kaggle.com/code/funkysaint/diabetic-retinopathy-severity-grading-v01)
2. Click **Copy & Edit**
3. Add APTOS 2019 dataset via **+ Add Input**
4. Enable **GPU T4 x2** in Settings
5. Click **Run All**

### Local
```bash
git clone https://github.com/YOUR_USERNAME/diabetic-retinopathy-severity-grading
cd diabetic-retinopathy-severity-grading
pip install -r requirements.txt
jupyter notebook diabetic-retinopathy-severity-grading-v01.ipynb
```

---

## Course Information

- **Course:** Machine Learning (ML-90)
- **Phase:** II — Proposal & Code Implementation
- **University:** University of Europe for Applied Sciences, Potsdam
- **Due:** June 7, 2026
