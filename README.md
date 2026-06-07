# Diabetic Retinopathy Severity Grading
### CNN + Transfer Learning | Machine Learning Course — Phase II

---

## Quick Links

| Resource | Link |
|----------|------|
| Kaggle Notebook | https://www.kaggle.com/code/funkysaint/diabetic-retinopathy-severity-grading-v01 |
| Live Demo (Gradio UI) | https://a936ba3c38e1effe82.gradio.live |
| Dataset | https://www.kaggle.com/competitions/aptos2019-blindness-detection |

---

## Project Overview

An automated deep learning system that classifies retinal fundus photographs into five diabetic retinopathy (DR) severity grades using CNN and Transfer Learning.

- **Dataset:** APTOS 2019 Blindness Detection — 3,662 labeled retinal images
- **Task:** 5-class severity grading (Grade 0–4)
- **Framework:** TensorFlow / Keras
- **Models:** Custom CNN, ResNet50, MobileNetV2

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

## Model Results (Test Set)

| Model | Accuracy | Quadratic WK | Macro F1 | Macro AUC |
|-------|----------|--------------|----------|-----------|
| Custom CNN | **67.85%** | 0.6404 | 0.4711 | 0.8607 |
| ResNet50 | 62.40% | **0.6709** | 0.4305 | 0.8711 |
| MobileNetV2 | 62.67% | 0.6647 | 0.4589 | **0.8935** |

> **Best model by QWK:** ResNet50 (QWK = 0.6709)
> **Best model by Accuracy:** Custom CNN (67.85%)
> **Best model by AUC:** MobileNetV2 (0.8935)

### Per-Class Results — Custom CNN

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No DR | 0.87 | 0.97 | 0.91 | 181 |
| Mild | 0.45 | 0.35 | 0.39 | 37 |
| Moderate | 0.76 | 0.47 | 0.58 | 100 |
| Severe | 0.22 | 0.42 | 0.29 | 19 |
| Proliferative DR | 0.16 | 0.20 | 0.18 | 30 |

### Per-Class Results — ResNet50

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No DR | 0.90 | 0.95 | 0.92 | 181 |
| Mild | 0.46 | 0.49 | 0.47 | 37 |
| Moderate | 0.69 | 0.24 | 0.36 | 100 |
| Severe | 0.16 | 0.47 | 0.24 | 19 |
| Proliferative DR | 0.13 | 0.20 | 0.16 | 30 |

### Per-Class Results — MobileNetV2

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No DR | 0.94 | 0.87 | 0.91 | 181 |
| Mild | 0.25 | 0.84 | 0.38 | 37 |
| Moderate | 0.72 | 0.29 | 0.41 | 100 |
| Severe | 0.35 | 0.32 | 0.33 | 19 |
| Proliferative DR | 0.38 | 0.20 | 0.26 | 30 |

---

## Models

| Model | Type | Parameters | Strategy |
|-------|------|------------|----------|
| Custom CNN | Trained from scratch | ~1.83M | 5 conv blocks + GlobalAvgPool |
| ResNet50 | Transfer Learning (ImageNet) | ~23M | Freeze → fine-tune conv4 & conv5 |
| MobileNetV2 | Transfer Learning (ImageNet) | ~3.4M | Freeze → fine-tune top 30% layers |

---

## Output Figures (PDF)

All figures saved in the `figures/` folder:

| Figure | Description |
|--------|-------------|
| fig1_sample_images.pdf | Sample retinal images for all 5 DR grades |
| fig2_class_distribution.pdf | Class imbalance bar chart |
| fig3_preprocessing.pdf | Raw vs Ben Graham enhanced images |
| fig4a_cnn_curves.pdf | Custom CNN accuracy & loss curves |
| fig4b_resnet_curves.pdf | ResNet50 accuracy & loss curves |
| fig4c_mobilenet_curves.pdf | MobileNetV2 accuracy & loss curves |
| fig6a_cm_cnn.pdf | Custom CNN confusion matrix |
| fig6b_cm_resnet.pdf | ResNet50 confusion matrix |
| fig6c_cm_mobilenet.pdf | MobileNetV2 confusion matrix |
| fig7a_roc_cnn.pdf | Custom CNN ROC curves (one-vs-rest) |
| fig7b_roc_resnet.pdf | ResNet50 ROC curves |
| fig7c_roc_mobilenet.pdf | MobileNetV2 ROC curves |
| fig8_gradcam.pdf | Grad-CAM saliency visualizations |
| table4_model_comparison.pdf | Model comparison table (visual) |

---

## Output Tables (CSV)

All tables saved in the `tables/` folder:

| Table | Description |
|-------|-------------|
| table1_class_distribution.csv | Class distribution across all 5 grades |
| table2_dataset_split.csv | Train / Validation / Test split per grade |
| table3_report_Custom_CNN.csv | Classification report — Custom CNN |
| table3_report_ResNet50.csv | Classification report — ResNet50 |
| table3_report_MobileNetV2.csv | Classification report — MobileNetV2 |
| table4_model_comparison.csv | Final model comparison (Accuracy, QWK, F1, AUC) |

---

## Dataset Information

- **Name:** APTOS 2019 Blindness Detection
- **Source:** Kaggle — Aravind Eye Hospital, India
- **Link:** https://www.kaggle.com/competitions/aptos2019-blindness-detection
- **Total Images:** 3,662 labeled training images

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
- Ensemble prediction combining all 3 models (Gradio UI)
- Quadratic Weighted Kappa (QWK) clinical evaluation metric

---

## How to Run

### Kaggle (Recommended)
1. Open the [Kaggle Notebook](https://www.kaggle.com/code/funkysaint/diabetic-retinopathy-severity-grading-v01)
2. Click **Copy & Edit**
3. Add APTOS 2019 dataset via **+ Add Input**
4. Enable **GPU T4 x2** in Settings
5. Click **Run All**

### Live Demo
Try the model instantly — no setup needed:
👉 https://a936ba3c38e1effe82.gradio.live

### Local
```bash
git clone https://github.com/YOUR_USERNAME/diabetic-retinopathy-severity-grading
cd diabetic-retinopathy-severity-grading
pip install -r requirements.txt
jupyter notebook diabetic-retinopathy-severity-grading-v01.ipynb
```

---

## Repository Structure

```
diabetic-retinopathy-severity-grading/
├── diabetic-retinopathy-severity-grading-v01.ipynb
├── README.md
├── requirements.txt
├── figures/
│   ├── fig1_sample_images.pdf
│   ├── fig2_class_distribution.pdf
│   ├── fig3_preprocessing.pdf
│   ├── fig4a_cnn_curves.pdf
│   ├── fig4b_resnet_curves.pdf
│   ├── fig4c_mobilenet_curves.pdf
│   ├── fig6a_cm_cnn.pdf
│   ├── fig6b_cm_resnet.pdf
│   ├── fig6c_cm_mobilenet.pdf
│   ├── fig7a_roc_cnn.pdf
│   ├── fig7b_roc_resnet.pdf
│   ├── fig7c_roc_mobilenet.pdf
│   ├── fig8_gradcam.pdf
│   └── table4_model_comparison.pdf
└── tables/
    ├── table1_class_distribution.csv
    ├── table2_dataset_split.csv
    ├── table3_report_Custom_CNN.csv
    ├── table3_report_ResNet50.csv
    ├── table3_report_MobileNetV2.csv
    └── table4_model_comparison.csv
```
