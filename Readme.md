# 🩺 Leukemia Cancer Prediction — ML Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-green" />
  <img src="https://img.shields.io/badge/Records-143%2C194-lightgrey" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

> **Author:** Vinayak Vivek Joshi
> **Dataset:** `leukemia.csv` — 143,194 patient records · 21 features · 1 binary target
> **Objective:** Predict whether a patient is **Leukemia Positive or Negative** using an end-to-end machine learning pipeline.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Models & Results](#-models--results)
- [Feature Engineering Summary](#-feature-engineering-summary)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)
- [Key Insights](#-key-insights)
- [Future Work](#-future-work)

---

## 🔬 Project Overview

This project builds a complete **supervised binary classification** pipeline to predict leukemia status from clinical, demographic, and lifestyle patient data. The pipeline covers every stage from raw data ingestion through EDA, cleaning, feature engineering, model training, evaluation, and best-model auto-selection.

**Target Variable:** `Leukemia_Status`
| Class | Label |
|-------|-------|
| Positive | 1 — Patient has leukemia |
| Negative | 0 — Patient does not have leukemia |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `leukemia.csv` |
| Records | 143,194 patients |
| Raw Features | 21 |
| Target | `Leukemia_Status` (Positive / Negative) |
| Missing Values | None |
| Duplicate Rows | Present (removed during cleaning) |

### Feature Categories

**Numerical (7)**
`Age`, `WBC_Count`, `RBC_Count`, `Platelet_Count`, `Hemoglobin_Level`, `Bone_Marrow_Blasts`, `BMI`

**Categorical — Binary Yes/No (8)**
`Genetic_Mutation`, `Family_History`, `Smoking_Status`, `Alcohol_Consumption`, `Radiation_Exposure`, `Infection_History`, `Chronic_Illness`, `Immune_Disorders`

**Categorical — Other (5)**
`Gender`, `Ethnicity`, `Socioeconomic_Status`, `Urban_Rural`, `Country` *(dropped — not predictive)*

**Administrative (dropped)**
`Patient_ID`, `Country`

---

## 📁 Project Structure

```
Leukemia-Cancer-Prediction/
│
├── leukemia_analysis.ipynb     # Main notebook — complete ML pipeline (71 cells)
├── leukemia.csv                # Raw dataset (not included; add locally)
└── README.md                   # This file
```

---

## 🔧 Pipeline Walkthrough

### 1. Setup & Imports
Standard data science stack: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

### 2. Load Data
Dataset loaded from `leukemia.csv`; shape confirmed as `(143194, 22)`.

### 3. Dataset Overview & Quality Check
- `df.info()` — dtype inspection
- Missing values confirmed: **zero**
- Duplicate rows identified and flagged
- Class imbalance ratio (Negative : Positive) checked

### 4. Exploratory Data Analysis (EDA)

| Analysis | Description |
|----------|-------------|
| **Univariate — Numerical** | Histograms with skewness annotations for all 7 numerical features |
| **Univariate — Categorical** | Bar charts for all 12 categorical features |
| **Bivariate — Numerical vs Target** | Side-by-side box plots per feature, outliers hidden |
| **Bivariate — Categorical vs Target** | Row-normalised stacked bar charts for 6 key risk factors |
| **Correlation Heatmap** | Lower-triangle Pearson correlation matrix |
| **Pairplot** | 3,000-row stratified sample — KDE diagonals, scatter by class |

### 5. Data Cleaning

| Step | Action |
|------|--------|
| Missing values | None — no imputation required |
| Duplicate removal | `drop_duplicates()` applied |
| Outlier treatment | **Winsorisation** at 1st / 99th percentile (preserves dataset size) |
| String normalisation | `strip()` + `title()` on all object columns |
| Column removal | `Patient_ID`, `Country` dropped |

### 6. Feature Engineering & Encoding

| Transformation | Columns / Details |
|----------------|-------------------|
| **Binary encoding (Yes→1 / No→0)** | `Genetic_Mutation`, `Family_History`, `Smoking_Status`, `Alcohol_Consumption`, `Radiation_Exposure`, `Infection_History`, `Chronic_Illness`, `Immune_Disorders` |
| **Binary encoding (label-based)** | `Gender` (Male=1), `Urban_Rural` (Urban=1) |
| **Ordinal encoding** | `Socioeconomic_Status` → Low=0, Medium=1, High=2 |
| **One-Hot encoding** | `Ethnicity` (drop_first=True) |
| **BMI_Category** (new feature) | 0=Underweight, 1=Normal, 2=Overweight, 3=Obese |
| **Age_Group** (new feature) | 0=Child (<18), 1=Young Adult (18–35), 2=Middle-Aged (35–60), 3=Senior (60+) |
| **WBC_High** (new binary feature) | 1 if WBC_Count > median, else 0 (clinically relevant threshold) |
| **Yeo-Johnson Power Transform** | `WBC_Count`, `Platelet_Count` (skewed distributions normalised) |
| **Target encoding** | Positive=1, Negative=0 |

### 7. Feature Scaling
`StandardScaler` applied **only** to continuous numerical columns inside a `ColumnTransformer`. Binary and ordinal features are passed through unchanged.

### 8. Train / Test Split
- 80% train / 20% test — `stratify=y`, `random_state=42`

### 9. Smart Pipeline — Auto-Selects Best Model
Three `sklearn.Pipeline` objects are built (preprocessor → classifier). All are trained, evaluated, and the best is auto-selected by **ROC-AUC**.

---

## 📈 Models & Results

### Models Trained

| Model | Key Hyperparameters |
|-------|---------------------|
| **Logistic Regression** | `max_iter=1000`, `class_weight='balanced'` |
| **Decision Tree** | `max_depth=10`, `min_samples_leaf=20`, `class_weight='balanced'` |
| **Random Forest** | `n_estimators=100`, `max_depth=15`, `min_samples_leaf=10`, `class_weight='balanced'` |

> `class_weight='balanced'` is used across all models to handle the inherent class imbalance in medical datasets.

### Evaluation Metrics
- **Accuracy** — overall correctness
- **F1-Score** — harmonic mean of precision & recall (important for imbalanced data)
- **ROC-AUC** — area under the Receiver Operating Characteristic curve (primary selection metric)
- **Confusion Matrix** — per-class breakdown
- **5-Fold Stratified Cross-Validation ROC-AUC** — generalisation stability

### Best Model
The best-performing model is **auto-selected at runtime** based on highest test ROC-AUC and stored in `best_pipeline` — ready for inference without any manual intervention.

---

## 🧬 Feature Engineering Summary

| Step | Detail |
|------|--------|
| Missing Values | None found |
| Duplicates | Removed |
| Outliers | Winsorised at 1st / 99th percentile |
| String Cleaning | Strip + Title-case |
| Dropped Columns | `Patient_ID`, `Country` |
| Binary Encoding | Yes/No columns, Gender, Urban_Rural |
| Ordinal Encoding | Socioeconomic_Status |
| One-Hot Encoding | Ethnicity |
| Feature Construction | `BMI_Category`, `Age_Group`, `WBC_High` |
| Power Transform | `WBC_Count`, `Platelet_Count` (Yeo-Johnson) |
| Scaling | StandardScaler inside ColumnTransformer |
| Models Tried | Logistic Regression, Decision Tree, Random Forest |
| Best Model | Auto-selected by highest ROC-AUC |

---

## ⚙️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

| Library | Version |
|---------|---------|
| Python | ≥ 3.8 |
| pandas | ≥ 1.3 |
| numpy | ≥ 1.21 |
| scikit-learn | ≥ 1.0 |
| matplotlib | ≥ 3.4 |
| seaborn | ≥ 0.11 |

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/vinayakjoshi04/Leukemia-Cancer-Prediction-Predictive-Project-.git
cd Leukemia-Cancer-Prediction-Predictive-Project-
```

2. **Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. **Add the dataset**
Place `leukemia.csv` in the project root directory.

4. **Launch the notebook**
```bash
jupyter notebook leukemia_analysis.ipynb
```

5. **Run all cells**
Use `Kernel → Restart & Run All`. The best model pipeline will be auto-selected and stored in `best_pipeline`.

6. **Inference (after running all cells)**
```python
# Predict on new data
import pandas as pd
new_patient = pd.DataFrame([{ ... }])  # fill in feature values
prediction = best_pipeline.predict(new_patient)
probability = best_pipeline.predict_proba(new_patient)[:, 1]
print(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
print(f"Probability of Leukemia: {probability[0]:.2%}")
```

---

## 💡 Key Insights

- **No missing data** — the dataset is clean; preprocessing focuses on outliers and encoding.
- **Class imbalance** is addressed through `class_weight='balanced'` in all classifiers, ensuring minority class (Positive) is not underweighted.
- **Winsorisation** (rather than removal) is chosen for outlier treatment to preserve the full 143K-record dataset.
- **Yeo-Johnson transform** is applied to `WBC_Count` and `Platelet_Count` which exhibit high skewness — a clinically expected distribution for hematological markers.
- **Three engineered features** (`BMI_Category`, `Age_Group`, `WBC_High`) capture non-linear clinical thresholds that raw continuous values cannot directly express.
- **Feature importance** plots from Decision Tree and Random Forest reveal the top 15 most predictive variables, with bone marrow blast percentage and WBC count typically dominating.
- **5-fold stratified cross-validation** confirms model stability and guards against overfitting on the large dataset.

---

## 🙋 Author

**Vinayak Vivek Joshi**
[GitHub](https://github.com/vinayakjoshi04) · [LinkedIn](https://linkedin.com/in/vinayakjoshi04)

> *This project is intended for educational and research purposes. It is not a substitute for professional medical diagnosis.*