# ❤️ Heart Disease Prediction with Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Accuracy](https://img.shields.io/badge/best%20accuracy-93.2%25-22c55e?style=flat-square)

> CS280 / CS485 — Introduction to Artificial Intelligence · SMU MedTech · Spring 2026

A complete machine-learning pipeline that predicts heart disease from 13 clinical measurements,
and gives a **prediction with a confidence score** for new patients.

## 📊 Data

[Kaggle Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) —
**1,025 patients**, 13 features (age, chest pain type, blood pressure, cholesterol, max heart rate, …),
balanced classes (51 % / 49 %), no missing values.

## ⚙️ Pipeline

1. **Exploration** — distributions, class balance, correlation heatmap
2. **Preprocessing** — stratified train/test split, feature scaling
3. **Models** — Logistic Regression · Decision Tree · Random Forest
4. **Tuning** — `GridSearchCV` with 5-fold cross-validation
5. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, ROC/AUC
6. **Inference** — predict for a new patient with a probability

## 🏆 Result

**Random Forest** was selected (best F1): **93.17 % accuracy** and **97.14 % recall** on the
disease class — the metric that matters most clinically, since missing a sick patient is the worst error.

## ▶️ Run it

```bash
pip install -r requirements.txt
python main.py
```

## 📄 Report

[Full lab report (Word)](docs/Heart_Disease_Report.docx)
