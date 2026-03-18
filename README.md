# OzoneWatch — Air Quality Risk Prediction System

## Problem
Predicting dangerous ozone level days using atmospheric weather data from Houston, TX.

## Dataset
2534 days of weather readings | 72 features | Binary classification

## What I Did
- EDA — found 94/6 class imbalance, multicollinearity in temperature features
- SMOTE — balanced training data from 127 to 1900 ozone samples
- Trained 3 models — Logistic Regression, Random Forest, XGBoost
- Selected Logistic Regression — highest recall (0.91) for ozone days

## Tech Stack
Python | Pandas | Scikit-learn | XGBoost | SMOTE | Matplotlib | Seaborn

## Results
| Model | Recall (Ozone) | F1 (Ozone) |
|---|---|---|
| Logistic Regression | 0.91 | 0.42 |
| Random Forest | 0.36 | 0.39 |
| XGBoost | 0.42 | 0.39 |
