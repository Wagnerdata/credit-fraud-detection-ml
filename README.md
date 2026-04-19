# Credit Fraud Detection — Machine Learning

End-to-end machine learning project for detecting potential fraud in credit loan contracts. Built with Python using a real-world dataset of over 80,000 records.

## Overview

This project covers the full ML pipeline — from raw data exploration to model comparison — applied to a financial fraud detection problem, a high-value domain for banking and fintech companies.

## What's Inside

| Step | Description |
|------|-------------|
| Exploratory Data Analysis | Overview, distributions, missing values, statistical summaries |
| Feature Engineering | Age, income, loan term, and overdue days binned into meaningful categories |
| Data Preprocessing | Label Encoding, MinMax Normalization, Z-score Standardization |
| Class Balancing | SMOTE applied to handle heavily imbalanced fraud labels |
| Model Training | Random Forest, SVM, and KNN with GridSearchCV hyperparameter tuning |
| Model Comparison | All three models compared on accuracy and best parameters |

## Dataset

- **80,000+ loan contract records**
- **24 features** including: income, loan amount, interest rate, payment history, days overdue, marital status, state, age, renegotiation count
- **Target variable:** `Possivel_Fraude` (Possible Fraud — Yes/No)
- **Class imbalance:** corrected with SMOTE oversampling

## Models & Techniques

- **Random Forest** — GridSearchCV over 1,000+ combinations (n_estimators, max_depth, criterion, max_features)
- **Support Vector Machine (SVC)** — tuned on C, gamma, degree, coef0
- **K-Nearest Neighbors** — tuned on n_neighbors, algorithm, leaf_size, metric
- Feature importance ranking from the best Random Forest model

## Tech Stack

```
Python · pandas · scikit-learn · imbalanced-learn · matplotlib · seaborn · numpy
```

## How to Run

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn numpy
jupyter notebook CreditFraudDetection_ML.ipynb
```

> The notebook defaults to the 10k sample. To use the full 80k dataset, uncomment `dados_coletados80k.csv` in cell 3.

## Project Structure

```
├── CreditFraudDetection_ML.ipynb   # Main notebook (English)
├── dados_coletados10k.csv          # 10,000-record sample
├── dados_coletados20k.csv          # 20,000-record sample
└── dados_coletados80k.csv          # Full 80,000-record dataset
```
