# Credit Fraud Detection — Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end machine learning system for detecting potential fraud in credit loan contracts. Built on a real-world dataset of **80,000+ records** with full pipeline coverage — from raw data exploration to tuned model comparison.

---

## System Overview

This project implements a supervised binary classification pipeline targeting `Possivel_Fraude` (Possible Fraud). The dataset contains loan contract records from Brazilian financial institutions, covering 24 features across borrower demographics, loan terms, and payment history.

The pipeline addresses a **heavily imbalanced** fraud label distribution using SMOTE oversampling, followed by rigorous GridSearchCV hyperparameter tuning across three distinct model families.

### ML Pipeline

| Stage | Description |
|---|---|
| Exploratory Data Analysis | Distributions, missing values, statistical summaries, outlier analysis |
| Feature Engineering | Age, income, loan term, and overdue days binned into meaningful categories |
| Preprocessing | Label Encoding, MinMax Normalization, Z-score Standardization |
| Class Balancing | SMOTE applied — training set expanded to 10,070 balanced samples |
| Model Training | Random Forest, SVM, KNN — each tuned with GridSearchCV |
| Model Comparison | All three models evaluated and ranked by accuracy |

---

## Model Performance

Results from GridSearchCV over the SMOTE-balanced training set (10,070 samples):

| Model | Training Accuracy | Models Evaluated | Best Hyperparameters |
|---|---|---|---|
| **Random Forest** | **99.26%** | 324 | `criterion=entropy, max_depth=20, max_features=log2, n_estimators=100` |
| SVM | 98.92% | 192 | `C=100, coef0=0.5, degree=2, gamma=0.01` |
| KNN | 97.04% | 120 | `algorithm=auto, leaf_size=30, metric=minkowski, n_neighbors=5` |

**Random Forest test accuracy: 98.97%**

### Top Feature Importances (Random Forest)

| Feature | Importance |
|---|---|
| `QT_Parcelas_Atraso` (overdue installments) | 0.53 |
| `QT_Total_Parcelas_Pagas` (total paid installments) | 0.21 |
| `Total_Pago` (total amount paid) | 0.10 |
| `QT_Total_Parcelas_Pagas_EmDia` (on-time installments) | 0.04 |
| `Qt_Renegociacao` (renegotiation count) | 0.03 |

---

## Dataset

- **80,000+ loan contract records** (samples: 10k, 20k, 80k versions)
- **24 features**: income, loan amount, interest rate, payment history, days overdue, marital status, state, age, renegotiation count
- **Target**: `Possivel_Fraude` — Yes / No
- **Period**: July–December 2022

---

## Tech Stack

```
Python 3.11 · pandas · scikit-learn · imbalanced-learn · matplotlib · seaborn · numpy · Jupyter
```

---

## Quick Start (Docker)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed

### 1. Clone the repository

```bash
git clone https://github.com/Wagnerdata/credit-fraud-detection-ml.git
cd credit-fraud-detection-ml
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set a secure JUPYTER_TOKEN
```

### 3. Build and start

```bash
docker compose up --build
```

### 4. Open Jupyter

Navigate to [http://localhost:8888](http://localhost:8888) and enter your token.

Open `CreditFraudDetection_ML.ipynb` to run the full pipeline.

### Stop the container

```bash
docker compose down
```

---

## Project Structure

```
├── CreditFraudDetection_ML.ipynb   # Main notebook (English)
├── ModeloPrevisaoFraude.ipynb      # Original notebook (Portuguese)
├── Dockerfile                      # Python 3.11 + Jupyter image
├── docker-compose.yml              # Service definition (port 8888)
├── requirements.txt                # Pinned Python dependencies
├── .env.example                    # Environment variable template
├── .dockerignore                   # Docker build exclusions
├── dados_coletados10k.csv          # 10,000-record sample
├── dados_coletados20k.csv          # 20,000-record sample
└── dados_coletados80k.csv          # Full 80,000-record dataset
```

> **Note:** The notebook defaults to the 10k sample. To use the full 80k dataset, uncomment `dados_coletados80k.csv` in cell 3.

---

## License

MIT
