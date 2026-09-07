# Bank Customer Churn Prediction using Artificial Neural Networks (ANN)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning classification system developed to predict customer churn in retail banking. The pipeline processes demographic, transaction, and behavioral features through an Artificial Neural Network (Multi-Layer Perceptron) built using TensorFlow/Keras.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset Description](#dataset-description)
* [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
* [Model Architecture](#model-architecture)
* [Training & Callbacks](#training--callbacks)
* [Performance & Evaluation](#performance--evaluation)
* [Installation & Usage](#installation--usage)
* [Repository Structure](#repository-structure)
* [License](#license)

---

## Project Overview

Customer attrition directly degrades recurring revenue and customer lifetime value (LTV). This project develops a predictive classifier to detect whether an account holder is likely to leave the bank (`Exited = 1`) or stay (`Exited = 0`).

**Key Implementations:**
* Data cleansing, one-hot categorical encoding, and standard feature scaling.
* Fully connected Deep Neural Network with intermediate Dropout regularization.
* Validation loss tracking paired with early stopping to eliminate overfitting.
* Generalization validation on unseen holdout test data.

---

## Dataset Description

The project utilizes the [Bank Customer Churn Modelling Dataset](https://www.kaggle.com/datasets/aakash50897/churn-modellingcsv) (10,000 records, 14 attributes) downloaded via `kagglehub`:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `CustomerId` / `Surname` / `RowNumber` | Metadata | Customer identifiers (dropped prior to training) |
| `CreditScore` | Numeric | Numerical customer credit rating |
| `Geography` | Categorical | Country of account registration (`France`, `Spain`, `Germany`) |
| `Gender` | Categorical | Biological sex (`Male`, `Female`) |
| `Age` | Numeric | Customer age in years |
| `Tenure` | Numeric | Number of years active with the bank |
| `Balance` | Numeric | Account balance amount |
| `NumOfProducts` | Numeric | Number of bank services utilized |
| `HasCrCard` | Binary | Credit card ownership flag (0/1) |
| `IsActiveMember` | Binary | Bank activity indicator flag (0/1) |
| `EstimatedSalary` | Numeric | Estimated gross annual compensation |
| **`Exited` (Target)** | **Binary** | **Churn status (`1` = Left, `0` = Retained)** |

---

## Data Preprocessing Pipeline

1. **Feature Pruning:** Stripped non-predictive metadata fields (`RowNumber`, `CustomerId`, `Surname`) from the input matrix[cite: 1].
2. **One-Hot Encoding:** Encoded categorical features `Geography` and `Gender` into binary dummy variables (`drop_first=True`) to avoid the dummy variable trap[cite: 1].
3. **Train/Test Partition:** Split dataset into an 80% training set (8,000 samples) and a 20% test set (2,000 samples) with `random_state=0`[cite: 1].
4. **Standard Scaling:** Scaled all continuous and encoded binary features using `StandardScaler` fitted strictly on `X_train` to prevent data leakage[cite: 1]:

$$z = \frac{x - \mu}{\sigma}$$

---

## Model Architecture

The Multi-Layer Perceptron is implemented sequentially via TensorFlow/Keras[cite: 1]:

```text
Input Features (11)
       │
       ▼
Dense Layer (11 units, ReLU)
       │
       ▼
Dense Layer (7 units, ReLU)
       │
       ▼
Dropout Layer (Rate = 0.3)
       │
       ▼
Dense Layer (6 units, ReLU)
       │
       ▼
Output Layer (1 unit, Sigmoid)
