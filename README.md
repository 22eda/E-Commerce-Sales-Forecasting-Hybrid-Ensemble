# E-Commerce Sales Forecasting: Hybrid Machine Learning Approach

A high-precision sales forecasting system designed for chaotic e-commerce environments, utilizing a hybrid **Stacking Ensemble** architecture to manage extreme volatility and campaign-driven demand spikes.

## 🚀 Overview
Predicting daily sales in e-commerce is challenging due to sudden promotional "spikes," high volatility, and non-linear demand patterns. This project implements a hybrid architecture that combines statistical rigor with deep learning flexibility to optimize stock management and reduce inventory costs.

## 🧠 Hybrid Stacking Ensemble Architecture
The core of this system is a two-layer **Stacking Ensemble** that blends the strengths of individual models:

* **Base Learners:**
    * **Log-XGBoost:** Excellent at capturing non-linear feature interactions and handling campaign-driven demand shocks.
    * **Multivariate Bi-LSTM:** Utilizes historical sequential dependencies to capture long-term trends and cyclical patterns.
* **Meta-Learner:**
    * **Ridge Regression:** Harmonizes predictions from base learners to produce a stable, final forecast.

## 📊 Performance Benchmark
The hybrid approach significantly outperforms baseline statistical models, achieving an **R² score of 0.7361**.

| Model Architecture | MAE (£) | RMSE (£) | R² Score | Improvement vs. SARIMAX |
| :--- | :---: | :---: | :---: | :---: |
| SARIMAX (Baseline) | 7.464 | 9.308 | 0.5724 | — |
| Log-XGBoost | 5.504 | 7.554 | 0.7184 | +25.50% |
| Bi-LSTM | 9.533 | 12.348 | 0.2475 | -56.7% |
| **Stacking Ensemble** | **5.743** | **7.313** | **0.7361** | **+28.60%** |

## 🛠️ Key Methodologies
* **ABC Analysis:** Applying Pareto principles to isolate high-value "A-Class" products for focused forecasting.
* **Volatility Management:** Addressing an extreme coefficient of variation (**CV: 86.2%**) in sales data to handle chaotic demand.
* **Feature Engineering:** Creation of 38 distinct features, including Fourier terms for 7-day seasonality.
* **Data Transformation:** `np.log1p` scaling for trees and MinMax scaling for neural networks to ensure stable convergence.

## 📂 Repository Structure
```text
├── data/                      # Cleaned daily aggregation (739 days)
├── main.ipynb                 # Full pipeline (SARIMAX, XGBoost, Bi-LSTM, Stacking)
└── README.md                  # Project documentation

🛠️ Requirements
Python 3.x

XGBoost

TensorFlow / Keras (Bi-LSTM implementation)

Statsmodels (SARIMAX)

Pandas, NumPy, Scikit-learn

Author: Edanur Demirel

Advisor: Prof. Serpil Türkyılmaz





