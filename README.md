# Eggplant Retail Price Volatility in the Philippines

## Introduction

This repository contains a Python script for analyzing and forecasting eggplant retail prices using ARIMA models. The analysis includes data loading, trend analysis, Augmented Dickey-Fuller (ADF) test, ARIMA model estimation, autocorrelation function (ACF) and partial autocorrelation function (PACF) plots, and forecasting.

## Requirements

- Python 3.6+
- Jupyter Notebook (optional, for interactive execution)
- pandas
- matplotlib
- seaborn
- statsmodels
- numpy

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository_https://github.com/john19645/EGGPLANTREADME/edit/main/README.md>
    cd <repository_folder>
    ```

2. **Set up a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: `env\\Scripts\\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install pandas matplotlib seaborn statsmodels numpy
    ```

## Running the Program

1. **Ensure the dataset file (`eggplant python.csv`) is in the same directory as the script.**

2. **Run the Python script:**
    ```bash
    python python_data_visualization_training.py
    ```

3. **Optional: Run in Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

## Code Explanation


### Data Loading and Exploration

The script starts by loading the dataset and exploring its contents.

```python
import pandas as pd

file_path = 'eggplant python.csv'
eggplant_data = pd.read_csv(file_path)
print(eggplant_data.head())
print(eggplant_data.info())
eggplant_data['Date'] = pd.to_datetime(eggplant_data['Date'])
eggplant_data.set_index('Date', inplace=True)
print(eggplant_data.describe())


### Trend Analysis

Plotting the time series data using Matplotlib.

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(eggplant_data.index, eggplant_data['Price of Eggplant'], label='Eggplant Price')
plt.title('Eggplant Retail Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price of Eggplant')
plt.legend()
plt.show()

3. Augmented Dickey-Fuller (ADF) Test
Performing the ADF test to check for stationarity of the data.

from statsmodels.tsa.stattools import adfuller

adf_result_original = adfuller(eggplant_data['Price of Eggplant'])
adf_result_diff = adfuller(eggplant_data['Price of Eggplant'].diff().dropna())

print("Original Data ADF Test Results:")
print(adf_result_original_summary)
print("\nDifferenced Data ADF Test Results:")
print(adf_result_diff_summary)

4. ARIMA Model Estimation and Diagnostics
Evaluating different ARIMA models and calculating accuracy metrics including MAE, MAPE, AIC, BIC, and log likelihood.

from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def evaluate_arima_model(order):
    model = ARIMA(eggplant_data['Price of Eggplant'], order=order)
    model_fit = model.fit()
    mae = np.mean(np.abs(model_fit.resid))
    mape = np.mean(np.abs(model_fit.resid / eggplant_data['Price of Eggplant'])) * 100
    aic = model_fit.aic
    bic = model_fit.bic
    log_likelihood = model_fit.llf
    return mae, mape, aic, bic, log_likelihood, model_fit

mae_112, mape_112, aic_112, bic_112, llf_112, model_fit_112 = evaluate_arima_model((1, 1, 2))
mae_311, mape_311, aic_311, bic_311, llf_311, model_fit_311 = evaluate_arima_model((3, 1, 1))
mae_113, mape_113, aic_113, bic_113, llf_113, model_fit_113 = evaluate_arima_model((1, 1, 3))

print('ARIMA(1,1,2) - MAE:', mae_112, 'MAPE:', mape_112, 'AIC:', aic_112, 'BIC:', bic_112, 'Log Likelihood:', llf_112)
print('ARIMA(3,1,1) - MAE:', mae_311, 'MAPE:', mape_311, 'AIC:', aic_311, 'BIC:', bic_311, 'Log Likelihood:', llf_311)
print('ARIMA(1,1,3) - MAE:', mae_113, 'MAPE:', mape_113, 'AIC:', aic_113, 'BIC:', bic_113, 'Log Likelihood:', llf_113)

5. ACF and PACF Plots
Creating ACF and PACF plots for both original and differenced data.

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

plot_acf(eggplant_data['Price of Eggplant'].dropna(), lags=40, ax=axes[0], color='blue', alpha=0.8)
axes[0].set_title('ACF - Original Data', fontsize=16, color='blue')
axes[0].set_xlabel('Lags', fontsize=14)
axes[0].set_ylabel('ACF', fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.5)

plot_pacf(eggplant_data['Price of Eggplant'].dropna(), lags=40, ax=axes[1], color='green', alpha=0.8)
axes[1].set_title('PACF - Original Data', fontsize=16, color='green')
axes[1].set_xlabel('Lags', fontsize=14)
axes[1].set_ylabel('PACF', fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

6. Forecasting
Forecasting future eggplant prices using the ARIMA models.

def forecast_arima_model(model_fit, steps=48):
    forecast = model_fit.get_forecast(steps=steps)
    forecast_df = forecast.summary_frame()
    return forecast_df

forecast_311 = model_fit_311.forecast(steps=48)
forecast_311.index = pd.date_range(start=eggplant_data.index[-1] + pd.DateOffset(1), periods=48, freq='M')

plt.figure(figsize=(14, 8))
plt.plot(eggplant_data['Price of Eggplant'], label='Original Data', color='blue', linewidth=2)
plt.plot(forecast_311, label='Forecasted Values', color='orange', linestyle='--', marker='o', markersize=8)
plt.fill_between(forecast_311.index, forecast_311['mean_ci_lower'], forecast_311['mean_ci_upper'], color='orange', alpha=0.2)
plt.title('Eggplant Price Forecast (ARIMA 3,1,1)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price of Eggplant', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate('Forecast Start', xy=(forecast_311.index[0], forecast_311[0]),
             xytext=(forecast_311.index[5], forecast_311[0] + 20),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

7. Displaying Results
Displaying the ADF test results and accuracy metrics in a table format.

adf_results = {
    'ADF Statistic': [adf_result_original[0], adf_result_diff[0]],
    'p-value': [adf_result_original[1], adf_result_diff[1]],
    'Critical Values': [adf_result_original[4], adf_result_diff[4]]
}

adf_results_df = pd.DataFrame(adf_results, index=['Original Data', 'Differenced Data'])
print(adf_results_df)

accuracy_metrics = {
    'Model': ['ARIMA(1,1,2)', 'ARIMA(3,1,1)', 'ARIMA(1,1,3)'],
    'MAE': [mae_112, mae_311, mae_113],
    'MAPE': [mape_112, mape_311, mape_113],
    'AIC': [aic_112, aic_311, aic_113],
    'BIC': [bic_112, bic_311, bic_113],
    'Log Likelihood': [llf_112, llf_311, llf_113]
}

accuracy_df = pd.DataFrame(accuracy_metrics)
print(accuracy_df)

print("ADF Test Results:")
print(adf_results_df)
print("\nAccuracy Metrics:")
 &#8203;:citation[oaicite:0]{index=0}&#8203;

