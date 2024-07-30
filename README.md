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
    git clone <repository_url>
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

### 1. Data Loading and Exploration

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
