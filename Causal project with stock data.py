"""Generated from Jupyter notebook: Causal project with stock data

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller


# Function to fetch stock data with error handling
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"Warning: No data returned for {ticker}")
            return None
        return data["Close"]
    except Exception as e:
        print(f"Failed to get ticker '{ticker}', reason: {str(e)}")
        return None


# Function to get data
def get_data():
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    xom = fetch_stock_data("XOM", start_date, end_date)  # ExxonMobil
    wti = fetch_stock_data("USO", start_date, end_date)  # WTI proxy

    # If either dataset is missing, return an empty DataFrame
    if xom is None or wti is None:
        print("Error: Missing data, exiting script.")
        return pd.DataFrame()

    # Combine into a DataFrame
    data = pd.DataFrame({"XOM": xom, "WTI": wti}).dropna()
    return data


# Get the data
data = get_data()

# Stop execution if data is missing
if data.empty:
    raise ValueError("No valid data available for analysis.")

# Plot the original data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["XOM"], label="ExxonMobil")
plt.plot(data.index, data["WTI"], label="WTI Crude")
plt.title("ExxonMobil Stock vs WTI Crude Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()


# Function to test stationarity
def check_stationarity(series, name):
    result = adfuller(series.dropna())  # Drop NaN values for ADF test
    print(f"{name}: p-value = {result[1]:.4f}")
