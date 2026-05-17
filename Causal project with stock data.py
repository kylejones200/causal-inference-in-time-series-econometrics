"""Generated from Jupyter notebook: Causal project with stock data

Magics and shell lines are commented out. Run with a normal Python interpreter."""

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"{name}: p-value = {result[1]:.4f}")


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


def get_data():
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    xom = fetch_stock_data("XOM", start_date, end_date)
    wti = fetch_stock_data("USO", start_date, end_date)
    if xom is None or wti is None:
        print("Error: Missing data, exiting script.")
        return pd.DataFrame()
    data = pd.DataFrame({"XOM": xom, "WTI": wti}).dropna()
    return data


def main() -> None:
    data = get_data()

    if data.empty:
        raise ValueError("No valid data available for analysis.")

    plt.figure(figsize=(12, 6))

    plt.plot(data.index, data["XOM"], label="ExxonMobil")

    plt.plot(data.index, data["WTI"], label="WTI Crude")

    plt.title("ExxonMobil Stock vs WTI Crude Prices")

    plt.xlabel("Time")

    plt.ylabel("Price")

    plt.legend()

    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
