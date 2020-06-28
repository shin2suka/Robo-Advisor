import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date


def download(tickers, risk_level, start='2018-01-01', end=date.today(), save=True):
    df = yf.download(tickers, start, end)
    if save:
        df.to_csv("data/" + risk_level + ".csv")
        print(risk_level + ".csv is saved in data")
    return df


def load_data(tickers, risk_level):
    df = download(tickers, risk_level, save=False)
    return df["Adj Close"]

