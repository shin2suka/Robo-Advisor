import pandas as pd
import yfinance as yf
import numpy as np
from datetime import date
tickers=['AAPL', 'MCD', 'SPY']
start='2015-01-01'
end=date.today()
df = pd.read_csv("data/" + "low" + '.csv')
df
