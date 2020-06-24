import numpy as np
from scipy.stats import skew


def maxdrawdown(timeseries, start=0, end=None):
    if end:
        end = len(timeseries)
    i = np.argmax(np.maximum.accumulate(timeseries[start:end]) - timeseries[start:end]) # end of the period
    if i == 0:
        return 0 
    j = np.argmax(timeseries[start:end][:i]) # start of period
    return timeseries[start:end][i] / timeseries[start:end][j] - 1


def get_performance(returns, risk_free):
    
    def CAGR(x):
        return np.cumprod(1 + x)[-1]**(1/len(x)) - 1
    
    price = np.cumprod(1+returns)
    print(price)
    risk_free = (risk_free.groupby(risk_free.index.year).mean()).values
    returns = ((1 + returns).groupby(returns.index.year).prod() - 1).values
    
    assert len(returns) == len(risk_free)
    
    stats = {}
    stats["CAGR"] = round(CAGR(returns), 5)
    stats["vol"] = round(np.std(returns), 5)
    stats["Sharpe"] = round(CAGR(returns) / stats["vol"], 5)
    stats["skew"] = round(float(skew(returns)), 5)
    stats["maxDD"] = round(maxdrawdown(returns), 5)
    return stats
            