import numpy as np
from scipy.stats import skew
import pandas as pd

def maxdrawdown(timeseries):
    i = np.argmax(np.maximum.accumulate(timeseries) - timeseries) # end of the period
    if i == 0:
        return 0
    j = np.argmax(timeseries[:i]) # start of period
    return timeseries[i] / timeseries[j] - 1


# def get_TW_performance(returns, risk_free = None):
#
#     def CAGR(x):
#         return np.cumprod(1 + x)[-1]**(252/len(x)) - 1
#
#     price = np.cumprod(1+returns)
#     print(price)
#     risk_free = (risk_free.groupby(risk_free.index.year).mean()).values
#     returns = ((1 + returns).groupby(returns.index.year).prod() - 1).values
#     print(returns)
#     assert len(returns) == len(risk_free)
#
#     stats = {}
#     stats["CAGR"] = round(CAGR(returns), 5)
#     stats["vol"] = round(np.std(returns)*np.sqrt(12), 5)
#     stats["Sharpe"] = round(CAGR(returns) / stats["vol"], 5)
#     stats["skew"] = round(float(skew(returns)), 5)
#     stats["maxDD"] = round(maxdrawdown(returns), 5)
#     return stats

def get_performance(return_value, acc_value, injection_dates, before_injection_dates, Time_weighted = True):

    portfolio_weights = get_weights(acc_value, injection_dates, before_injection_dates, Time_weighted)
    index, returns = get_returns(acc_value, injection_dates, before_injection_dates, portfolio_weights)
    std = get_vol(return_value, index, portfolio_weights)

    stats = {}
    stats["MVRR"] = returns
    stats["vol"] = std
    stats["Sharpe"] = stats["MVRR"] / stats["vol"]
    stats["skew"] = return_value.apply(skew).values
    stats["maxDD"] = return_value.apply(maxdrawdown).values
    df_stats = pd.DataFrame(stats)
    # returns = portfolio_value_half_yr_weight
    return df_stats

def get_weights(acc_value, injection_dates, before_injection_dates, Time_weighted = True):
    portfolio_value_half_yr_weight = acc_value.iloc[acc_value.index.isin([acc_value.index[0]] + injection_dates)]
    portfolio_value_half_yr_weight = portfolio_value_half_yr_weight.apply(lambda x: x/portfolio_value_half_yr_weight.sum(axis = 0),axis = 1).transpose().values
    size = portfolio_value_half_yr_weight.shape[1]-1
    if Time_weighted == True:
        return np.array([[1/size] * size] * portfolio_value_half_yr_weight.shape[0])
    portfolio_value_half_yr_weight = np.delete(portfolio_value_half_yr_weight, -1, axis=1)
    return portfolio_value_half_yr_weight

def get_returns(acc_value, injection_dates, before_injection_dates,weights):
    portfolio_value_half_yr = acc_value.iloc[acc_value.index.isin([acc_value.index[0]] + before_injection_dates + injection_dates)].sort_index()
    index = portfolio_value_half_yr.index
    portfolio_value_half_yr = portfolio_value_half_yr.reset_index(drop = True)
    if portfolio_value_half_yr.shape[0]%2 != 0:
        portfolio_value_half_yr.drop(portfolio_value_half_yr.tail(1).index,inplace=True)
    semi_annually_return = portfolio_value_half_yr.groupby(portfolio_value_half_yr.index // 2).apply(lambda x: (x.iloc[1] - x.iloc[0]) / x.iloc[0]).values
    returns = np.diag(weights.dot(semi_annually_return))
    vfunc_rt = np.vectorize(lambda x: (1 + x / 2)**2 - 1)
    returns = vfunc_rt(returns)
    return index, returns

def get_vol(return_value, index, weights):
    breaked_vol = return_value.groupby(pd.cut(return_value.index, index)).apply(lambda x: np.std(x)).dropna().values
    std = np.diag(weights.dot(breaked_vol))
    vfunc_std = np.vectorize(lambda x: x * np.sqrt(2))
    std = vfunc_std(std)
    return std
