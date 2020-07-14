import numpy as np
from scipy.stats import skew
import pandas as pd


def maxdrawdown(timeseries):
    i = np.argmax(np.maximum.accumulate(timeseries) - timeseries) # end of the period
    if i == 0:
        return 0
    j = np.argmax(timeseries[:i]) # start of period
    return timeseries[i] / timeseries[j] - 1

def get_performance(return_value, acc_value, injection_dates, before_injection_dates, risk_free_rate, std_risk_free, Time_weighted = True):

    portfolio_weights = get_weights(acc_value, injection_dates, before_injection_dates, Time_weighted)
    index, returns = get_returns(acc_value, injection_dates, before_injection_dates, portfolio_weights)
    std = get_vol(return_value, index, portfolio_weights,isinstance(return_value, pd.Series))

    stats = {}
    stats["MVRR"] = returns - risk_free_rate
    stats["vol"] = std
    stats["Sharpe"] = stats["MVRR"] / stats["vol"]
    if isinstance(return_value, pd.Series) == False:
        stats["skew"] = return_value.apply(skew).values
        stats["maxDD"] = return_value.apply(maxdrawdown).values
    df_stats = pd.DataFrame(stats)
    return df_stats

def get_weights(acc_value, injection_dates, before_injection_dates, Time_weighted = True):
    portfolio_value_half_yr_weight = acc_value.iloc[acc_value.index.isin([acc_value.index[0]] + injection_dates)]
    portfolio_value_half_yr_weight = portfolio_value_half_yr_weight.apply(lambda x: x/portfolio_value_half_yr_weight.sum(axis = 0),axis = 1).transpose().values
    size = portfolio_value_half_yr_weight.shape[1]
    if Time_weighted == True:
        return np.array([[1/size] * size] * portfolio_value_half_yr_weight.shape[0])
    return portfolio_value_half_yr_weight

def get_returns(acc_value, injection_dates, before_injection_dates,weights):
    portfolio_value_half_yr = acc_value.iloc[acc_value.index.isin([acc_value.index[0]] + before_injection_dates + injection_dates + [acc_value.index[-1]])].sort_index()
    index = portfolio_value_half_yr.index
    portfolio_value_half_yr = portfolio_value_half_yr.reset_index(drop = True)
    semi_annually_return = portfolio_value_half_yr.groupby(portfolio_value_half_yr.index // 2).apply(lambda x: (x.iloc[1] - x.iloc[0]) / x.iloc[0]).values
    returns = np.diag(weights.dot(semi_annually_return))
    returns = returns * 2
    return index, returns

def get_vol(return_value, index, weights, is_series):

    breaked_vol = return_value.groupby(pd.cut(return_value.index, index)).apply(lambda x: np.std(x)).replace(0, np.nan).dropna().values
    breaked_vol = breaked_vol*np.sqrt(252)

    if is_series == True:
        breaked_vol = breaked_vol.reshape(breaked_vol.shape[0],1)
    std = np.diag(np.sqrt(np.square(weights).dot(np.square(breaked_vol))))

    return std
