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
        return np.cumprod(1 + x)[-1]**(252/len(x)) - 1

    price = np.cumprod(1+returns)
    print(price)
    risk_free = (risk_free.groupby(risk_free.index.year).mean()).values
    returns = ((1 + returns).groupby(returns.index.year).prod() - 1).values

    assert len(returns) == len(risk_free)

    stats = {}
    stats["CAGR"] = round(CAGR(returns), 5)
    stats["vol"] = round(np.std(returns)*np.sqrt(12), 5)
    stats["Sharpe"] = round(CAGR(returns) / stats["vol"], 5)
    stats["skew"] = round(float(skew(returns)), 5)
    stats["maxDD"] = round(maxdrawdown(returns), 5)
    return stats

def get_MW_performance(portfolio_value, injective_capital, injection_cycle, injection_dates):
    # use numpy.irr calculate the IRR which is the get_MW_performance
    # outflow - inflow = 0
    # -100,000 - sum(10,000/(1 + IRR)**i) + end_value/(1 + IRR)**n
    # if injection_cycle == 6 months, IRR is compounding half-yearly
    initial_outflow = portfolio_value[0]
    ending_inflow = portfolio_value[-1]

    cash_flow = [-initial_outflow]
    cash_flow.extend([-injective_capital] * len(injection_dates))
    cash_flow.append(ending_inflow)

    compounding_return = numpy.irr(cash_flow)
    yearly_return = round((1 + compounding_return)**(12/injection_cycle),5)
    return yearly_return
