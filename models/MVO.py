import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm

class MVPort:
    """
    Mean variance Optimization portfolio
    """
    def __init__(self, rtnM):
        self.rtnM = rtnM

    def price_to_log_return(self, timeseries):
        log_return = np.diff(np.log(timeseries), axis=0)
        return log_return
    
    def calculate_miu_cov(self, timeseries, return_timeseries):
        if not return_timeseries:
            timeseries = self.price_to_log_return(timeseries)
        miu = np.mean(timeseries, axis=0)
        cov = np.cov(timeseries.T)
        return miu, cov
    
    def portfolio_std(self, weight, cov):
        return np.sqrt(np.dot(np.dot(weight, cov), weight.T))
    
    def objection_error(self, weight, args):
        miu = args[0]
        cov = args[1]
        total_risk_of_portfolio = self.portfolio_std(weight, cov)
        total_return_of_portfolio = (weight*miu).sum()
        error=(self.rtnM.mean()-total_return_of_portfolio*12)/total_risk_of_portfolio
        return error
    
    def get_signal(self, timeseries, initial_weights, return_timeseries, tol = 1e-10):
        miu, cov = self.calculate_miu_cov(timeseries, return_timeseries)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                        {'type': 'ineq', 'fun': lambda x: x})

        optimize_result = minimize(fun=self.objection_error,
                                    x0=initial_weights,
                                    args=[miu, cov],
                                    method='SLSQP',
                                    constraints=constraints,
                                    tol=tol,
                                    options={'disp': False})

        weight = optimize_result.x
        return weight

    def get_allocations(self, timeseries, return_timeseries=False, rolling_window = 24):
        allocations = np.zeros(timeseries.shape)*np.nan
        initial_weights = [1 / timeseries.shape[1]] * timeseries.shape[1]
        for i in tqdm(range(rolling_window, timeseries.shape[0])):
            allocations[i,] = self.get_signal(timeseries[i-rolling_window:i+1,], initial_weights, return_timeseries)
        return allocations
    
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/portfolio_returns.csv").dropna()
    df.head()
    df = df.drop(["Dates","HFRIFOF Index","USGG3M Index"],axis=1)
    mvo = MVPort(df.values)
    print(mvo.get_allocations(df.values, return_timeseries=True))
    