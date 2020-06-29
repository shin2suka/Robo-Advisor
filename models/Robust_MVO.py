import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.optimize import Bounds
from scipy.stats import norm
from scipy.stats.mstats import gmean
from tqdm import tqdm
from numpy import linalg as LA
from scipy.stats.distributions import chi2
import math


class Robust_MVPort:
    """
    Robust Mean variance Optimization portfolio
    """
    def __init__(self, rtnM, alpha=0.95):
        self.rtnM = rtnM
        self.ep = np.sqrt(chi2.ppf(alpha, self.rtnM.shape[1]))
        # print("ep: \t %f"%(self.ep))

    def price_to_log_return(self, timeseries):
        log_return = np.diff(np.log(timeseries), axis=0)
        return log_return

    def calculate_miu_cov(self, timeseries, return_timeseries):
        if not return_timeseries:
            timeseries = self.price_to_log_return(timeseries)
        #miu = np.mean(timeseries, axis=0).reshape(1, -1)
        miu = (gmean(1+timeseries, axis=0)-1).reshape(1, -1)
        cov = np.cov(timeseries.T)
        return miu, cov

    def get_sqrtThe(self, timeseries, cov):
        Theta = np.diag( np.diag(cov) ) / (timeseries.shape[0]-1)
        return np.sqrt(Theta)

    def portfolio_std(self, weight, cov):
        return np.sqrt(np.dot(np.dot(weight, cov), weight.T))

    def objection_error(self, weight, args):
        miu = args[0]
        cov = args[1]
        total_risk_of_portfolio = self.portfolio_std(weight, cov)
        total_return_of_portfolio = (weight*miu).sum()
        error=(self.rtnM.mean()-total_return_of_portfolio)/total_risk_of_portfolio
        return error

    def get_signal(self, timeseries, lb, ub, initial_weights, return_timeseries, tol = 1e-10):
        miu, cov = self.calculate_miu_cov(timeseries, return_timeseries)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                        {'type': 'ineq', 'fun': lambda x: x})

        x0_bounds = (0, 1)
        x1_bounds = (0, 1)
        x2_bounds = (0, 1)
        x3_bounds = (0, 1)
        bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds]

        optimize_result = minimize(fun=self.objection_error,
                                    x0=initial_weights,
                                    args=[miu, cov],
                                    method='SLSQP',
                                    constraints=constraints,
                                    #bounds=bounds,
                                    tol=tol,
                                    options={'disp': False})

        weight = optimize_result.x
        return weight

    def get_allocations(self, timeseries, lb=0, ub=0, return_timeseries=False, rolling_window = 24):
        allocations = np.zeros(timeseries.shape)
        initial_weights = [1 / timeseries.shape[1]] * timeseries.shape[1]
        for i in tqdm(range(rolling_window, timeseries.shape[0])):
            allocations[i,] = self.get_signal(timeseries[i-rolling_window:i+1,], lb, ub, initial_weights, return_timeseries)
        return allocations

    def get_signal_ray(self, timeseries, target_return, return_timeseries, tol = 1e-10):
        initial_weights = [1 / timeseries.shape[1]] * timeseries.shape[1]
        miu, cov = self.calculate_miu_cov(timeseries, return_timeseries)
        sqrtThe = self.get_sqrtThe(timeseries,cov)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                        # {'type': 'ineq', 'fun': lambda x: x},
                        # {'type': 'ineq', 'fun': lambda x: 1 - x},
                        {'type': 'eq', 'fun': lambda x: np.dot(miu, x) - target_return - self.ep * LA.norm(np.dot(sqrtThe,x),2)})
        optimize_result = minimize(fun=self.portfolio_std,
                            x0=initial_weights,
                            args=cov,
                            method='SLSQP',
                            constraints=constraints,
                            bounds=[(0, 0.5) for i in range(timeseries.shape[1])],
                            tol=tol,
                            options={'disp': False})
        weight = optimize_result.x
        return weight



if __name__ == "__main__":
    import yfinance as yf
    import matplotlib.pyplot as plt
    tickers = ["CCL", "RCL", "MCD"]
    data = yf.download(tickers, start="2019-01-01")
    # print(data)
    timeseries = data[["Adj Close"]].values
    mvo = Robust_MVPort(timeseries,0.02,0.95)
    initial_weights = [1 / timeseries.shape[1]] * timeseries.shape[1]
    print(mvo.get_signal_ray(timeseries, target_return = 0.2, return_timeseries=False))
