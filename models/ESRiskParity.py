"""
Created on Thu Feb 24 2020

@author: ShinR
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm

class ESRiskParity:
    """
    Expected Shortfall (CVaR) portfolio
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def price_to_log_return(self, timeseries):
        """
        timeseries: 2099 * 23 numpy ndarray
        return: 2098 * 23 numpy ndarray
        """
        log_return = np.diff(np.log(timeseries), axis=0)
        return log_return
    
    def calculate_miu_cov(self, timeseries, return_timeseries):
        if not return_timeseries:
            timeseries = self.price_to_log_return(timeseries)
        miu = np.mean(timeseries, axis=0)
        cov = np.cov(timeseries.T)
        return miu, cov
    
    def portfolio_std(self,  weight, cov):
        return np.sqrt(np.dot(np.dot(weight, cov), weight.T))

    def portfolio_risk(self, weight, miu, cov):
        """
        weight: 1 * 23 numpy ndarray
        miu: weight: 1 * 23 numpy ndarray
        cov: 23 * 23 numpy ndarray
        return a float
        """
        total_risk_of_portfolio = self.portfolio_std(weight, cov) * norm.pdf(norm.ppf(self.alpha)) / self.alpha - \
                                    np.dot(weight, miu)
        return total_risk_of_portfolio
    
    def marginal_risk_contribution(self, weight, miu, cov):
        """
        return a 23 * 1 numpy ndarray
        """
        ratio = np.dot(cov, weight.T) / self.portfolio_std(weight, cov) * norm.pdf(norm.ppf(self.alpha)) / self.alpha - miu
        risk_contribution = np.multiply(weight.T, ratio)  
        return risk_contribution
    
    def objection_error(self, weight, args):
        miu = args[0]
        cov = args[1]
        risk_target_percent = args[2]
        total_risk_of_portfolio = self.portfolio_risk(weight, miu, cov)
        risk_contribution = self.marginal_risk_contribution(weight, miu, cov)
        risk_target = np.multiply(risk_target_percent, total_risk_of_portfolio)
        error = np.sum(np.square(risk_contribution - risk_target))
        return error
    
    def get_signal(self, timeseries, initial_weights, risk_target_percent, return_timeseries, tol = 1e-10):
        miu, cov = self.calculate_miu_cov(timeseries, return_timeseries)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                        {'type': 'ineq', 'fun': lambda x: x})

        optimize_result = minimize(fun=self.objection_error,
                                    x0=initial_weights,
                                    args=[miu, cov, risk_target_percent],
                                    method='SLSQP',
                                    constraints=constraints,
                                    tol=tol,
                                    options={'disp': False})

        weight = optimize_result.x
        return weight

    def get_allocations(self, timeseries, return_timeseries=False, rolling_window = 14):
        allocations = np.zeros(timeseries.shape)
        initial_weights = [1 / timeseries.shape[1]] * timeseries.shape[1]
        risk_target_percent = [1 / timeseries.shape[1]] * timeseries.shape[1]
        for i in tqdm(range(rolling_window, timeseries.shape[0])):
            allocations[i,] = self.get_signal(timeseries[i-rolling_window:i+1,], initial_weights,
                                                risk_target_percent, return_timeseries)
        return allocations
    