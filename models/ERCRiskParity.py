"""
Created on Thu Feb 20 2020

@author: ShinR
"""
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

class ERCRiskParity:
    """
    Simple Equal-Risk Contribution (ERC) portfolio
    """
    def __init__(self):
        pass

    def price_to_log_return(self, timeseries):
        log_return = np.diff(np.log(timeseries), axis=0)
        return log_return
    
    def calculate_cov_matrix(self, timeseries, return_timeseries):
        if not return_timeseries:
            timeseries = self.price_to_log_return(timeseries)
        cov = np.cov(timeseries.T)
        return cov

    def portfolio_risk(self, weight, cov):
        total_risk_of_portfolio = np.sqrt(np.dot(np.dot(weight, cov), weight.T))
        return total_risk_of_portfolio
    
    def marginal_risk_contribution(self, weight, cov):
        ratio = np.dot(cov, weight.T) / self.portfolio_risk(weight, cov)
        risk_contribution = np.multiply(weight.T, ratio)  
        return risk_contribution
    
    def objection_error(self, weight, args):
        cov = args[0]
        risk_target_percent = args[1]
        total_risk_of_portfolio = self.portfolio_risk(weight, cov)
        risk_contribution = self.marginal_risk_contribution(weight, cov)
        risk_target = np.multiply(risk_target_percent, total_risk_of_portfolio)
        error = np.sum(np.square(risk_contribution - risk_target))
        return error
    
    def get_signal(self, timeseries, initial_weights, risk_target_percent, return_timeseries, tol = 1e-10):
        cov = self.calculate_cov_matrix(timeseries, return_timeseries)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                        {'type': 'ineq', 'fun': lambda x: x})

        optimize_result = minimize(fun=self.objection_error,
                                    x0=initial_weights,
                                    args=[cov, risk_target_percent],
                                    method='SLSQP',
                                    constraints=constraints,
                                    tol=tol,
                                    options={'disp': False})

        weight = optimize_result.x
        return weight

    def get_allocations(self, timeseries, return_timeseries=False, rolling_window = 50):
        allocations = np.zeros(timeseries.shape)
        initial_weights = [1 / timeseries.shape[1]] * timeseries.shape[1]
        risk_target_percent = [1 / timeseries.shape[1]] * timeseries.shape[1]
        for i in tqdm(range(rolling_window, timeseries.shape[0])):
            allocations[i,] = self.get_signal(timeseries[i-rolling_window:i+1,], initial_weights,
                                                risk_target_percent, return_timeseries)
        return allocations

if __name__ == "__main__":
    import yfinance as yf
    import matplotlib.pyplot as plt
    tickers = ["AAPL", "TSLA", "SPY"]
    data = yf.download(tickers, start="2011-01-01")    
    timeseries = data[["Adj Close"]].values
    erc_rp = ERCRiskParity()
    allocations = erc_rp.get_allocations(timeseries)
    plt.plot(np.mean(np.multiply(allocations, timeseries), axis=1))
    