from preprocessing.data_wrangling import *
from configs.inputs import *
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


def _get_tickers():
    if CLIENTS_INFO["risk_level"] == "high":
        tickers = HIGH_TICKERS
    elif CLIENTS_INFO["risk_level"] == "med":
        tickers = MED_TICKERS
    else:
        tickers = LOW_TICKERS
    return tickers, CLIENTS_INFO["risk_level"]


def main():
    df = load_data(*_get_tickers())
    
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimise for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ef.save_weights_to_file("weights.csv")  # saves to file
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)
    
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
    allocation, leftover = da.lp_portfolio()
    print("Discrete allocation:", allocation)
    print("Funds remaining: ${:.2f}".format(leftover))

if __name__ == "__main__":
    agent = main()
    
    # import os
    # import subprocess
    # subprocess.call(['python', os.path.join("/Users/shin/desktop/RL-Trading", 'visualization/app.py')])
