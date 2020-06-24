from preprocessing.data_wrangling import *
from configs.inputs import *
from backtest.backtest import *
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
    
    # # Calculate expected returns and sample covariance
    # mu = expected_returns.mean_historical_return(df)
    # S = risk_models.sample_cov(df)

    # # Optimise for maximal Sharpe ratio
    # ef = EfficientFrontier(mu, S)
    # raw_weights = ef.max_sharpe()
    # cleaned_weights = ef.clean_weights()
    # ef.save_weights_to_file("weights.csv")  # saves to file
    # print(cleaned_weights)
    # ef.portfolio_performance(verbose=True)
    
    # latest_prices = get_latest_prices(df)
    # da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
    # allocation, leftover = da.lp_portfolio()
    # print("Discrete allocation:", allocation)
    # print("Funds remaining: ${:.2f}".format(leftover))

if __name__ == "__main__":
    account1 = Account("USD", 10000, {'AAPL': 0.2, 'MCD': 0.3, 'SPY': 0.5})
    account2 = Account("CAD", 10000, {'LK': 0.9, 'RCL': 0.1})
    df = load_data(*_get_tickers())
    price_arr_dict1 = df[['AAPL', 'MCD', 'SPY']].to_dict('list')
    price_arr_dict2 = df[['LK', 'RCL']].to_dict('list')
    
    acc_gen1 = account1.generate_market_value(price_arr_dict1)
    acc_gen2 = account2.generate_market_value(price_arr_dict2)
    
    account_value_list1 = list()
    account_value_list2 = list()
    breach_point_list = list()
    for i in range(100):
        v1 = next(acc_gen1)
        v2 = next(acc_gen2)
        adjustment = breach_check(v1, v2)
        if adjustment:
            if v1 > v2:
                acc_gen1.send(-adjustment)
                acc_gen2.send(adjustment)
            else:
                acc_gen1.send(adjustment)
                acc_gen2.send(-adjustment)
            breach_point_list.append((i,v1))
            breach_point_list.append((i,v2))
        account_value_list1.append(v1)
        account_value_list2.append(v2)
    plt.plot(account_value_list1, label = 'account1')
    plt.plot(account_value_list2, label = 'account2')
    plt.scatter(*zip(*breach_point_list))
    plt.legend()
    plt.show()
        

                
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
