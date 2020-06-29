from preprocessing.data_wrangling import *
from configs.inputs import *
from backtest.backtest import *
from models.MVO import *
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta, date
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
    
    # USD/CAD account asset universe
    ASSET_UNIVERSE_USD = ["BND", "VTI", "EFA", "VWO"] 
    ASSET_UNIVERSE_CAD = ["XBB.TO", "XIU.TO", "XIN.TO", "XEM.TO"]

    #load data
    df = yf.download(ASSET_UNIVERSE_USD + ASSET_UNIVERSE_CAD, start='2010-01-01', end='2014-12-31')
    df = df['Adj Close']
    df.fillna(method = 'ffill', axis=0, inplace=True) #forward fill
    dt_list = df.index
    
    # rolling window & rebalance frequency (both in weeks)
    rolling_window = 52
    rebalance_freq = 12
    
    # two lists to record account value
    acc_CAD_val_list = list()
    acc_USD_val_list = list()
    
    # datetime 
    train_start_date = dt_list[0]
    train_end_date = dt_list[0] + timedelta(weeks = rolling_window)
    
    test_start_date = train_end_date + timedelta(days = 1)
    test_end_date = test_start_date + timedelta(weeks = rebalance_freq)
    
    n_rebalance_freq = len(pd.period_range(test_start_date, dt_list[-1], freq='Q'))
    
    # prepare data
    df_period_train = df[train_start_date: train_end_date]
    df_period_test = df[test_start_date: test_end_date]
    timeseriesUSD = df_period_train[ASSET_UNIVERSE_USD].values
    timeseriesCAD = df_period_train[ASSET_UNIVERSE_CAD].values
    
    # portfolio optimize
    mvoUSD = MVPort(timeseriesUSD)
    mvoCAD = MVPort(timeseriesCAD)
    weightUSD = mvoUSD.get_signal_ray(timeseriesUSD,target_return = 0.1, return_timeseries=False)
    weightCAD = mvoCAD.get_signal_ray(timeseriesCAD,target_return = 0.1, return_timeseries=False)
    
    # array to dict
    portfolio_USD = dict(zip(ASSET_UNIVERSE_USD, weightUSD))
    portfolio_CAD = dict(zip(ASSET_UNIVERSE_CAD, weightCAD))
    
    # initialize two accounts  
    accountUSD = Account("USD", 10000, portfolio_USD)
    accountCAD = Account("CAD", 10000, portfolio_CAD)
    
    for i in range(n_rebalance_freq):
        price_arr_dict_CAD = df_period_test[ASSET_UNIVERSE_CAD].to_dict('list')
        price_arr_dict_USD = df_period_test[ASSET_UNIVERSE_USD].to_dict('list')
        
        acc_gen_CAD = accountCAD.generate_market_value(price_arr_dict_CAD)
        acc_gen_USD = accountUSD.generate_market_value(price_arr_dict_USD)
        
        for j in range(len(df_period_test)):
            
            v1 = next(acc_gen_CAD)
            v2 = next(acc_gen_USD)
            
            acc_CAD_val_list.append(v1)
            acc_USD_val_list.append(v2)
            
            # FX ignored
            adjustment = breach_check(v1, v2)
            
            if adjustment:
                print("v1: "+ str(v1))
                print("v2: "+ str(v1))
                print("difference: "+ str((v1 - v2) / 2))
                print("adjustment amout: " + str(adjustment))
                if v1 > v2:
                    acc_gen_CAD.send(-adjustment)
                    acc_gen_USD.send(adjustment)
                else:
                    acc_gen_CAD.send(adjustment)
                    acc_gen_USD.send(-adjustment)
        
        # shift rolling window
        test_start_date = test_end_date + timedelta(days = 1)
        test_end_date = test_start_date + timedelta(weeks = rebalance_freq)
        train_start_date = test_start_date - timedelta(weeks = rolling_window)
        train_end_date = test_start_date - timedelta(days = 1)
        
        # re-slice data
        df_period_train = df[train_start_date: train_end_date]
        timeseriesUSD = df_period_train[ASSET_UNIVERSE_USD].values
        timeseriesCAD = df_period_train[ASSET_UNIVERSE_CAD].values
        df_period_test = df[test_start_date: test_end_date]
        
        # portfolio optimize
        mvoUSD = MVPort(timeseriesUSD)
        mvoCAD = MVPort(timeseriesCAD)
        weightUSD = mvoUSD.get_signal_ray(timeseriesUSD,target_return = 0.01, return_timeseries=False)
        weightCAD = mvoCAD.get_signal_ray(timeseriesCAD,target_return = 0.01, return_timeseries=False)
        print(weightUSD)
        print(weightCAD)
    
        # array to dict
        portfolio_USD = dict(zip(ASSET_UNIVERSE_USD, weightUSD))
        portfolio_CAD = dict(zip(ASSET_UNIVERSE_CAD, weightCAD))
        
        # feed new portfolios into accounts
        accountUSD.rebalance_active(portfolio_USD)
        accountCAD.rebalance_active(portfolio_CAD)
        
  
    plt.plot(acc_CAD_val_list, label = 'accountCAD')
    plt.plot(acc_USD_val_list, label = 'accountUSD')
    plt.legend()
    plt.show()
        

                
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
