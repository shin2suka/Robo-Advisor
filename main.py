from preprocessing.data_wrangling import *
from configs.inputs import *
from backtest.backtest import *
from models.MVO import *
from models.Robust_MVO import *
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

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def main():
    df = load_data(*_get_tickers())
    
    
if __name__ == "__main__":
    
    # USD/CAD account asset universe
    ASSET_UNIVERSE_USD = ["BND", "VTI", "EFA", "VWO", "USO", "GLD"] 
    ASSET_UNIVERSE_CAD = ["XBB.TO", "XIU.TO", "XIN.TO", "XEM.TO", "HOU.TO", "HUG.TO"]
    
    
    INJECTION_FREQ = 24

    #load data
    df = yf.download(ASSET_UNIVERSE_USD + ASSET_UNIVERSE_CAD, start='2010-01-01', end='2014-12-31')
    df = df['Adj Close']
    dt_list = df.index
    
    # FX
    df_fx = yf.download("CAD=X", start='2010-01-01', end='2014-12-31')
    df_fx = df_fx['Adj Close']
    
    # join two dataframes
    df = df.join(df_fx)
    df = df.rename(columns={"Adj Close": "FX"})
    
    # fill na (forward)
    df.fillna(method = 'ffill', axis=0, inplace=True) #forward fill
    
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
    # substract the transaction cost
    accountUSD = Account("USD", 50000 / df_fx.loc[test_start_date] - len(ASSET_UNIVERSE_USD) * 5, portfolio_USD)
    accountCAD = Account("CAD", 50000 - len(ASSET_UNIVERSE_CAD) * 5, portfolio_CAD)
    
    # initialize previous weights to calculate transaction cost
    # pre_weight_usd, pre_weight_cad = np.zeros(len(ASSET_UNIVERSE_USD)), np.zeros(len(ASSET_UNIVERSE_CAD))
    
    for i in range(n_rebalance_freq):
        price_arr_dict_CAD = df_period_test[ASSET_UNIVERSE_CAD].to_dict('list')
        price_arr_dict_USD = df_period_test[ASSET_UNIVERSE_USD].to_dict('list')
        FX_arr = df_period_test["FX"].values
        
        acc_gen_CAD = accountCAD.generate_market_value(price_arr_dict_CAD)
        acc_gen_USD = accountUSD.generate_market_value(price_arr_dict_USD)
        
        for j in range(len(df_period_test)):
            
            v1 = next(acc_gen_CAD)
            v2 = next(acc_gen_USD) * FX_arr[j]
            
            acc_CAD_val_list.append(v1)
            acc_USD_val_list.append(v2)
            
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
        mvoUSD = Robust_MVPort(timeseriesUSD)
        mvoCAD = Robust_MVPort(timeseriesCAD)
        weightUSD = mvoUSD.get_signal_ray(timeseriesUSD,target_return = 0.05, return_timeseries=False)
        weightCAD = mvoCAD.get_signal_ray(timeseriesCAD,target_return = 0.05, return_timeseries=False)
        
        # calculate transaction cost
        pre_weight_CAD = np.array(list(accountCAD.get_weight().values()))
        pre_weight_USD = np.array(list(accountUSD.get_weight().values()))
                
        # calculate t cost for each turnover
        temp1 = abs(weightUSD - pre_weight_USD)
        temp2 = abs(weightCAD - pre_weight_CAD)
        t_cost_usd = len(temp1[temp1>0.001]) * 5
        t_cost_cad = len(temp2[temp2>0.001]) * 5
        
        acc_gen_CAD.send(-t_cost_cad)
        acc_gen_USD.send(-t_cost_usd)
               
        # portfolio weights array to dict
        portfolio_USD = dict(zip(ASSET_UNIVERSE_USD, weightUSD))
        portfolio_CAD = dict(zip(ASSET_UNIVERSE_CAD, weightCAD))
    
        # feed new portfolios into accounts
        accountUSD.rebalance_active(portfolio_USD)
        accountCAD.rebalance_active(portfolio_CAD)
        
        # update previous weights
        # pre_weight_usd = weightUSD
        # pre_weight_cad = weightCAD
        
        if i % (INJECTION_FREQ / rebalance_freq) == 0 and i != 0:
            acc_gen_CAD.send(5000)
            try:
                acc_gen_USD.send(5000 / df_fx.loc[test_start_date])
            except KeyError:
                acc_gen_USD.send(5000 / df_fx.loc[nearest(df_fx.index, test_start_date)])
            print('injection = ' + str(i))
            
  
    plt.plot(acc_CAD_val_list, label = 'accountCAD')
    plt.plot(acc_USD_val_list, label = 'accountUSD')
    plt.legend()
    plt.show()
        

                
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
