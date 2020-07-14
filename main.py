from preprocessing.data_wrangling import *
from configs.inputs import *
from backtest.backtest import *
from models.MVO import *
from models.Robust_MVO import *
from analyticTools.performance import *
from models.ERCRiskParity import ERCRiskParity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import yfinance as yf
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from scipy import stats
import calendar
import statsmodels.api as sm
import time
import pandas as pd
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import scatter_2d
# from pypfopt import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
# from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

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

def leverage_return(r, rf, leverage_ratio):
    return r + leverage_ratio * (r - rf)

def splitted_returns(df, timestamps):
    pre_timestamp = df.index[0]
    df_list = list()
    for i in range(len(timestamps)):
        df_split = df[(df.index >= pre_timestamp) & (df.index < timestamps[i])]
        df_list.append(df_split.pct_change())
        pre_timestamp = timestamps[i]
    df_list.append(df[(df.index >= pre_timestamp)].pct_change())
    return_df = pd.concat(df_list)
    return_df.dropna(axis=0, how="any", inplace=True)
    return return_df

def copula_sample_one_var_fixed(var_value, var_name, copula_obj):
    res = dict()
    tmp = zip(copula_obj.columns, copula_obj.univariates)
    tmp = [x for x in tmp if x[0] != var_name]
    for i, (column_name, univariate) in enumerate(tmp):
        cdf = stats.norm.cdf(var_value)
        j = univariate.percent_point(cdf).item(0)
        res[column_name] = j
    return res

def main():
    df = load_data(*_get_tickers())


if __name__ == "__main__":

    # USD/CAD account asset universe
    # real estate etf: VNQ(USD), XRE.TO(CAD)
    # Govn bond etf: XGB.TO(CAD), GOVT(USD)
    # high yield bond etf: XHY.TO(CAD), HYG(USD)
    # global infrastructure etf: ZGI.TO(CAD), IGF(USD)
    ASSET_UNI_USD_Credit = ["IEF", "HYG", "LQD"]
    ASSET_UNI_CAD_Credit = ["XGB.TO", "XHY.TO", "XIG.TO"]
    ASSET_UNI_CAD_Equity = ["XIU.TO", "XIN.TO", "XEM.TO"]
    ASSET_UNI_USD_Equity = ["VTI", "EFA", "VWO"] 
    RF_TICKER = ["^IRX"]

    # injection frequency = half year
    INJECTION_FREQ = 6

    # start date & end_date
    # investment period: 2015.4.1 - 2020.5.31
    # backtest period: 2010.4.1 - 2015.03.31
    start_date = '2014-10-01'
    end_date = '2020-05-31'

    #load data
    df = yf.download(ASSET_UNI_USD_Credit + ASSET_UNI_CAD_Credit + ASSET_UNI_CAD_Equity + ASSET_UNI_USD_Equity + RF_TICKER, start=start_date, end=end_date)      
    df = df['Adj Close']
    df["^IRX"] = df["^IRX"] / 100 
    dt_list = list(df.index)

    # FX
    df_fx = yf.download("CAD=X", start=start_date, end=end_date)
    df_fx = df_fx['Adj Close']

    # join two dataframes
    df = df.join(df_fx)
    df = df.rename(columns={"Adj Close": "FX"})

    # fill na (forward)
    df.fillna(method = 'ffill', axis=0, inplace=True) #forward fill

    # rolling window & rebalance frequency (both in months)
    rolling_window = 6
    rebalance_freq = 3

    # two lists to record account value
    acc_CAD_val_list = list()
    acc_USD_val_list = list()

    # datetime
    train_start_date = dt_list[0]
    train_end_date = train_start_date + relativedelta(months=rolling_window - 1)
    train_end_date = [x for x in dt_list if x.month == train_end_date.month and x.year == train_end_date.year][-1]

    test_start_date = train_end_date + relativedelta(months = 1)
    test_start_date =  [x for x in dt_list if x.month == test_start_date.month and x.year == test_start_date.year][0]
    _test_start_date_index = dt_list.index(test_start_date)

    test_end_date = test_start_date + relativedelta(months=rebalance_freq - 1)
    test_end_date = [x for x in dt_list if x.month == test_end_date.month and x.year == test_end_date.year][-1]

    n_rebalance_freq = len(pd.period_range(test_start_date, dt_list[-1], freq='Q'))

    # prepare data
    df_period_train = df[train_start_date: train_end_date]
    df_period_test = df[test_start_date: test_end_date]

    timeseriesUSDCredit = df_period_train[ASSET_UNI_USD_Credit].values
    timeseriesUSDEquity = df_period_train[ASSET_UNI_USD_Equity].values
    timeseriesCADCredit = df_period_train[ASSET_UNI_CAD_Credit].values
    timeseriesCADEquity = df_period_train[ASSET_UNI_CAD_Equity].values
    
    # portfolio optimize
    erc_rp = ERCRiskParity()
    # mvoUSD = MVPort(timeseriesUSD)
    # mvoCAD = MVPort(timeseriesCAD)
    # weightUSD = mvoUSD.get_signal_ray(timeseriesUSD,target_return = 0.1, return_timeseries=False)
    # weightCAD = mvoCAD.get_signal_ray(timeseriesCAD,target_return = 0.1, return_timeseries=False)

    initial_weights = [1 / timeseriesUSDEquity.shape[1]] * timeseriesUSDEquity.shape[1]
    risk_target_percent = [1 / timeseriesUSDEquity.shape[1]] * timeseriesUSDEquity.shape[1]
    weightUSDEquity = erc_rp.get_signal(timeseriesUSDEquity, initial_weights, risk_target_percent, False)
    weightUSDEquity *= 0.6
    
    initial_weights = [1 / timeseriesUSDCredit.shape[1]] * timeseriesUSDCredit.shape[1]
    risk_target_percent = [1 / timeseriesUSDCredit.shape[1]] * timeseriesUSDCredit.shape[1]
    weightUSDCredit = erc_rp.get_signal(timeseriesUSDCredit, initial_weights, risk_target_percent, False)
    weightUSDCredit *= 0.4
    
    initial_weights = [1 / timeseriesCADEquity.shape[1]] * timeseriesCADEquity.shape[1]
    risk_target_percent = [1 / timeseriesCADEquity.shape[1]] * timeseriesCADEquity.shape[1]
    weightCADEquity = erc_rp.get_signal(timeseriesCADEquity, initial_weights, risk_target_percent, False)
    weightCADEquity *= 0.6
    
    initial_weights = [1 / timeseriesCADCredit.shape[1]] * timeseriesCADCredit.shape[1]
    risk_target_percent = [1 / timeseriesCADCredit.shape[1]] * timeseriesCADCredit.shape[1]
    weightCADCredit = erc_rp.get_signal(timeseriesCADCredit, initial_weights, risk_target_percent, False)
    weightCADCredit *= 0.4
    
    # array to dict
    portfolio_USD = {**dict(zip(ASSET_UNI_USD_Credit, weightUSDCredit)), **dict(zip(ASSET_UNI_USD_Equity, weightUSDEquity))}
    portfolio_CAD = {**dict(zip(ASSET_UNI_CAD_Credit, weightCADCredit)), **dict(zip(ASSET_UNI_CAD_Equity, weightCADEquity))}
    
    # initialize two accounts  
    # substract the transaction cost
    accountUSD = Account("USD", 50000 / df_fx.loc[test_start_date] - len(portfolio_USD) * 5, portfolio_USD)
    accountCAD = Account("CAD", 50000 - len(portfolio_CAD) * 5, portfolio_CAD)

    # initialize injection dates
    injection_dates = []
    before_injection_dates = []

    for i in range(n_rebalance_freq):
        
        price_arr_dict_CAD = df_period_test[ASSET_UNI_CAD_Credit + ASSET_UNI_CAD_Equity].to_dict('list')
        price_arr_dict_USD = df_period_test[ASSET_UNI_USD_Credit + ASSET_UNI_USD_Equity].to_dict('list')

        FX_arr = df_period_test["FX"].values

        acc_gen_CAD = accountCAD.generate_market_value(price_arr_dict_CAD)
        acc_gen_USD = accountUSD.generate_market_value(price_arr_dict_USD)

        for j in range(len(df_period_test)):

            v1 = next(acc_gen_CAD)
            v2 = next(acc_gen_USD)

            acc_CAD_val_list.append(v1)
            acc_USD_val_list.append(v2)
        
            v2_cad = v2 * FX_arr[j]
            
            adjustment = breach_check(v1, v2_cad)
            
            if adjustment:
                print("v1: "+ str(v1))
                print("v2: "+ str(v2_ad))
                print("difference: "+ str((v1 - v2_cad) / 2))
                print("adjustment amout: " + str(adjustment))
                if v1 > v2_cad:
                    acc_gen_CAD.send(-adjustment)
                    acc_gen_USD.send(adjustment)
                else:
                    acc_gen_CAD.send(adjustment)
                    acc_gen_USD.send(-adjustment)

        # avoid redundant loop
        if i == n_rebalance_freq - 1:
            break

        # shift rolling window
        test_start_date = test_end_date + relativedelta(months = 1)

        test_start_date =  [x for x in dt_list if x.month == test_start_date.month and x.year == test_start_date.year][0]
        # print(test_start_date)
        test_end_date = test_start_date + relativedelta(months = rebalance_freq - 1)
        test_end_date = [x for x in dt_list if x.month == test_end_date.month and x.year == test_end_date.year]
        test_end_date = test_end_date[-1] if test_end_date else dt_list[-1]

        # print(test_end_date)
        train_end_date = test_start_date - relativedelta(months = 1)
        train_end_date = [x for x in dt_list if x.month == train_end_date.month and x.year == train_end_date.year][-1]
        train_start_date = train_end_date - relativedelta(months = rolling_window - 1)
        train_start_date = [x for x in dt_list if x.month == train_start_date.month and x.year == train_start_date.year][0]

        # re-slice data
        df_period_train = df[train_start_date: train_end_date]
        timeseriesUSDCredit = df_period_train[ASSET_UNI_USD_Credit].values
        timeseriesUSDEquity = df_period_train[ASSET_UNI_USD_Equity].values
        timeseriesCADCredit = df_period_train[ASSET_UNI_CAD_Credit].values
        timeseriesCADEquity = df_period_train[ASSET_UNI_CAD_Equity].values
        df_period_test = df[test_start_date: test_end_date]

        # portfolio optimize

        initial_weights = [1 / timeseriesUSDEquity.shape[1]] * timeseriesUSDEquity.shape[1]
        risk_target_percent = [1 / timeseriesUSDEquity.shape[1]] * timeseriesUSDEquity.shape[1]
        weightUSDEquity = erc_rp.get_signal(timeseriesUSDEquity, initial_weights, risk_target_percent, False)
        weightUSDEquity *= 0.6
        
        initial_weights = [1 / timeseriesUSDCredit.shape[1]] * timeseriesUSDCredit.shape[1]
        risk_target_percent = [1 / timeseriesUSDCredit.shape[1]] * timeseriesUSDCredit.shape[1]
        weightUSDCredit = erc_rp.get_signal(timeseriesUSDCredit, initial_weights, risk_target_percent, False)
        weightUSDCredit *= 0.4
        
        initial_weights = [1 / timeseriesCADEquity.shape[1]] * timeseriesCADEquity.shape[1]
        risk_target_percent = [1 / timeseriesCADEquity.shape[1]] * timeseriesCADEquity.shape[1]
        weightCADEquity = erc_rp.get_signal(timeseriesCADEquity, initial_weights, risk_target_percent, False)
        weightCADEquity *= 0.6
        
        initial_weights = [1 / timeseriesCADCredit.shape[1]] * timeseriesCADCredit.shape[1]
        risk_target_percent = [1 / timeseriesCADCredit.shape[1]] * timeseriesCADCredit.shape[1]
        weightCADCredit = erc_rp.get_signal(timeseriesCADCredit, initial_weights, risk_target_percent, False)
        weightCADCredit *= 0.4
        
        # array to dict
        portfolio_USD = {**dict(zip(ASSET_UNI_USD_Credit, weightUSDCredit)), **dict(zip(ASSET_UNI_USD_Equity, weightUSDEquity))}
        portfolio_CAD = {**dict(zip(ASSET_UNI_CAD_Credit, weightCADCredit)), **dict(zip(ASSET_UNI_CAD_Equity, weightCADEquity))}
        #weightCAD = initial_weights
        
        # previous portfolio weight
        pre_weight_CAD = accountCAD.get_weight()
        pre_weight_USD = accountUSD.get_weight()

        # calculate t cost for each turnover
        portf_USD_diff = {key: abs(pre_weight_USD[key] - portfolio_USD[key]) for key in portfolio_USD.keys()}
        portf_CAD_diff = {key: abs(pre_weight_CAD[key] - portfolio_CAD[key]) for key in portfolio_CAD.keys()}
        
        temp1 = sum(portf_USD_diff.values())
        temp2 = sum(portf_CAD_diff.values())
        t_cost_usd = len(temp1[temp1>0.001]) * 5
        t_cost_cad = len(temp2[temp2>0.001]) * 5

        accountCAD.transaction_cost(t_cost_cad)
        accountUSD.transaction_cost(t_cost_usd)

        # feed new portfolios into accounts
        accountUSD.rebalance_active(portfolio_USD)
        accountCAD.rebalance_active(portfolio_CAD)

        # injection (half year)
        if i % (INJECTION_FREQ / rebalance_freq) == 1:
            acc_gen_CAD.send(5000)
            acc_gen_USD.send(5000 / df_fx.loc[test_start_date])
            injection_dates.append(test_start_date)
            # print("injection = " + str(i))

        if i % (INJECTION_FREQ / rebalance_freq) == 0:
            before_injection_dates.append(test_end_date)


    test_time_period = dt_list[_test_start_date_index:]

    acc_value_df = pd.DataFrame({"accountCAD": acc_CAD_val_list, "accountUSD": acc_USD_val_list}, index = test_time_period)

    # convert USD to CAD
    acc_value_df['accountUSD_CADHDG'] = acc_value_df["accountUSD"].multiply(df['FX'][test_time_period[0]:])
    acc_value_df['portfolio'] = acc_value_df['accountUSD_CADHDG'] + acc_value_df['accountCAD']


    
#%%

    # calculate daily return

    acc_ret_df_d = splitted_returns(acc_value_df, injection_dates)
    
    # calculate quartly return
    acc_ret_df_q = acc_value_df.groupby(pd.PeriodIndex(acc_value_df.index, freq='Q'), axis=0).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
    
    
    
#%% calculate daily leveraged return 
     
    # read libor overnight rate
    Libor_overnight = pd.read_csv("C:\\Users\\zhong\\Documents\\MMF\\Risk Management Lab\\Robo-Advisor\\data\\LIBOR_OVERNIGHT.csv", header=0)
    Libor_overnight.index = Libor_overnight["DATE"]
    del Libor_overnight["DATE"]
    Libor_overnight.replace('.', np.nan, inplace=True)
    Libor_overnight.fillna(method='ffill', inplace=True, axis=0)
    Libor_overnight = Libor_overnight.astype(float)
    Libor_overnight = Libor_overnight.groupby(pd.PeriodIndex(Libor_overnight.index, freq='Q'), axis=0).first()
    
    # acc_ret_df_d = acc_ret_df_d.join(Libor_overnight)
    # leverage_ratio = 0.2
    # acc_ret_df_d['lev_portf_ret'] = acc_ret_df_d['portfolio'] + leverage_ratio * (acc_ret_df_d['portfolio'] - acc_ret_df_d['LIBOR_OVERNIGHT'] / 100)

    # try get_performance
    print(get_performance(acc_ret_df_d, acc_value_df, injection_dates, before_injection_dates, True))
    
    #calculate leveraged quarterly return
    acc_ret_df_q = acc_ret_df_q.join(Libor_overnight)
    leverage_ratio = 0.2
    acc_ret_df_q['lev_portf_ret'] = acc_ret_df_q['portfolio'] + leverage_ratio * (acc_ret_df_q['portfolio'] - acc_ret_df_q['LIBOR_OVERNIGHT'] / 100)
    
    acc_value_df.plot()
    plt.legend()
    plt.show()
    
#%%
    # factor analysis

    factor_df = pd.read_excel('C:\\Users\\zhong\\Documents\\MMF\\Risk Management Lab\\Robo-Advisor\\data\\factor data.xlsx', header=0)
    factor_df.index = factor_df['DATE']
    del factor_df['DATE']

    # factor return
    factor_ret_df = factor_df.pct_change()
    factor_ret_df.dropna(axis=0, how="any", inplace=True)

    # factor model
    X = sm.add_constant(factor_ret_df.loc['2011-04-01': '2020-04-01'])
    Y = acc_ret_df_q['portfolio'].values
    model = sm.OLS(Y,X).fit()
    print(model.summary())

    # scenario analysis

    # scenario 1: 1997-1999 Asia crisis and Russia sovereign boond default
    # time period: '1997-01-01', '1999-01-01'
    factor_s1 = factor_ret_df.loc['1997-01-01': '1999-01-01']
    portf_ret_pred = model.predict(sm.add_constant(factor_s1))
    portf_ret_pred = portf_ret_pred.groupby(pd.PeriodIndex(portf_ret_pred.index, freq='Q'), axis=0).min()
    portf_ret_pred = portf_ret_pred.to_frame('return')
    portf_ret_pred['group'] = 1

    # scenario 2: 911
    # time period: '2001-07-01', '2001-10-01'
    factor_s2 = factor_ret_df.loc['2001-07-01': '2001-10-01']
    portf_ret_pred_i = model.predict(sm.add_constant(factor_s2))
    portf_ret_pred_i = portf_ret_pred_i.groupby(pd.PeriodIndex(portf_ret_pred_i.index, freq='Q'), axis=0).min()
    portf_ret_pred_i = portf_ret_pred_i.to_frame('return')
    portf_ret_pred_i['group'] = 2
    portf_ret_pred = portf_ret_pred.append(portf_ret_pred_i)
    # portf_ret_pred.groupby(pd.PeriodIndex(portf_ret_pred.index, freq='Q'), axis=0).min().plot(kind='bar')

    # scenario 3: great financial crisis
    # time period: '2007-10-01', '2009-04-01'
    factor_s3 = factor_ret_df.loc['2007-10-01': '2009-04-01']
    portf_ret_pred_i = model.predict(sm.add_constant(factor_s3))
    portf_ret_pred_i = portf_ret_pred_i.groupby(pd.PeriodIndex(portf_ret_pred_i.index, freq='Q'), axis=0).min()
    portf_ret_pred_i = portf_ret_pred_i.to_frame('return')
    portf_ret_pred_i['group'] = 3
    portf_ret_pred = portf_ret_pred.append(portf_ret_pred_i)
    # portf_ret_pred.groupby(pd.PeriodIndex(portf_ret_pred.index, freq='Q'), axis=0).min().plot(kind='bar')

    # scenrio 4: covid-19
    # time period: '2020-01-01', '2020-04-01'
    factor_s4 = factor_ret_df.loc['2020-01-01': '2020-04-01']
    portf_ret_pred_i = model.predict(sm.add_constant(factor_s4))
    portf_ret_pred_i = portf_ret_pred_i.groupby(pd.PeriodIndex(portf_ret_pred_i.index, freq='Q'), axis=0).min()
    portf_ret_pred_i = portf_ret_pred_i.to_frame('return')
    portf_ret_pred_i['group'] = 4
    portf_ret_pred = portf_ret_pred.append(portf_ret_pred_i)

    colors = {1: '#833ab4', 2: '#fd1d1d', 3: '#fcb045', 4: '#fc466b'}

    s1 = mpatches.Patch(color='#833ab4', label='scenario 1')
    s2 = mpatches.Patch(color='#fd1d1d', label='scenario 2')
    s3 = mpatches.Patch(color='#fcb045', label='scenario 3')
    s4 = mpatches.Patch(color='#fc466b', label='scenario 4')
    plt.legend(handles=[s1, s2, s3, s4], loc=2)
    ax = (100 * portf_ret_pred['return']).plot(kind='bar', color=[colors[i] for i in portf_ret_pred['group']], rot =45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
