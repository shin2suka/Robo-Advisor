# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:17:22 2020

@author: zhongheng
"""

import pandas as pd
import numpy as np

class Account:
    def __init__(self, currency, init_invest, portfolio):
        self.__currency = currency
        self.__portfolio = portfolio
        self.__value = init_invest
        
    def generate_market_value(self, price_array_dict):
        """
        Parameters
        ----------
        price_array_dict : dict
            key: tickers
            value: price arrays (iterable)

        Returns
        -------
        a generator

        """
        price_matrix = np.array([price_array_dict[ticker] for ticker in price_array_dict.keys()])
        self.__portfolio = {k: self.__portfolio[k] for k in price_array_dict.keys()} # reorder the dict  
        weight_arr = np.array(list(self.__portfolio.values())).reshape(1, -1)
        amount_arr = np.divide(weight_arr * self.__value, price_matrix[:, 0].reshape(1, -1))
        i = 0
        while True:
            self.__value = np.dot(amount_arr, price_matrix[:, i])[0]
            adjustment = yield self.__value
            if adjustment:
                self.__value += adjustment
                amount_arr = np.divide(weight_arr * self.__value, price_matrix[:, i].reshape(1, -1))
            i += 1
                

    def rebalance_active(self, new_portfolio):
        """
        Parameters
        ----------
        new_portfolio : dict
            from portfolio optimizer

        Returns
        -------
            reset the accout's portfolio

        """
        self.__portfolio = new_portfolio
        
def breach_check(v1, v2):
    ratio = v1 / v2
    if ratio > 1.5 or ratio <= 2/3:
        return abs(v1 - v2) / 2
               
        
if __name__ == "__main__":
    _price_array_dict = {'A': np.array([1,2,3,4,5]), 'B': np.array([2,3,4,5,6])}
    a = Account("USD", 10000, {'A': 0.2, 'B': 0.8})
    a_gen = a.generate_market_value(_price_array_dict)
    while True:
        print(next(a_gen))
    
        
    
    
            
            
        
        
        
        
        
        
        
        
        
        
 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        