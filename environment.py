# -*- coding: utf-8 -*-
"""
Created on Wed Mar 4 19:12:26 2020

@author: Kumar
"""
import sys
sys.path.append('/Users/Kumar/Desktop/PortfolioOptimization')
import numpy as np
import pandas as pd
import tsfresh
from utils import portfolio
from tsfresh import feature_extraction
import warnings
warnings.filterwarnings('ignore')
from autoencoder import *
from LSTMautoencode import *
from Kalman import *
from autoencodeTrain import *
from LSTMautoencoderTrain import *
from zoomSVD import *
import os
class CryptoEnvironment:
    
    def __init__(self, prices = '/Users/Kumar/Desktop/PortfolioOptimization/final_data.csv', 
                 prices1 = '/Users/Kumar/Desktop/PortfolioOptimization/val_data.csv',capital = 1e6):       
        self.prices = prices
        self.prices1 = prices1
        self.capital = capital  
        self.data, self.fil_data, self.data_test, self.fil_data_test = self.load_data()
        self.fil_data1=self.fil_data*100
        self.U_list, self.S_list, self.V_list = SVD_list(self.fil_data1)
        self.model = autoencodetrain(self.fil_data1)
        #self.model.save_weights("Users\Kumar\Desktop\PortfolioOptimization\modelCNN.h5")
        print('ok')
        self.model_lstm = LSTMautoencodetrain(self.fil_data1)
        #self.model_lstm.save_weights("modelLSTM.h5")
        
    def load_data(self):
        data =  pd.read_csv(self.prices)
        data_test =  pd.read_csv(self.prices1)
        del data['Unnamed: 0']
        del data_test['Unnamed: 0']
        data.index = data['Date']
        data = data.drop(['Date'], axis=1)
        data_test.index = data_test['Date']
        data_test = data_test.drop(['Date'], axis=1)
        df = data.pct_change().dropna()
        df1 = klfilter(df)
        df_test = data_test.pct_change().dropna()
        df_test1 = klfilter(df_test)
        return data, df1, data_test, df_test1
    
    def preprocess_state(self, state):
        return state
    
    def get_state(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        
        #decision_making_state = self.data.iloc[t-lookback:t]
        #decision_making_state = decision_making_state.pct_change().dropna()
        decision_making_state = self.fil_data.iloc[t-lookback:t]
        decision_making_state=decision_making_state*100
        if is_cov_matrix:
            x = decision_making_state.cov()
            return x
        else:
            if is_raw_time_series:
                decision_making_state = self.data.iloc[t-lookback:t]
            return self.preprocess_state(decision_making_state)
    
    def get_state2(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        decision_making_state = self.fil_data.iloc[t-lookback:t]
        decision_making_state=decision_making_state*100
        s=autoencode(decision_making_state,self.model)
        return pd.DataFrame(s)
    
    def get_state3(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        assert lookback <= t
        decision_making_state = self.fil_data.iloc[t-lookback:t]
        decision_making_state=decision_making_state*100
        U, S, V = partial_svd(self.U_list,self.S_list,self.V_list,t-lookback,t, 60)
        s=np.dot(np.diag(S), V)
        return pd.DataFrame(s).transpose()
    
    def get_state4(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        decision_making_state = self.fil_data.iloc[t-lookback:t]
        decision_making_state=decision_making_state*100
        s=LSTMautoencode(decision_making_state,self.model_lstm)
        return pd.DataFrame(s)
    
    def get_state5(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        decision_making_state = self.fil_data.iloc[t-lookback:t]
        decision_making_state=decision_making_state*100
        s=DNNautoencode(decision_making_state,self.model_lstm)
        return pd.DataFrame(s)
    
    
    def get_state_test(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        
        #decision_making_state = self.data.iloc[t-lookback:t]
        #decision_making_state = decision_making_state.pct_change().dropna()
        decision_making_state = self.fil_data_test.iloc[t-lookback:t]
        decision_making_state=decision_making_state*100
        if is_cov_matrix:
            x = decision_making_state.cov()
            return x
        else:
            if is_raw_time_series:
                decision_making_state = self.data.iloc[t-lookback:t]
            return self.preprocess_state(decision_making_state)
    
    def get_state_test2(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        decision_making_state = self.fil_data_test.iloc[t-lookback:t]
        decision_making_state=decision_making_state*100
        s=autoencode(decision_making_state,self.model)
        return pd.DataFrame(s)
    
    def get_statetest3(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        decision_making_state = self.fil_data_test.iloc[t-lookback:t]
        decision_making_state=decision_making_state*100
        U, S, V = partial_svd(self.U_list,self.S_list,self.V_list,t-lookback,t, 60)
        s=np.dot(np.diag(S), V)
        return pd.DataFrame(s).transpose()
            
    def get_state_test4(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        decision_making_state = self.fil_data_test.iloc[t-lookback:t]
        decision_making_state=decision_making_state*100
        s=LSTMautoencode(decision_making_state,self.model_lstm)
        return pd.DataFrame(s)
    
    def get_state_test5(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        decision_making_state = self.fil_data_test.iloc[t-lookback:t]
        decision_making_state=decision_making_state*100
        s=DNNautoencode(decision_making_state,self.model_lstm)
        return pd.DataFrame(s)    
    
    def get_reward(self, action, action_t, reward_t, alpha = 0.01):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])

        data_period = self.data[action_t:reward_t]
        weights = action
        returns = data_period.pct_change().dropna()
      
        sharpe = local_portfolio(returns, weights)[-1]
        sharpe = np.array([sharpe] * len(self.data.columns))          
        rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]
        
        return np.dot(returns, weights), rew
        
    def get_testreward(self, action, action_t, reward_t, alpha = 0.01):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])

        data_period = self.data_test[action_t:reward_t]
        weights = action
        returns = data_period.pct_change().dropna()
      
        sharpe = local_portfolio(returns, weights)[-1]
        sharpe = np.array([sharpe] * len(self.data.columns))          
        rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]
        
        return np.dot(returns, weights), rew
    
    
    def get_reward1(self, action, action_t, reward_t, alpha = 0.01):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])

        data_period = self.data[action_t:reward_t]
        weights = action
        returns = data_period.pct_change().dropna()
      
        sharpe = local_portfolio(returns, weights)[-1]
        sharpe = np.array([sharpe] * len(self.data.columns))          
        rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]
        std1= data_period.std()
        rew=rew/std1
        rew=rew*10
        rew=np.array(rew)
        return np.dot(returns, weights), rew
        
    def get_testreward1(self, action, action_t, reward_t, alpha = 0.01):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])

        data_period = self.data_test[action_t:reward_t]
        weights = action
        returns = data_period.pct_change().dropna()
      
        sharpe = local_portfolio(returns, weights)[-1]
        sharpe = np.array([sharpe] * len(self.data.columns))          
        rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]
        std1= data_period.std()
        rew=rew/std1
        rew=rew*10
        rew=np.array(rew)
        return np.dot(returns, weights), rew
    
    
class ETFEnvironment:
    
    def __init__(self, volumes = './data/volumes.txt',
                       prices = './data/prices.txt',
                       returns = './data/returns.txt', 
                       capital = 1e6):
        
        self.returns = returns
        self.prices = prices
        self.volumes = volumes   
        self.capital = capital  
        
        self.data = self.load_data()

    def load_data(self):
        volumes = np.genfromtxt(self.volumes, delimiter=',')[2:, 1:]
        prices = np.genfromtxt(self.prices, delimiter=',')[2:, 1:]
        returns=pd.read_csv(self.returns, index_col=0)
        assets=np.array(returns.columns)
        dates=np.array(returns.index)
        returns=returns.as_matrix()
        return pd.DataFrame(prices, 
             columns = assets,
             index = dates
            )
    
    def preprocess_state(self, state):
        return state
    
    def get_state(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        
        decision_making_state = self.data.iloc[t-lookback:t]
        decision_making_state = decision_making_state.pct_change().dropna()

        if is_cov_matrix:
            x = decision_making_state.cov()
            return x
        else:
            if is_raw_time_series:
                decision_making_state = self.data.iloc[t-lookback:t]
            return self.preprocess_state(decision_making_state)

    def get_reward(self, action, action_t, reward_t):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])
        
        weights = action
        returns = self.data[action_t:reward_t].pct_change().dropna()
        
        rew = local_portfolio(returns, weights)[-1]
        rew = np.array([rew] * len(self.data.columns))
        
        return np.dot(returns, weights), rew