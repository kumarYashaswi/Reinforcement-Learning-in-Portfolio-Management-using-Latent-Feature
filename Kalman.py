# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 00:29:35 2020

@author: Kumar
"""
import sys
sys.path.append('/Users/Kumar/Desktop/SPIF-Project')
import warnings
warnings.filterwarnings('ignore')
import numpy as np  
import matplotlib.pyplot as plt  
import cvxopt as opt  
from cvxopt import blas, solvers  
from pykalman import KalmanFilter
import pandas as pd
import matplotlib.pyplot as plt
from scipy import poly1d
np.random.seed(123)


def klfilter(data):
        
    
        kf = KalmanFilter(transition_matrices = [1],
                          observation_matrices = [1],
                          initial_state_mean = 0,
                          initial_state_covariance = 1,
                          observation_covariance=1,
                          transition_covariance=.01)
        
    # Use the observed values of the price to get a rolling mean
        for col in data.columns:
            x=data[col]
            state_means, _ = kf.filter(x.values)
            state_means = pd.Series(state_means.flatten(), index=x.index)
            # Compute the rolling mean with various lookback windows
            #mean30 = pd.rolling_mean(x, 30)
            #mean60 = pd.rolling_mean(x, 60)
            #mean90 = pd.rolling_mean(x, 90)
            data[col]=state_means
            # Plot original data and estimated mean
            #plt.plot(state_means)
 
        plt.plot(x)
        plt.plot(state_means)
        plt.title('Kalman filter estimate of average')
        plt.legend(['Kalman Estimate', 'X'])
        plt.xlabel('Day')
        plt.ylabel('Price');    
        return data     
    
def RL_klfilter(data):
        
    
        kf = KalmanFilter(transition_matrices = [1],
                          observation_matrices = [1],
                          initial_state_mean = 0,
                          initial_state_covariance = 1,
                          observation_covariance=1,
                          transition_covariance=.01)
        
        state_means, _ = kf.filter(data)
        #state_means = pd.Series(state_means.flatten(), index=x.index)
        
        
        return state_means
 
def RL_klfilter(data):
        
    
        kf = KalmanFilter(transition_matrices = [1],
                          observation_matrices = [1],
                          initial_state_mean = 0,
                          initial_state_covariance = 1,
                          observation_covariance=1,
                          transition_covariance=.01)
        
        state_means, _ = kf.filter(data)
        #state_means = pd.Series(state_means.flatten(), index=x.index)
        
        
        return state_means     
            