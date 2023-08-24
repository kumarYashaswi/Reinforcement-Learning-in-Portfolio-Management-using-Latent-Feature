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
import sys
sys.path.append('/Users/Kumar/Desktop/PortfolioOptimization')
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers
import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



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
 
           
        return data     
'''   
def RL_klfilter(data):
        
    
        kf = KalmanFilter(transition_matrices = [1],
                          observation_matrices = [1],
                          initial_state_mean = 0,
                          initial_state_covariance = 1,
                          observation_covariance=1,
                          transition_covariance=.01)
        
        state_means, _ = kf.filter(data)
        state_means=state_means.tolist()
        state_means=[x[0] for x in state_means]
        #state_means = pd.Series(state_means.flatten(), index=x.index)
        
        
        return pd.Series(state_means)
    
'''
 
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

def autoencodetrain():
            j=1
            window_length = 60

            #x_test = x_test.astype('float32')
            
            
            #x_test_simple = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
            input_window = Input(shape=(window_length,1))
            x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
            #x = BatchNormalization()(x)
            x = MaxPooling1D(2, padding="same")(x) # 5 dims
            encoded_conv = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
            encoder_conv = Model(input_window, encoded_conv)
            #x = BatchNormalization()(x)
            encoded1_conv = MaxPooling1D(2, padding="same")(encoded_conv) # 3 dims
            
            
            # 3 dimensions in the encoded layer
            
            x = Conv1D(1, 3, activation="relu", padding="same")(encoded1_conv) # 3 dims
            #x = BatchNormalization()(x)
            x = UpSampling1D(2)(x) # 6 dims
            x = Conv1D(16, 1, activation='relu')(x) # 5 dims
            #x = BatchNormalization()(x)
            x = UpSampling1D(2)(x) # 10 dims
            decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
            
            encoder_conv.load_weights(filepath="C:/Users/KUMAR YASHASWI/Documents/Reinforcement-learning-in-portfolio-management--master/Reinforcement-learning-in-portfolio-management--master/modelCNN.h5")

            
            return encoder_conv
'''
df1 = pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/final_data.csv')
del df1['Unnamed: 0']
df1.index =df1['Date']
df1 = df1.drop(['Date'], axis=1)
df = df1.pct_change().dropna()
df=df.loc[5:64, 'Close5']
a=pd.Series(decoded_stocks) 

encoder_conv.summary()

df1.shape
 df1=df.transpose()
            df1= np.expand_dims(df1, 2)
     len(b)       
            df1=np.reshape(df1, (1,60,1))  
'''
def autoencode(df,encoder_conv):
            df1= RL_klfilter(df)
            df1=df1*100
            df1=np.reshape(df1, (1,len(df),1))
            decoded_stocks = encoder_conv.predict(df1)
            decoded_stocks = decoded_stocks[0,:,0]
            return pd.Series(decoded_stocks)     
        
def ZoomState(df,encoder_conv):
            df1= RL_klfilter(df)
            df1=df1*100
            df1=np.reshape(df1, (1,len(df),1))
            decoded_stocks = encoder_conv.predict(df1)
            decoded_stocks = decoded_stocks[0,:,0]
            return pd.Series(decoded_stocks)        

