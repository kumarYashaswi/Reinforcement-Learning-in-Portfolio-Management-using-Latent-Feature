# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:12:33 2020

@author: Kumar
"""

import sys
sys.path.append('/Users/Kumar/Desktop/PortfolioOptimization')
import warnings
warnings.filterwarnings('ignore')
import tensorflow.keras

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers
import datetime
from pykalman import KalmanFilter
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
from Kalman import *
from environment import CryptoEnvironment, ETFEnvironment
from utils import *

feature='close'

data1 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/AAPL.csv')
data1 = data1[['Date', feature]]
data2 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/AXP.csv')
data2 = data2[['Date', feature]]
data3 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/BRK.csv')
data3 = data3[['Date', feature]]
data4 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/C.csv')
data4 = data4[['Date', feature]]
data5 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/GILD.csv')
data5 = data5[['Date', feature]]
data6 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/HON.csv')
data6 = data6[['Date', feature]]
data7 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/INTU.csv')
data7 = data7[['Date', feature]]
data8 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/JPM.csv')
data8 = data8[['Date', feature]]
data9 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/NKE.csv')
data9 = data9[['Date', feature]]
data10 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/NVDA.csv')
data10 = data10[['Date', feature]]
data11 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/ORCL.csv')
data11 = data11[['Date', feature]]
data12 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/PG.csv')
data12 = data12[['Date', feature]]
data13 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/UAL.csv')
data13 = data13[['Date', feature]]
data14 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/WMT.csv')
data14 = data14[['Date', feature]]
data15 =  pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/Datasets/XOM.csv')
data15 = data15[['Date', feature]]


dfs = [data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15 ]

dfs=[]
for j,i in enumerate(codes):
    data1=asset_dict[str(i)]
    data1=data1.reset_index()
    data1 = data1[['index', feature]]
    data1.columns=['Date',feature+str(j)]
    dfs.append(data1)
    
i=1
for df in dfs:
    df.columns=['Date',feature+str(i)]
    i=i+1
    


dfs = [df.set_index('Date') for df in dfs]
dfs=dfs[0].join(dfs[1:])
dfs=dfs.reset_index()


a=dfs

a.to_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/'+feature+'_data.csv')


feature='close'
def getSVDData(asset_dict,feature,codes):
    dfs=[]
    for j,i in enumerate(codes):
        data1=asset_dict[str(i)]
        data1=data1.reset_index()
        data1 = data1[['index', feature]]
        data1.columns=['Date',feature+str(j)]
        dfs.append(data1)
    
    dfs = [df.set_index('Date') for df in dfs]
    dfs=dfs[0].join(dfs[1:])
    dfs=dfs.reset_index()
    
    return dfs.drop(['Date'], axis=1)
    
    