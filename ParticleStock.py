# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:11:02 2020

@author: Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:50:36 2020

@author: Kumar
"""
import pandas as pd
import math
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/Kumar/Desktop/SPIF-Project')
import warnings
warnings.filterwarnings('ignore')
import cvxopt as opt  
from cvxopt import blas, solvers  
from pykalman import KalmanFilter
import pandas as pd
from scipy import poly1d
from sklearn.metrics import mean_squared_error
from math import sqrt

kf = KalmanFilter(transition_matrices = [1],
                          observation_matrices = [1],
                          initial_state_mean = 0,
                          initial_state_covariance = 1,
                          observation_covariance=1,
                          transition_covariance=.01)
        
    # Use the observed values of the price to get a rolling mean
data =  pd.read_csv('/Users/Kumar/Desktop/PortfolioOptimization/final_data.csv')
x=data['TSLA'].pct_change().dropna()
state_means, _ = kf.filter(x.values)
state_means = pd.Series(state_means.flatten(), index=x.index)
# Compute the rolling mean with various lookback windows
#mean30 = pd.rolling_mean(x, 30)
#mean60 = pd.rolling_mean(x, 60)
#mean90 = pd.rolling_mean(x, 90)
# Plot original data and estimated mean
#plt.plot(state_means)
def StateEstimation(P_X, P_Y, M_0, P_0, N, y,T):
    
    x1_ = [0] * T
    x=np.zeros((N))
    #Initialization
    InitialParticle = np.random.normal(0, 1, N)  #Sampling fromInitial State Density
    x[:] = InitialParticle[:]  #Particles for Initial State 1
    x1_[0] = M_0
    x1= np.zeros((N))
    #Start Time Loop
    for t in range(1,T):
        #Prediction
        Snoise = np.random.normal(0, 0.01, N)
        for k in range(0,N):
            x1[k] = f1(x[k]) + Snoise[k]
        
        #Update
        PredError =  np.asarray([y[t]]* N) - g(x1)
        w= pe(PredError, P_Y)
        w=w/np.sum(w)
        
        #State Estimate
        x1_[t]= np.dot(w , x1)
        
        #Resampling
        x= x1 
        ind= resampling(w)
        x=x[ind]
        
    return  x1_ 


def resampling(q):
    qc = q.cumsum() 
    M = len(q)
    u = np.arange(M) 
    u = u + np.random.uniform(0,1)
    u = u/M
    i= [0]*M
    k=0
    for j in range(0,M):
        while(qc[k]<u[j]):
            k=k+1
        i[j]=k
    return i


N=100000
T=1759
M_0 = 0 
P_0 = 1
P_X = 0.01
P_Y = 1
x1 = [0] * T
x1[0]=0

y=data['TSLA'].pct_change().dropna()

def f1(x1):
    return x1
  

def g(x1):
    return x1

def pe(PredError, P_Y):
    w=[]
    for i in range(0,len(PredError)):
        w.append(math.exp(-(math.sqrt(abs(PredError[i]))/(2*P_Y)))/math.sqrt(2*3.14*P_Y))
    return np.array(w)
rmse=[]
for i in [1,10,100,500,1000,5000,10000,100000]:
    N=i
    x1_ = StateEstimation(P_X, P_Y, M_0, P_0, N, y, T)
    rms = sqrt(mean_squared_error(x1_[1:], state_means[1:]))
    rmse.append(rms)
    
plt.plot([10,100,500,1000,5000,10000,100000],rmse[1:])
plt.title('RMSE PF vs Kalman')
plt.xlabel('N_SIMULATONS_POINTS')
plt.ylabel('RMSE');

plt.plot(x1_)
plt.plot(state_means[1:])
plt.title('Pf performance')
plt.legend(['PF Estimate', 'Kalman State'])
plt.xlabel('timestep')
plt.ylabel('Returns');

plt.plot(y)
plt.plot(state_means)
plt.title('Kalman filter estimate of average')
plt.legend(['returns','Kalman Estimate'])
plt.xlabel('timestep')
plt.ylabel('returns');

plt.plot(y)
plt.plot(x1_)
plt.title('Paricle filter estimate')
plt.legend(['returns','PF Estimate'])
plt.xlabel('timestep')
plt.ylabel('returns');          


rms = sqrt(mean_squared_error(x1_, state_means))