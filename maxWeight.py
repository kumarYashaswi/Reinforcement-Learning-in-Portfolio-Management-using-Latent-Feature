# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:46:33 2020

@author: Kumar
"""
import sys
sys.path.append('/Users/Kumar/Desktop/PortfolioOptimization')
import warnings
warnings.filterwarnings('ignore')
import numpy as np  
import matplotlib.pyplot as plt  
import cvxopt as opt  
from cvxopt import blas, solvers  
import pandas as pd


# Turn off progress printing  
solvers.options['show_progress'] = False  


def maxweight(temp,ans,rebalance_period,window,size):
    
    temp=temp.pct_change().dropna()
    buy_signal=[]
    ret=[]
    vari=[]
    for i in range(rebalance_period,size,15):
        weights,co_var,mea,var=optimal_portfolio(temp,i,window)
        sharpe = [x/y for x, y in zip(map(float, mea), map(float, var))]
        weights=[0.0]*len(mea)
        weights[np.argmax(v)]=1.0
        w_= weights
        mea=np.array(mea)
        buy_signal.append(w_)
        ret.append(mea.dot(np.array(w_).transpose()))
        vari.append(np.array(w_).transpose().dot(co_var.dot(np.array(w_))))
    return buy_signal,ret,vari

    
def optimal_portfolio(data,t, lookback):
    assert lookback <= t
    
    returns=data.iloc[t-lookback:t]
    x=returns.cov()
    mea=list(returns.mean())
    var=list(returns.var())
    x_=pd.DataFrame(np.linalg.pinv(x.values), x.columns, x.index)
    e= np.ones([len(x),1], dtype = float) 
    
    denom=e.transpose().dot(x_.dot(e))
    wt=x_.dot(e)
    wt=wt/denom[0][0]
    
    
    return wt.ix[:,0],x.values,mea,var

def optimal_portfolio1(data,t, lookback):
    assert lookback <= t
    
    returns=data.iloc[t-lookback:t]
    returns=returns.pct_change().dropna()
    returns=returns.transpose()
    
    #returns=np.asarray(data_list, dtype=np.float64)
    n = len(returns)  
    returns = np.asmatrix(returns)  
    N = 1000  
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]  
    
    # Convert to cvxopt matrices  
    S = opt.matrix(np.cov(returns))  
    pbar = opt.matrix(np.mean(returns, axis=1))  
    
    # Create constraint matrices  
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix  
    h = opt.matrix(0.0, (n ,1))  
    A = opt.matrix(1.0, (1, n))  
    b = opt.matrix(1.0)  
    
    # Calculate efficient frontier weights using quadratic programming  
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']  
                  for mu in mus]  
    
    
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER  
    returns = [blas.dot(pbar, x) for x in portfolios]  
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]  
    
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE  
    m1 = np.polyfit(returns, risks, 2)  
    x1 = np.sqrt(m1[2] / m1[0]) 
    
    
    # CALCULATE THE OPTIMAL PORTFOLIO  
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  
    return np.asarray(wt), returns, risks


