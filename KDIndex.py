# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:12:37 2020

@author: Kumar
"""

import numpy as np  
import matplotlib.pyplot as plt  
import cvxopt as opt  
from cvxopt import blas, solvers  
import pandas as pd

np.random.seed(123)

# Turn off progress printing  
solvers.options['show_progress'] = False  

 

def rand_weights(n):  
        ''' Produces n random weights that sum to 1 '''  
        k = np.random.rand(n)  
        return k / sum(k)
 

def random_portfolio(returns):  
    '''  
    Returns the mean and standard deviation of returns for a random portfolio  
    '''

    p = np.asmatrix(np.mean(returns, axis=1))  
    w = np.asmatrix(rand_weights(returns.shape[0]))  
    C = np.asmatrix(np.cov(returns))  
    mu = w * p.T  
    sigma = np.sqrt(w * C * w.T)  
    # This recursion reduces outliers to keep plots pretty  
    if sigma > 2:  
        return random_portfolio(returns)  
    return mu, sigma  

def greater(w):
    
    if(w>0):
        return 1
   
    else:
        return 0
    
    
def optimal_portfolio(data,t, lookback):
    assert lookback <= t
    
    returns=data.iloc[t-lookback:t]
    x=returns.cov()
    mea=list(returns.mean())

    x_=pd.DataFrame(np.linalg.pinv(x.values), x.columns, x.index)
    e= np.ones([len(x),1], dtype = float) 
    
    denom=e.transpose().dot(x_.dot(e))
    wt=x_.dot(e)
    wt=wt/denom[0][0]
    
    
    return wt.ix[:,0],x.values,np.array(mea)

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

def covmean_portfolio(covariances, mean_returns):
    ''' returns an optimal portfolio given a covariance matrix and matrix of mean returns '''
    n = len(mean_returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    S = opt.matrix(covariances)  # how to convert array to matrix?  

    pbar = opt.matrix(mean_returns)  # how to convert array to matrix?

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    port_list = convert_portfolios(portfolios)
    
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    frontier_returns = [blas.dot(pbar, x) for x in portfolios]  
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios] 
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(frontier_returns, risks, 2)
    #print m1 # result: [ 159.38531535   -3.32476303    0.4910851 ]
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  

    return np.asarray(wt), frontier_returns, risks, port_list

weights, returns, risks = optimal_portfolio(ret)
plt.plot(stds, means, 'o')  
plt.ylabel('mean')  
plt.xlabel('std')  
plt.plot(risks, returns, 'y-o') 