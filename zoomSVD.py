# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:18:16 2020

@author: Kumar
"""
import sys
sys.path.append('/Users/Kumar/Desktop/PortfolioOptimization')
import warnings
warnings.filterwarnings('ignore')

import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler



def  incremental_svd(U,S,V,a):
    new_singular=np.dot(np.diag(S), V)
    shape1= U.shape[0]+1
    
    new_singular=np.concatenate((new_singular,a), axis=0)
    U_upd, S_upd, V_upd=  np.linalg.svd(new_singular,full_matrices=False)
    shape2= U_upd.shape[0]
    U_2=np.zeros((shape1, shape2))
    U_2[0:U.shape[0], 0:U.shape[1]]=U
    U_2[shape1-1, shape2-1]=1
    return np.dot(U_2, U_upd), S_upd, V_upd


def SVD_list(data,b):
        U_list=[]
        S_list=[]
        V_list=[]
        counter=0
        for i in range(0,len(data)):
            if(counter==0):
                 U, S, V = np.linalg.svd(data.iloc[i:i+1,:].values,full_matrices=False)
                 counter=1
            else:
                U, S, V= incremental_svd(U,S,V,data.iloc[i:i+1,:].values)
            if(U.shape[0]==b):
                U_list.append(U)
                S_list.append(S)
                V_list.append(V)
                counter=0
            elif(i==len(data)-1):
                U_list.append(U)
                S_list.append(S)
                V_list.append(V)
                counter=0
        return U_list, S_list ,V_list 
        

   
def  partial_svd(U_list,S_list,V_list,t1,t2, b):
    begin_index= int(t1/b)
    cut_begin= t1%b
    end_index=int(t2/b)
    cut_end=t2%b
    U_1, S_1, V_1= U_list[begin_index],S_list[begin_index] ,V_list[begin_index]
    U_2, S_2, V_2= U_list[end_index],S_list[end_index] ,V_list[end_index]
    U_1 = U_1[cut_begin:, :]
    U_2 = U_2[:cut_end, :]
    U_upd1, S_upd1, V_upd1 = np.linalg.svd(np.dot(U_1, np.diag(S_1)),full_matrices=False)
    U_upd2, S_upd2, V_upd2 = np.linalg.svd(np.dot(U_2, np.diag(S_2)),full_matrices=False)
    
    V_upd1 = np.dot(V_upd1 , V_1)
    V_upd2 = np.dot(V_upd2 , V_2)
    
    new_singular=np.dot(np.diag(S_upd1), V_upd1)
    
    for i in range(begin_index+1,end_index):
        temp_singular = np.dot(np.diag(S_list[i]), V_list[i])
        new_singular=np.vstack((new_singular,temp_singular))
          
    new_singular=np.vstack((new_singular,np.dot(np.diag(S_upd2), V_upd2)))
    
    U_upd_final, S_upd_final, V_upd_final = np.linalg.svd(new_singular,full_matrices=False)
    
    block_diagnol= np.zeros((t2-t1, U_upd_final.shape[0]))
    block_diagnol[0:U_upd1.shape[0],0:U_upd1.shape[1]]=U_upd1
    ro= U_upd1.shape[0]
    col=U_upd1.shape[1]
    for i in range(begin_index+1,end_index):
        block_diagnol[ro:ro+U_list[i].shape[0],col:col+U_list[i].shape[1]]=U_list[i]
        ro = ro+U_list[i].shape[0]
        col= col+U_list[i].shape[1]
        
        #temp_singular = np.multiply(U_list[begin_index+i+1], U_upd_final[b-cut_begin+b*(i):b-cut_begin+b*(i+1),:])
        #block_diagnol = np.vstack((block_diagnol,temp_singular))
     
    block_diagnol[ro:ro + U_upd2.shape[0], col:col + U_upd2.shape[1]]=U_upd2
    #temp_singular = np.dot(U_upd2, U_upd_final[0:U_upd2.shape[1],:])
   # block_diagnol = np.vstack((block_diagnol,temp_singular))
    block_diagnol=np.dot(block_diagnol, U_upd_final)
    return block_diagnol,  S_upd_final, V_upd_final 



def GetCov(Cov_dict,t1,t2,feature,b,i):
    data= Cov_dict[str(feature)]
    data=data.iloc[t1:t2]
    data1=np.array(data.cov())
    s=pd.Series(data1[i,:]) 
    s=(s-s.mean())/s.std()
    return s

def ZoomSVD(SVD_dict,t1,t2,feature,b,i):
    [U_list,S_list,V_list]= SVD_dict[str(feature)]
    block_diagnol,  S, V=partial_svd(U_list,S_list,V_list,t1,t2, b)
    #svd_encode=np.dot(np.diag(S), V.T)
    svd_encode= V
    s=pd.Series(svd_encode[i,:]) 
    s=(s-s.mean())/s.std()
    return s
#block_diagnol,  S_upd_final, V_upd_final = partial_svd(U_list,S_list,V_list,t1,t2, b)

def getSVDData(asset_dict,feature,codes):
    dfs=[]
    for j,i in enumerate(codes):
        data1=asset_dict[str(i)]
        data1=data1.reset_index()
        data1 = data1[['index', feature]]
        data1.columns=['Date',feature+str(j)]
        data1=data1.fillna(method='bfill')
        dfs.append(data1)
    
    dfs = [df.set_index('Date') for df in dfs]
    dfs=dfs[0].join(dfs[1:])
    dfs=dfs.reset_index()
    
    return dfs.drop(['Date'], axis=1)

#res= np.dot(np.diag(S_upd_final), V_upd_final)     
#a=list(res)
'''
def plot_examples(stock_input, stock_decoded):
    n = 10  
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, 1000, 100))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)
        
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)
'''
'''
res=pd.DataFrame(res)

ax = plt.subplot(2, n, i + 1)
if i == 0:
    ax.set_ylabel("Input", fontweight=600)
else:
    ax.get_yaxis().set_visible(False)
plt.plot(data.iloc[200:260,0:1])
ax.get_xaxis().set_visible(False)

ax = plt.subplot(2, n, i + 1 + n)
if i == 0:
    ax.set_ylabel("Output", fontweight=600)
else:
    ax.get_yaxis().set_visible(False)
plt.plot(a[1:2][0])
plt.plot(res.ix[:,0])
plt.plot(res.ix[0,:])
ax.get_xaxis().set_visible(False)
'''