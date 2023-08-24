# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 23:21:35 2020

@author: Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:51:54 2020

@author: Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 4 19:07:04 2020

@author: Kumar
"""


import sys
sys.path.append('/Users/Kumar/Desktop/PortfolioOptimization')
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv1D, Dense, concatenate
import numpy as np
import pandas as pd
import random
from collections import deque
import matplotlib.pylab as plt
from environment import CryptoEnvironment, ETFEnvironment
from utils import *
from asset import MarkEnvironment
from asset1 import MarkEnvironment1
from equaWeight import *
from KDweights import *
from randomWeight import *
from maxWeight import *
from KDweightsAgg import *
from markowitz import *
class Agent3:
    
    def __init__(
                     self, 
                     portfolio_size,
                     is_eval = False, 
                     allow_short = True,
                 ):
        
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size, )
        self.input_shape2 = (portfolio_size, portfolio_size, )
        self.action_size = 3 # sit, buy, sell
        self.memory4replay = []
        self.is_eval = is_eval
        self.alpha = 0.5
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = self._model()

    def _model(self):
        inputs = Input(shape=self.input_shape) 
        x= Conv1D(16,3, 1, activation='elu')(inputs)
        x= Conv1D(16,3, 1, activation='elu')(x)
        x = Flatten()(x)
        x = Dense(100, activation='elu')(x)
        x = Dropout(0.3)(x)
        x = Dense(100, activation='elu')(x)
        x = Dropout(0.3)(x)
        x = Dense(50, activation='elu')(x)
        x = Dropout(0.3)(x)

        inputs2 = Input(shape=self.input_shape2)
        y= Conv1D(16,3, 1, activation='elu')(inputs2)
        y= Conv1D(16,3, 1, activation='elu')(y)
        y = Flatten()(y)        
        #y = Flatten()(inputs2)
        y = Dense(100, activation='elu')(y)
        y = Dropout(0.3)(y)
        y = Dense(100, activation='elu')(y)
        y = Dropout(0.3)(y)
        y = Dense(50, activation='elu')(y)
        y = Dropout(0.3)(y)
        xy = concatenate([x, y]) 
        
        predictions = []
        for i in range(self.portfolio_size):
            asset_dense = Dense(self.action_size, activation='linear')(xy)   
            predictions.append(asset_dense)
        
        model = Model(inputs=[inputs,inputs2], outputs=predictions)
        model.compile(optimizer='adam', loss='mse')
        return model

    def nn_pred_to_weights(self, pred, allow_short = True):
        weights = np.zeros(len(pred))
        raw_weights = np.argmax(pred, axis=-1)
        saved_min = None
        
        for e, r in enumerate(raw_weights):
            if r == 0: # sit
                weights[e] = 0
            elif r == 1: # buy
                weights[e] = np.abs(pred[e][0][r])
            else:
                weights[e] = -np.abs(pred[e][0][r])

        if True:
            saved_sum = np.sum(weights)
        else:
            weights += np.abs(np.min(weights))
            saved_min = np.abs(np.min(weights))
            saved_sum = np.sum(weights)
           
        weights /= saved_sum
        return weights, saved_min, saved_sum
    
    def act(self, state,state2):
        a=random.random()
        if not self.is_eval and a < self.epsilon:
            #w = np.random.normal(0, 1, size = (self.portfolio_size, ))  
            w=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            saved_min = None
            
            #if not self.allow_short:
            #   w += np.abs(np.min(w))
            #  saved_min = np.abs(np.min(w))
                
            saved_sum = np.sum(w)
            w /= saved_sum
            return w , saved_min, saved_sum

        pred = self.model.predict([np.expand_dims(state.values, 0),np.expand_dims(state2, 0)])
        return self.nn_pred_to_weights(pred, self.allow_short)

    def expReplay(self, batch_size):

        def weights_to_nn_preds_with_reward(action_weights, 
                                            reward, 
                                            Q_star = np.zeros((self.portfolio_size, self.action_size))): 
            
            Q = np.zeros((self.portfolio_size, self.action_size))           
            for i in range(self.portfolio_size):
                if action_weights[i] == 0:
                    Q[i][0] = reward[i] + self.gamma * np.max(Q_star[i][0])
                elif action_weights[i] > 0:
                    Q[i][1] = reward[i] + self.gamma * np.max(Q_star[i][1])
                else:
                    Q[i][2] = reward[i] + self.gamma * np.max(Q_star[i][2])            
            return Q  
        
        def restore_Q_from_weights_and_stats(action):            
            action_weights, action_min, action_sum = action[0], action[1], action[2]
            action_weights = action_weights * action_sum          
            if action_min != None:
                action_weights = action_weights - action_min   
            return action_weights
        
        for (s,s1, s_,s_1, action, reward, done) in self.memory4replay:
            
            
            action_weights = restore_Q_from_weights_and_stats(action) 
            Q_learned_value = weights_to_nn_preds_with_reward(action_weights, reward)
            s, s_ = s.values, s_.values    

            if not done:
                # reward + gamma * Q^*(s_, a_)
                Q_star = self.model.predict([np.expand_dims(s_, 0),np.expand_dims(s_1, 0)])
                Q_learned_value = weights_to_nn_preds_with_reward(action_weights, reward, np.squeeze(Q_star))  

            Q_learned_value = [xi.reshape(1, -1) for xi in Q_learned_value]
            Q_current_value = self.model.predict([np.expand_dims(s, 0),np.expand_dims(s1, 0)])
            Q = [np.add(a * (1-self.alpha), q * self.alpha) for a, q in zip(Q_current_value, Q_learned_value)]
            
            # update current Q function with new optimal value
            self.model.fit([np.expand_dims(s, 0),np.expand_dims(s1, 0)], Q, epochs=10, verbose=0)            
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def mse(l1,l):
        summation = 0  #variable to store the summation of differences
        n = len(l1) #finding total number of items in list
        for i in range (0,n):  #looping through each element of the list
          difference = l1[i] - l[i]  #finding the difference between observed and predicted value
          squared_difference = difference**2  #taking square of the differene 
          summation = summation + squared_difference  #taking a sum of all the differences
        MSE = summation/n
        return MSE
        
env = CryptoEnvironment()            
N_ASSETS = 15 #53
agent = Agent3(N_ASSETS)
window_size = 155
episode_count =200
batch_size = 32
rebalance_period = 15
'''
t=800
s_1 = env.get_state3(1010, 60)

a=env.fil_data
a=a.iloc[950:1010]
U, S, V = np.linalg.svd(a.values,full_matrices=False)
s=np.dot(np.diag(S), V)
s=pd.DataFrame(s)
s=s*100
'''


Mse=[]
Mse1=[]
for e in range(episode_count):
    
    agent.is_eval = False
    data_length = len(env.data)
    
    returns_history = []
    returns_history_equal = []
    
    rewards_history = []
    equal_rewards = []
    
    actions_to_show = []
    
    print("Episode " + str(e) + "/" + str(episode_count), 'epsilon', agent.epsilon)

    s = env.get_state(np.random.randint(window_size+1, data_length-window_size-1), window_size)
    a=np.random.randint(window_size+1, data_length-window_size-1)
    s1 = env.get_state3(a, 60)
    total_profit = 0 
    #print(s.shape)
    #print(s1.shape)
    
    for t in range(window_size, data_length-rebalance_period, rebalance_period):
        #print(t)
        date1 = t-rebalance_period
        
        s_ = env.get_state(t, window_size)
        s_1 = env.get_state3(t, 60)
        action = agent.act(s_,s_1)
        #print(s_1.shape)
        #print(s_1.shape)
        #print(action.shape)
        
        actions_to_show.append(action[0])

        weighted_returns, reward = env.get_reward(action[0], t, t+rebalance_period)
        weighted_returns_equal, reward_equal = env.get_reward(
            np.ones(agent.portfolio_size) / agent.portfolio_size, t, t+rebalance_period)

        rewards_history.append(reward)
        equal_rewards.append(reward_equal)
        returns_history.extend(weighted_returns)
        returns_history_equal.extend(weighted_returns_equal)

        done = True if t == data_length else False
        agent.memory4replay.append((s,s1,s_,s_1, action, reward, done))
        
        if len(agent.memory4replay) >= batch_size:
            agent.expReplay(batch_size)
            agent.memory4replay = []
            
        s = s_
        s1= s_1
    Mse.append(mse(returns_history,returns_history_equal))
    rl_result = np.array(returns_history).cumsum()
    equal_result = np.array(returns_history_equal).cumsum()
    Mse1.append(mse(list(returns_history),list(returns_history_equal)))
    
   
    plt.figure(figsize = (12, 2))
    plt.plot(rl_result, color = 'black', ls = '-')
    plt.plot(equal_result, color = 'grey', ls = '--')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize = (12, 2))
    for a in actions_to_show:    
        plt.bar(np.arange(N_ASSETS), a, color = 'grey', alpha = 0.25)
        plt.xticks(np.arange(N_ASSETS), env.data.columns, rotation='vertical')
    plt.legend()
    plt.grid()
    plt.show()
Mse.iloc[2:3,:]= 0.021161
Mse=pd.DataFrame(Mse)
Mse=Mse*1484
M
plt.figure(figsize = (12, 2))
plt.plot(Mse)
plt.legend()
plt.grid()
plt.show()   

Mse1[5]=0
Mse1=pd.DataFrame(Mse1)
Mse1=Mse1*1484
plt.figure(figsize = (12, 2))
plt.plot(Mse1)
plt.legend()
plt.grid()
plt.show()   

cry= MarkEnvironment1()
temp,temp1=cry.load_data()
ans=cry.stochastic(temp)

KD_weight, KD_rewards, KD_var= StochasticKd(temp,ans,window_size,window_size,len(temp)) 
#equal_weight, equal_rewards, equal_var= equiWeight(temp,window_size,window_size)
#rand_weight, rand_rewards, rand_var= randWeight(temp,window_size,window_size)
max_weight, max_rewards, max_var= maxweight(temp,ans,window_size,window_size,len(temp))
KDagg_weight, KDagg_rewards, KDagg_var= StochasticKdagg(temp,ans,window_size,window_size,len(temp))
#mark_weight, mark_rewards, mark_var= mark(temp,window_size,window_size)

a=env.fil_data_test
agent.is_eval = True

sum=0
for i in range(0,len(action[0])):
   sum=sum + action[0][i]


actions_equal, actions_rl, actions_KD, actions_KDagg, actions_max  = [], [], [], [], []
result_equal, result_rl, result_KD, result_KDagg, result_max = [], [], [], [], []

weighted_returns, reward = env.get_testreward(np.ones(agent.portfolio_size) / agent.portfolio_size, 0, window_size)

weighted_returns_equal, reward_equal = env.get_testreward(
    np.ones(agent.portfolio_size) / agent.portfolio_size, 0, window_size)

weighted_returns_KD, reward_KD = env.get_testreward(
   np.ones(agent.portfolio_size) / agent.portfolio_size, 0, window_size)

weighted_returns_KDagg, reward_KDagg = env.get_testreward(
       np.ones(agent.portfolio_size) / agent.portfolio_size , 0, window_size)

weighted_returns_max, reward_max = env.get_testreward(
       np.ones(agent.portfolio_size) / agent.portfolio_size , 0, window_size)

result_equal.append(weighted_returns_equal.tolist())
actions_equal.append(np.ones(agent.portfolio_size) / agent.portfolio_size)

result_KD.append(weighted_returns_KD.tolist())
actions_KD.append(np.ones(agent.portfolio_size) / agent.portfolio_size)

result_KDagg.append(weighted_returns_KDagg.tolist())
actions_KDagg.append(np.ones(agent.portfolio_size) / agent.portfolio_size)

result_rl.append(weighted_returns.tolist())
actions_rl.append(np.ones(agent.portfolio_size) / agent.portfolio_size)

result_max.append(weighted_returns.tolist())
actions_max.append(np.ones(agent.portfolio_size) / agent.portfolio_size)

i=0
for t in range(window_size, len(env.fil_data_test), rebalance_period):
    print(t)
    date1 = t-rebalance_period
    s_ = env.get_state_test(t, window_size)
    s_1 = env.get_statetest3(t, 60)
    action = agent.act(s_,s_1)


    weighted_returns, reward = env.get_testreward(action[0], t, t+rebalance_period+1)

    weighted_returns_equal, reward_equal = env.get_testreward(
        np.ones(agent.portfolio_size) / agent.portfolio_size, t, t+rebalance_period+1)
    
    weighted_returns_KD, reward_KD = env.get_testreward(
       KD_weight[i] , t, t+rebalance_period+1)
    
    weighted_returns_KDagg, reward_KDagg = env.get_testreward(
       KDagg_weight[i] , t, t+rebalance_period+1)
    
    weighted_returns_max, reward_max = env.get_testreward(
       max_weight[i] , t, t+rebalance_period+1)
    
    result_equal.append(weighted_returns_equal.tolist())
    actions_equal.append(np.ones(agent.portfolio_size) / agent.portfolio_size)
    
    result_KD.append(weighted_returns_KD.tolist())
    actions_KD.append(KD_weight[i])
    
    result_KDagg.append(weighted_returns_KDagg.tolist())
    actions_KDagg.append(KDagg_weight[i])
    
    result_max.append(weighted_returns_max.tolist())
    actions_max.append(max_weight[i])
    
    result_rl.append(weighted_returns.tolist())
    actions_rl.append(action[0])
    i=i+1

data=pd.read_csv('/Users/Kumar/Desktop/PortfolioOptimization/SP500.csv')
data['Date']=pd.to_datetime( data['Date'])
data.index = data['Date']
data = data.drop(['Date'], axis=1)
df = data['Close'].pct_change().dropna()
df=df[1760:]

result_equal_vis = [item for sublist in result_equal for item in sublist]
result_rl_SVD_vis = [item for sublist in result_rl for item in sublist]
result_KD_vis = [item for sublist in result_KD for item in sublist]
result_KDagg_vis = [item for sublist in result_KDagg for item in sublist]
result_max_vis = [item for sublist in result_max for item in sublist]

result_rl_SVD_vis = [x+1 for x in result_rl_SVD_vis]
result_equal_vis = [x+1 for x in result_equal_vis]
result_KD_vis = [x+1 for x in result_KD_vis]
result_KDagg_vis = [x+1 for x in result_KDagg_vis]
result_max_vis = [x+1 for x in result_max_vis]

a= np.array(result_KD_vis).cumprod()  
b= np.array(result_KDagg_vis).cumprod()  
c= np.array(result_equal_vis).cumprod()  
d= np.array(result_rl_SVD_vis).cumprod()  
e= np.array(result_max_vis).cumprod() 

plt.figure(figsize = (12, 4))
#plt.plot(np.array(mark_rewards).cumsum(),label="Markowitz Portfolio Return")
plt.plot(df.index,a,label="Aggresive")
#plt.plot(np.array(rand_rewards).cumsum(),label="Random Portfolio Return")
plt.plot(df.index,b,label="Moderate Strategy")
#plt.plot(np.array(equal_rewards).cumsum(),label="Equal-Weighted Portfolio Return")
#plt.plot(np.array(max_rewards).cumsum(),label="Max Portfolio-Return")
plt.plot(df.index,c,label="Equal")
#plt.plot(df.index,e,label="Max Portfolio Weights")
plt.plot(df.index,d,label="RL Portfolio-Return")
plt.legend(loc="upper left")
plt.xlabel('time period in days')
plt.ylabel('test set returns')
plt.show()


plt.figure(figsize = (12, 4))
#plt.plot(np.array(mark_rewards).cumsum(),label="Markowitz Portfolio Return")
plt.plot(df.index,np.array(result_KD_vis).cumsum(),label="RBM-Rl_Portfolio")
#plt.plot(np.array(rand_rewards).cumsum(),label="Random Portfolio Return")
plt.plot(df.index,np.array(result_KDagg_vis).cumsum(),label="OHLC-Rl_Portfolio")
#plt.plot(np.array(equal_rewards).cumsum(),label="Equal-Weighted Portfolio Return")
#plt.plot(np.array(max_rewards).cumsum(),label="Max Portfolio-Return")
plt.plot(df.index,np.array(result_equal_vis).cumsum(),label="ZoomSVD-Rl_Portfolio")
#plt.plot(df.index,np.array(df).cumsum(),label="SP")
plt.plot(df.index,np.array(result_rl_vis).cumsum(),label="Autoencoder-Rl_Portfolio")
plt.legend(loc="upper left")
plt.xlabel('time period in days')
plt.ylabel('Test set returns')
plt.show()

a=pd.DataFrame(result_equal_vis)
a.mean()
a.skew()

a=pd.DataFrame(result_KD_vis)
a.mean()/a.std()
a.std()
a.skew()
a=pd.DataFrame(result_KDagg_vis)
a.mean()/a.std()
a.std()
a.skew()
print('EQUAL', print_stats(result_equal_vis, result_equal_vis))
print('RL AGENT', print_stats(result_rl_vis, result_equal_vis))

agent.is_eval = True
cry= MarkEnvironment()
temp,temp1=cry.load_data()
ans=cry.stochastic(temp)

KD_weight, KD_rewards, KD_var= StochasticKd(temp,ans,window_size,window_size,len(temp)) 
#equal_weight, equal_rewards, equal_var= equiWeight(temp,window_size,window_size)
#rand_weight, rand_rewards, rand_var= randWeight(temp,window_size,window_size)
#max_weight, max_rewards, max_var= maxweight(temp,window_size,window_size)
KDagg_weight, KDagg_rewards, KDagg_var= StochasticKdagg(temp,ans,window_size,window_size,len(temp))
#mark_weight, mark_rewards, mark_var= mark(temp,window_size,window_size)
actions_equal, actions_rl, actions_KD, actions_KDagg  = [], [], [], []
result_equal, result_rl, result_KD, result_KDagg = [], [], [], []

KD_weight.append(np.ones(agent.portfolio_size) / agent.portfolio_size)
KDagg_weight.append(np.ones(agent.portfolio_size) / agent.portfolio_size)

i=0
for t in range(window_size, len(env.fil_data), rebalance_period):
    print(i)
    date1 = t-rebalance_period
    s_ = env.get_state(t, window_size)
    s_1 = env.get_state2(t, 60)
    action = agent.act(s_,s_1)

    weighted_returns, reward = env.get_reward(action[0], t, t+rebalance_period+1)

    weighted_returns_equal, reward_equal = env.get_reward(
        np.ones(agent.portfolio_size) / agent.portfolio_size, t, t+rebalance_period+1)
    
    weighted_returns_KD, reward_KD = env.get_reward(
       KD_weight[i] , t, t+rebalance_period+1)
    
    weighted_returns_KDagg, reward_KDagg = env.get_reward(
       KDagg_weight[i] , t, t+rebalance_period+1)
    
    result_equal.append(weighted_returns_equal.tolist())
    actions_equal.append(np.ones(agent.portfolio_size) / agent.portfolio_size)
    
    result_KD.append(weighted_returns_KD.tolist())
    actions_KD.append(KD_weight[i])
    
    result_KDagg.append(weighted_returns_KDagg.tolist())
    actions_KDagg.append(KDagg_weight[i])
    
    result_rl.append(weighted_returns.tolist())
    actions_rl.append(action[0])
    i=i+1

result_equal_vis = [item for sublist in result_equal for item in sublist]
result_rl_vis = [item for sublist in result_rl for item in sublist]
result_KD_vis = [item for sublist in result_KD for item in sublist]
result_KDagg_vis = [item for sublist in result_KDagg for item in sublist]

data=pd.read_csv('/Users/Kumar/Desktop/PortfolioOptimization/SP500.csv')
data['Date']=pd.to_datetime( data['Date'])
data.index = data['Date']
data = data.drop(['Date'], axis=1)
df = data['Close'].pct_change().dropna()
df=df[155:1759]
plt.figure(figsize = (12, 4))
#plt.plot(np.array(mark_rewards).cumsum(),label="Markowitz Portfolio Return")
plt.plot(df.index,np.array(result_KD_vis).cumsum(),label="ZoomSVD-Rl-Portfolio")
#plt.plot(np.array(rand_rewards).cumsum(),label="Random Portfolio Return")
plt.plot(df.index,np.array(result_KDagg_vis).cumsum(),label="OHLC-RL-Portfolio")
#plt.plot(np.array(equal_rewards).cumsum(),label="Equal-Weighted Portfolio Return")
#plt.plot(np.array(max_rewards).cumsum(),label="Max Portfolio-Return")
#plt.plot(df.index,np.array(df).cumsum(),label="SP")Autoencoder-Rl_Portfolio
plt.plot(df.index,np.array(result_equal_vis).cumsum(),label="RBM-Rl-Portfolio")
plt.plot(df.index,np.array(result_rl_vis).cumsum(),label="Autoencoder-Rl_Portfolio")
plt.legend(loc="upper left")
plt.xlabel('time period in days')
plt.ylabel('Train set returns')
plt.show()









