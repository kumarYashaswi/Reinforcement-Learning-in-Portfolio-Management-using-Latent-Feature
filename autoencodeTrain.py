# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:35:51 2020

@author: Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:55:33 2020

@author: Kumar
"""
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
from Kalman import *



def mkdate(ts):
    return datetime.datetime.fromtimestamp(
        int(ts)
    ).strftime('%Y-%m-%d')

def plot_examples(stock_input, stock_decoded):
    n = 10  
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, test_samples, 200))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)
        

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)
        
        
def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")
    
(self, prices = '/Users/Kumar/Desktop/PortfolioOptimization/final_data.csv', 
                 prices1 = '/Users/Kumar/Desktop/PortfolioOptimization/val_data.csv',capital = 1e6):       
        self.prices = prices
        self.prices1 = prices1
df1 = pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/final_data.csv')
df2 = pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/RLPortfolio/PortfolioOptimization/val_data.csv')

del df1['Unnamed: 0']

df1.index =df1['Date']
df1 = df1.drop(['Date'], axis=1)
df = df1.pct_change().dropna()
df = klfilter(df)
df=df*100

def autoencodetrain(df1):
            j=1
            window_length = 60

            epochs = 100
            for i,col in enumerate(df1.columns):
                    x_train_nonscaled = np.array([df1[col].values[i-window_length:i].reshape(-1, 1) for i in tqdm(range(window_length+1,len(df1)))])
                    if(j==1):
                        x_train= x_train_nonscaled
                        j=0
                    else:
                        x_train = np.concatenate([x_train,x_train_nonscaled], axis=0)
            
            #from sklearn.model_selection  import train_test_split
            #x_train, x_test = train_test_split(x_train, test_size = 0.2, random_state = 0)
            
            
            x_train = x_train.astype('float32')
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
            autoencoder = Model(input_window, decoded)
            autoencoder.summary()
            encoder_conv.load_weights(filepath="modelCNN.h5")

            '''
            autoencoder.compile(optimizer='adam', loss='mse')
            history = autoencoder.fit(x_train, x_train,
                            epochs=epochs,
                            batch_size=512,
                            shuffle=True)
            '''
            return encoder_conv
'''
# this is our input placeholder
input_window = Input(shape=(window_length,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_window)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(window_length, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_window, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_window, encoded)


autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train_simple, x_train_simple,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test_simple, x_test_simple))

decoded_stocks = autoencoder.predict(x_test_simple)



plot_history(history)
plot_examples(x_test_simple, decoded_stocks)


x_train_deep = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_deep = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

input_window = Input(shape=(window_length,))

x = Dense(6, activation='relu')(input_window)
x = BatchNormalization()(x)
encoded = Dense(encoding_dim, activation='relu')(x)
# "decoded" is the lossy reconstruction of the input

x = Dense(6, activation='relu')(encoded)
x = BatchNormalization()(x)
decoded = Dense(window_length, activation='sigmoid')(x)

# this model maps an input to its reconstruction
autoencoder = Model(input_window, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_window, encoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train_deep, x_train_deep,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test_deep, x_test_deep))

decoded_stocks = autoencoder.predict(x_test_deep)

plot_history(history)

input_window = Input(shape=(window_length,1))
x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
#x = BatchNormalization()(x)
x = MaxPooling1D(2, padding="same")(x) # 5 dims
x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
#x = BatchNormalization()(x)
encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

encoder = Model(input_window, encoded)

# 3 dimensions in the encoded layer

x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 6 dims
x = Conv1D(16, 2, activation='relu')(x) # 5 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 10 dims
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
autoencoder = Model(input_window, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_stocks = autoencoder.predict(x_test)
'''