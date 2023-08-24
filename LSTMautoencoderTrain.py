# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:35:05 2020

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
    


def LSTMautoencodetrain(df1):
            j=1
            window_length = 60
            encoding_dim=30
            epochs = 125
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
            
            
            #x_train_simple = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
            #x_test_simple = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))




            inputs = Input(shape=(60, 1))
            encoded_lstm = LSTM(encoding_dim)(inputs)
            
            decoded_lstm = RepeatVector(60)(encoded_lstm)
            decoded_lstm = LSTM(1, return_sequences=True)(decoded_lstm)
            
            sequence_autoencoder = Model(inputs, decoded_lstm)
            encoder_lstm = Model(inputs, encoded_lstm)
            sequence_autoencoder.summary()
            
            sequence_autoencoder.compile(optimizer='adam', loss='mse')
            '''
            history = sequence_autoencoder.fit(x_train, x_train,
                            epochs=150,
                            batch_size=512,
                            shuffle=True)
            '''
            encoder_lstm.load_weights(filepath="modelLSTM.h5")

            return encoder_lstm