from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime
import mplfinance as mpf
import tensorflow as tf
import os
import const



def make_dataset(orig_data):

    data, target = [], []


    for i in range(len(orig_data)-const.TIME_LENGTH):  #get [i]~[i+const.TIME_LENGTH] as data and get [i+const.TIME_LENGTH] as target
        data.append(orig_data.iloc[i:i + const.TIME_LENGTH,:]) #row i to i + const.TIME_LENGTH-1
        dif = orig_data.iloc[i + const.TIME_LENGTH,1] - orig_data.iloc[i,0]  #difference between first open and last close
        #print(str(orig_data.iloc[i,1]) + ":" + str(orig_data.iloc[i,0]))
        target.append(2 if dif > 0.00025 else 0 if dif < -0.00025 else 1)
        #print(2 if dif > 0.0001 else 0 if dif < -0.0001 else 1)

    re_data = np.array(data).reshape(len(data), const.TIME_LENGTH, len(orig_data.columns))
    re_target = np.array(target).reshape(len(data), 1)


    return re_data, re_target


