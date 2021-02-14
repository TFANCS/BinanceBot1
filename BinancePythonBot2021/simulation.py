from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime
import mplfinance as mpf
import tensorflow as tf
import os
import const
import make_dataset






def sell(base_balance, quote_balance, quantity, price):
    base_balance -= quantity
    quote_balance += quantity*price
    return base_balance, quote_balance


def buy(base_balance, quote_balance, quantity, price):
    base_balance += quantity
    quote_balance -= quantity*price
    return base_balance, quote_balance



def simulation(binance,model):

    df_list_test = {}
    df_list_orig = {}

    for symbol in const.PAIR_SYMBOLS:
        df_list_test[symbol] = pd.read_csv("..\\Data\\" + symbol + "_data_test.csv", index_col=0, parse_dates=True)
        df_list_orig[symbol] = pd.read_csv("..\\Data\\" + symbol + "_data_test.csv", index_col=0, parse_dates=True)
        #mpf.plot(df_list[symbol], type='candle')
        df_list_test[symbol] = tf.keras.utils.normalize(df_list_test[symbol], axis=0, order=2)

    symbol = "BTCUSDT"

    base_balance = 0.002
    quote_balance = 100
    price = df_list_orig[symbol].iloc[0,1]
    #price = float(binance.get_ticker("BTCUSDT")["lastPrice"])

    data, _ = make_dataset.make_dataset(df_list_test[symbol])


    model.load_weights(const.CHECKPOINT_PATH)
    df = pd.DataFrame(model.predict(data))
    df = df.idxmax(axis=1)

    first_balance = (base_balance*price)+quote_balance

    for i in range(300):
        print("Period:" + str(i))
        price = df_list_orig[symbol].iloc[i,1]
        print("Price:"+str(price))
        print("Balance:"+str((base_balance*price)+quote_balance))
        if df.iloc[i] == 0:
            base_balance, quote_balance = sell(base_balance, quote_balance,0.0002,price)
        elif df.iloc[i] == 2:
            base_balance, quote_balance = buy(base_balance, quote_balance,0.0002,price)
        print("")



    print("Start:" + str(first_balance) + "  Last" + str((base_balance*price)+quote_balance))
    print("result:" + str((base_balance*price)+quote_balance-first_balance))





