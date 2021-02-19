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
import matplotlib.pyplot as plt





def sell(base_balance, quote_balance, quantity, price):
    if base_balance >= quantity:
        base_balance -= quantity
        quote_balance += quantity*price*0.999
    else:
        print("Insufufficient balance")
    return base_balance, quote_balance


def buy(base_balance, quote_balance, quantity, price):
    if quote_balance >= quantity*price:
        quote_balance -= quantity*price
        base_balance += quantity*0.999
    else:
        print("Insufufficient balance")
    return base_balance, quote_balance







def simulation(binance,model):

    df_list_normalized = {}
    df_list = {}

    for symbol in const.PAIR_SYMBOLS:
        df_list_normalized[symbol] = make_dataset.make_current_data(binance,symbol,10,0)
        df_list[symbol] = make_dataset.make_current_data(binance,symbol,10,0, normalized=False)
        #mpf.plot(df_list[symbol], type='candle')

    symbol = "BTCUSDT"

    first_base_balance = 0.002
    first_quote_balance = 100
    base_balance = first_base_balance
    quote_balance = first_quote_balance
    price = df_list[symbol].iloc[0,1]
    #price = float(binance.get_ticker("BTCUSDT")["lastPrice"])

    data,_ = make_dataset.make_dataset(df_list_normalized[symbol])
    print(df_list[symbol])
    print(data)

    model.load_weights(const.CHECKPOINT_PATH.format(time_length=const.TIME_LENGTH))
    df = pd.DataFrame(model.predict(data))
    df = df.idxmax(axis=1)

    print(df)

    first_balance = (base_balance*price)+quote_balance

    for i in range(len(df)):
        print("Period:" + str(i))
        price = df_list[symbol].iloc[i,1]
        print("Price:"+str(price))
        print("Balance:"+str((base_balance*price)+quote_balance) + " Base:" + str(base_balance) + " Quote:" + str(quote_balance))
        if df.iloc[i] == 0:
            print("SELL")
            base_balance, quote_balance = sell(base_balance, quote_balance,0.0002,price)
        elif df.iloc[i] == 2:
            print("BUY")
            base_balance, quote_balance = buy(base_balance, quote_balance,0.0002,price)
        print("")



    print("Start:" + str(first_balance) + "  Last" + str((base_balance*price)+quote_balance))
    print("Result:" + str((base_balance*price)+quote_balance-first_balance))
    print("Without Trading:" + str((first_base_balance*price + first_quote_balance)-first_balance))

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    x=range(len(df))
    ax1.plot(x,df_list[symbol].iloc[:len(df),1])
    ax2.plot(x,df)
    plt.show()


