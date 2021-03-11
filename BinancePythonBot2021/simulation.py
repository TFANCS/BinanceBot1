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
    base_balance -= quantity
    quote_balance += quantity*price * 0.999
    return base_balance, quote_balance


def buy(base_balance, quote_balance, quantity, price):
    quote_balance -= quantity*price * 0.999
    base_balance += quantity
    return base_balance, quote_balance







def simulation(binance,model):

    df_list = {}

    for symbol in const.PAIR_SYMBOLS:
        df_list[symbol] = make_dataset.make_current_data(binance,symbol,12,10)
        #mpf.plot(df_list[symbol], type='candle')

    symbol = "BTCUSDT"

    first_base_balance = 0.002
    first_quote_balance = 100
    base_balance = first_base_balance
    quote_balance = first_quote_balance
    price = df_list[symbol].iloc[0,0]
    #price = float(binance.get_ticker("BTCUSDT")["lastPrice"])

    data,_ = make_dataset.make_dataset(df_list[symbol])
    print(df_list[symbol])
    print(data)



    model.load_weights(const.CHECKPOINT_PATH)
    prediction = pd.DataFrame(model.predict(data))
    #prediction = prediction.idxmax(axis=1)
    prediction = prediction.iloc[:,0].tolist()


    
    first_balance = (base_balance*price)+quote_balance


    for i in range(len(prediction)):
        print("Period:" + str(i))
        price = df_list[symbol].iloc[i+const.TIME_LENGTH,0]
        print("Price:"+str(price))
        print("Balance:"+str((base_balance*price)+quote_balance) + " Base:" + str(base_balance) + " Quote:" + str(quote_balance))
        if prediction[i] < -0.05:
            print("SELL")
            base_balance, quote_balance = sell(base_balance, quote_balance,0.0002,price)
        elif prediction[i] > 0.05:
            print("BUY")
            base_balance, quote_balance = buy(base_balance, quote_balance,0.0002,price)
        print("")



    print("Start:" + str(first_balance) + "  Last" + str((base_balance*price)+quote_balance))
    print("Result:" + str((base_balance*price)+quote_balance-first_balance))
    print("Without Trading:" + str((first_base_balance*price + first_quote_balance)-first_balance))

    print()

    x=range(len(prediction))
    plt.plot(x, df_list[symbol].iloc[-len(prediction):,1], prediction)
    plt.legend(["actual", "predict"], loc="upper left")
    plt.show()










def simulation_c(binance,model):

    df_list = {}

    for symbol in const.PAIR_SYMBOLS:
        df_list[symbol] = make_dataset.make_current_data(binance,symbol,520,420)
        #mpf.plot(df_list[symbol], type='candle')

    symbol = "BTCUSDT"

    first_base_balance = 0.0
    first_quote_balance = 100
    base_balance = first_base_balance
    quote_balance = first_quote_balance
    price = df_list[symbol].iloc[0,0]
    #price = float(binance.get_ticker("BTCUSDT")["lastPrice"])

    data, target = make_dataset.make_classification_dataset(df_list[symbol])
    print(df_list[symbol])
    print(data)



    model.load_weights(const.CHECKPOINT_PATH)
    prediction = pd.DataFrame(model.predict(data))
    prediction = prediction.idxmax(axis=1)
    #prediction = prediction.iloc[:,0].tolist()


    
    first_balance = (base_balance*price)+quote_balance
    last_balance = (base_balance*price)+quote_balance
    prev_action = 1

    total_lose = 0
    total_win = 0

    for i in range(len(prediction)):
        print("Period:" + str(i))
        price = df_list[symbol].iloc[i+const.TIME_LENGTH-1,0] #data does not include [i+const.TIME_LENGTH]
        if prev_action == 0:
            base_balance, quote_balance = buy(base_balance, quote_balance,0.001,price)
            print("stop")
        elif prev_action == 2:
            base_balance, quote_balance = sell(base_balance, quote_balance,0.001,price)
        if (base_balance*price)+quote_balance-last_balance > 0:
            total_win += 1
        elif (base_balance*price)+quote_balance-last_balance < 0:
            total_lose += 1
        print("Price:"+str(price))
        print("Balance:"+str((base_balance*price)+quote_balance))
        print("Profit:" + str((base_balance*price)+quote_balance-first_balance))
        last_balance = (base_balance*price)+quote_balance
        if prediction[i] == 0:
            print("SELL")
            base_balance, quote_balance = sell(base_balance, quote_balance,0.001,price)
            prev_action = 0
        elif prediction[i] == 2:
            print("BUY")
            base_balance, quote_balance = buy(base_balance, quote_balance,0.001,price)
            prev_action = 2
        else:
            prev_action = 1
        print("")



    print("Start:" + str(first_balance) + "  Last" + str((base_balance*price)+quote_balance))
    print("Result:" + str((base_balance*price)+quote_balance-first_balance))
    print("Without Trading:" + str((first_base_balance*price + first_quote_balance)-first_balance))
    print("Total Lose:" + str(total_lose))
    print("Total Win:" + str(total_win))

    print()



    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    x=range(len(prediction))
    ax1.plot(x, df_list[symbol].iloc[-len(prediction):,1])
    ax2.plot(x, prediction)
    plt.show()
