from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from time import sleep
import datetime
import tensorflow as tf
import const
import os
import make_dataset
import technical_indicators



def practical_trade(binance):


    symbol = "BTCUSDT"
    base_asset = binance.get_base_asset(symbol)
    quote_asset = binance.get_quote_asset(symbol)

    amount = 0.001

    current_balance = binance.get_futures_balance(quote_asset)
    price = binance.get_current_price(symbol)

    wait_time = 0
    prev_action = 0
    stop_time = 0

    prev_price = 0

    trade_history = []

    touched_boll = 0

    while True:

        wait_time -= 1

        df = make_dataset.make_current_data(binance,symbol,0,0,normalized = False)
        if df is None:
            if prev_action == 1:
                trade_history.append("BUY  " + dt_formed + "  Balance:" + str(binance.get_futures_balance(quote_asset)))
                binance.create_futures_order(symbol, amount, "BUY")
                stop_time = 0
                prev_action = 0
            if prev_action == 2:
                trade_history.append("SELL " + dt_formed + "  Balance:" + str(binance.get_futures_balance(quote_asset)))
                binance.create_futures_order(symbol, amount, "SELL")
                stop_time = 0
                prev_action = 0
            continue
        df = df.iloc[-1,:]

        os.system("cls") #Clear Screen
        dt_now = datetime.datetime.now()
        dt_formed = dt_now.strftime("%Y-%m-%d %H:%M:%S")
        print(dt_formed)
        
        current_balance = binance.get_futures_balance(quote_asset)
        price = binance.get_current_price(symbol)
        if price is None:
            if prev_action == 1:
                trade_history.append("BUY  " + dt_formed + "  Balance:" + str(current_balance))
                binance.create_futures_order(symbol, amount, "BUY")
                stop_time = 0
                prev_action = 0
            if prev_action == 2:
                trade_history.append("SELL " + dt_formed + "  Balance:" + str(current_balance))
                binance.create_futures_order(symbol, amount, "SELL")
                stop_time = 0
                prev_action = 0
            continue
        print("Price:"+str(price))
        print(quote_asset + " Available : " + str(current_balance))

        dif = df["Close"] - df["Open"] 

        print("Dif:" + str(dif) + " WaitTime:" + str(wait_time) + " StopTime:" + str(stop_time))




        if wait_time < 0 and prev_action == 0:
            if df["BOLL_UP"] < price:
                touched_boll = 1
            elif df["BOLL_DOWN"] > price:
                touched_boll = 2
            if touched_boll == 1 < price and df["SAR"] > price:
                trade_history.append("SELL " + dt_formed + "  Balance:" + str(current_balance))
                wait_time = 100
                prev_action = 1
                prev_price = price
                binance.create_futures_order(symbol, amount, "SELL")
                touched_boll = 0
            elif touched_boll == 2 > price and df["SAR"] < price:
                trade_history.append("BUY  " + dt_formed + "  Balance:" + str(current_balance))
                wait_time = 100
                prev_action = 2
                prev_price = price
                binance.create_futures_order(symbol, amount, "BUY")
                touched_boll = 0
            


        #close position after some time
        #if wait_time <= 0:
        #    if prev_action == 1 and stop_time >= 3:
        #        trade_history.append("BUY  " + dt_formed + "  Balance:" + str(current_balance))
        #        binance.create_futures_order(symbol, amount, "BUY")
        #        prev_action = 0
        #    if prev_action == 2 and stop_time >= 3:
        #        trade_history.append("SELL")
        #        binance.create_futures_order(symbol, amount, "SELL")
        #        prev_action = 0



        #close position if there is loss
        if (prev_action == 1 and dif > 0) or (prev_action == 2 and dif < 0):
            stop_time += 1
        else:
            stop_time = 0
        
        if stop_time >= 18:
            if prev_action == 1:
                trade_history.append("BUY  " + dt_formed + "  Balance:" + str(current_balance))
                binance.create_futures_order(symbol, amount, "BUY")
            if prev_action == 2:
                trade_history.append("SELL")
                binance.create_futures_order(symbol, amount, "SELL")
            stop_time = 0
            prev_action = 0




        if prev_action == 1 and df["SAR"] < price:
            trade_history.append("BUY  " + dt_formed + "  Balance:" + str(current_balance))
            binance.create_futures_order(symbol, amount, "BUY")
            stop_time = 0
            prev_action = 0
        if prev_action == 2 and df["SAR"] > price:
            trade_history.append("SELL " + dt_formed + "  Balance:" + str(current_balance))
            binance.create_futures_order(symbol, amount, "SELL")
            stop_time = 0
            prev_action = 0




        #stop_loss
        if prev_action == 1 and (price - prev_price) > price/700:
            trade_history.append("BUY  " + dt_formed + "  Balance:" + str(current_balance))
            binance.create_futures_order(symbol, amount, "BUY")
            stop_time = 0
            prev_action = 0
        if prev_action == 2 and (price - prev_price) < -price/700:
            trade_history.append("SELL " + dt_formed + "  Balance:" + str(current_balance))
            binance.create_futures_order(symbol, amount, "SELL")
            stop_time = 0
            prev_action = 0






        print("\nHistory:")
        for i in trade_history:
            print(i)


        sleep(3) #it takes 2 second to calculate






