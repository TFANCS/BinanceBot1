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





def spot_trade(binance,model):

    os.system("cls") #Clear Screen
    dt_now = datetime.datetime.now()
    print(dt_now.strftime("%Y-%m-%d %H:%M:%S"))

    symbol = "BTCUSDT"
    base_asset = binance.get_base_asset(symbol)
    quote_asset = binance.get_quote_asset(symbol)

    amount = 0.0005

    base_balance = binance.get_free_balance(base_asset)
    quote_balance = binance.get_free_balance(quote_asset)
    price = binance.get_current_price(symbol)

    print(base_asset + " Available : " + binance.get_free_balance(base_asset))
    print(quote_asset + " Available : " + binance.get_free_balance(quote_asset))

    model.load_weights(const.CHECKPOINT_PATH.format(time_length=const.TIME_LENGTH))

    ready = False

    df_test = make_dataset.make_current_data(binance,symbol,0,0)
    test_data, test_target = make_dataset.make_dataset(df_test)
    test_loss, test_acc = model.evaluate(test_data,  test_target, verbose=2)
    print("Test accuracy:", test_acc)

    while True:

        loop = True
        while loop:
            sleep(5)
            dt_now = datetime.datetime.now()
            if dt_now.minute%5 == 4 and dt_now.second >= 47  and dt_now.second <= 57 and ready == True:
                print(dt_now.strftime("%Y-%m-%d %H:%M:%S"))
                loop = False
                ready = False
            elif dt_now.minute%5 == 0 and ready == False:
                base_balance = binance.get_free_balance(base_asset)
                quote_balance = binance.get_free_balance(quote_asset)
                print("Balance:"+str((base_balance*price)+quote_balance) + " Base:" + str(base_balance) + " Quote:" + str(quote_balance))
                ready = True


        df = make_dataset.make_current_data(binance,symbol,0,0)
        df = df.iloc[len(df)-const.TIME_LENGTH:,:]
        input_data = np.array(df).reshape(1, const.TIME_LENGTH, len(df.columns))

        predicted = model.predict(input_data)[-1]
        result = np.argmax(predicted)

        price = binance.get_current_price(symbol)
        print("Price:"+str(price))
        if result == 0:
            print("SELL")
            binance.create_market_order(symbol, 0.0002, "SELL")
        elif result == 2:
            print("BUY")
            binance.create_test_order(symbol, 0.0002, "BUY")
        print("")





