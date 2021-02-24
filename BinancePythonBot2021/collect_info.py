from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
import mplfinance as mpf
import os
import const
import technical_indicators
import tensorflow as tf
import make_dataset


def collect_info(binance):

    df_list = {}
    df_list_test = {}

    print("collect data")

    for symbol in const.PAIR_SYMBOLS:
        if os.path.exists("..\\Data\\" + symbol + "_data.csv"):
            df_list[symbol] = pd.read_csv("..\\Data\\" + symbol + "_data.csv", index_col=0, parse_dates=True)
            mpf.plot(df_list[symbol], type='candle')
        else:
            df_list[symbol] = make_dataset.make_current_data(binance,symbol,120,10,normalized = True)
            df_list[symbol].to_csv("..\\Data\\" + symbol + "_data.csv")
            
 


    for symbol in const.PAIR_SYMBOLS:
        base_asset = binance.get_base_asset(symbol)
        quote_asset = binance.get_quote_asset(symbol)


    for i in const.BALANCE_SYMBOLS:
        print(i + " Available : " + binance.get_balance(i)["free"])




    