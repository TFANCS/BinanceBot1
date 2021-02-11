from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
import mplfinance as mpf
import os
import const


def collect_info(binance):

    df_list = {}

    for symbol in const.PAIR_SYMBOLS:
        if os.path.exists("..\\Data\\" + symbol + "_data.csv"):
            df_list[symbol] = pd.read_csv("..\\Data\\" + symbol + "_data.csv", index_col=0, parse_dates=True)
            mpf.plot(df_list[symbol], type='candle')
        else:
            df_list[symbol] = pd.DataFrame(columns=["Time","Open","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount"])
            df_list[symbol].loc[:, "Time"] = pd.to_datetime(df_list[symbol]["Time"])
            df_list[symbol] = df_list[symbol].set_index("Time")
            

 

    for symbol in const.PAIR_SYMBOLS:

        for i in range(30,10,-1):
            klines = binance.get_historical_klines(symbol, str(i+1) + " days ago UTC", str(i) + " days ago UTC");
            df_list[symbol] = df_list[symbol].append(klines)
        #klines = binance.get_klines(symbol);
        #df_list[symbol] = df_list[symbol].append(klines)
        df_list[symbol].to_csv("..\\Data\\" + symbol + "_data.csv")
        
        base_asset = binance.get_base_asset(symbol)
        quote_asset = binance.get_quote_asset(symbol)

    for i in const.BALANCE_SYMBOLS:
        print(i + " Available : " + binance.get_balance(i)["free"])




    