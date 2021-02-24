from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime
import mplfinance as mpf
import tensorflow as tf
import os
import const
import technical_indicators


def make_dataset(orig_data):

    data, target = [], []

    #threshold = 0.0005
    threshold = 0.0

    for i in range(len(orig_data)-(const.TIME_LENGTH+1)):  #get [i]~[i+const.TIME_LENGTH] as data and get [i+const.TIME_LENGTH] as target
        #data.append(orig_data.iloc[i:i + const.TIME_LENGTH,:]) #row i to i + const.TIME_LENGTH-1
        data.append(orig_data.iloc[i:i + const.TIME_LENGTH,:]) #row i to i + const.TIME_LENGTH-1
        #dif = orig_data.iloc[i + const.TIME_LENGTH,1] - orig_data.iloc[i,0]  #difference between first open and last close
        #dif = orig_data.iloc[i + const.TIME_LENGTH+((const.TIME_LENGTH//10)*7),1] - orig_data.iloc[i + ((const.TIME_LENGTH//10)*7),0]
        #dif = orig_data.iloc[i + const.TIME_LENGTH+(const.TIME_LENGTH//10),1] - orig_data.iloc[i + const.TIME_LENGTH,0]
        #print(str(orig_data.iloc[i,1]) + ":" + str(orig_data.iloc[i,0]))
        #print(dif)
        #threshold = orig_data.iloc[i+const.TIME_LENGTH,1]*0.001
        #target.append(2 if dif > threshold else 0 if dif < -threshold else 1)
        #target.append(orig_data.iloc[i+const.TIME_LENGTH,1])
        target.append(orig_data.iloc[i+const.TIME_LENGTH,len(orig_data.columns)-1])
        #target.append(orig_data.iloc[i+const.TIME_LENGTH,1])

    #down_num = 0
    #up_num = 0
    #for i in target:
    #    if i == 0:
    #        down_num += 1
    #    elif i == 2:
    #        up_num += 1
    #print(down_num/(len(orig_data)-const.TIME_LENGTH))
    #print(up_num/(len(orig_data)-const.TIME_LENGTH))

    #mpf.plot(orig_data, type="candle")

    re_data = np.array(data).reshape(len(data), const.TIME_LENGTH, len(orig_data.columns))
    re_target = np.array(target).reshape(len(data), 1)

    np.nan_to_num(re_data, copy=False)
    np.nan_to_num(re_target, copy=False)

    return re_data, re_target










def make_current_data(binance,symbol, day_start,day_end, normalized = True):
    df = pd.DataFrame(columns=["Time","Open","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount","MA7","MA25","MA99","EMA7","EMA25","EMA99","WMA7","WMA25","WMA99","BOLL_UP","BOLL_DOWN","VWAP","TEMA","SAR","MACD","MACD_SIGNAL","RSI6","RSI12","RSI24","OBV","WR","PL","OrigClose"])
    #df = pd.DataFrame(columns=["Time","Open","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount","MA7","MA25","MA99","EMA7","EMA25","EMA99","WMA7","WMA25","WMA99","BOLL_UP","BOLL_DOWN","VWAP","TEMA","SAR","MACD","MACD_SIGNAL","RSI6","RSI12","RSI24","K","D","J","OBV","WR","DI+","DI-","ADX","MTM","EMV"])
    df.loc[:, "Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time")


    #to match the minimum size
    pad_multiplier = 200//const.TIME_LENGTH

    if day_start == 0 and day_end == 0:
        klines = binance.get_klines(symbol,const.TIME_LENGTH*pad_multiplier)
        df = df.append(klines)
    else:
        for i in range(day_start,day_end,-1):
            klines = binance.get_historical_klines(symbol, str(i+1) + " days ago UTC", str(i) + " days ago UTC");
            df = df.append(klines)

    try:
        df = df.astype("float64")
        #mpf.plot(df, type="candle")

        orig_close = df["Close"]
        if normalized:
            df = tf.keras.utils.normalize(df, axis=0, order=2)
        df["OrigClose"] = orig_close
        df.loc[:, "MA7"] =  technical_indicators.ma(df,7)
        df.loc[:, "MA25"] =  technical_indicators.ma(df,25)
        df.loc[:, "MA99"] =  technical_indicators.ma(df,99)
        df.loc[:, "EMA7"] =  technical_indicators.ema(df,7)
        df.loc[:, "EMA25"] =  technical_indicators.ema(df,25)
        df.loc[:, "EMA99"] =  technical_indicators.ema(df,99)
        df.loc[:, "WMA7"] =  technical_indicators.wma(df,7)
        df.loc[:, "WMA25"] =  technical_indicators.wma(df,25)
        df.loc[:, "WMA99"] =  technical_indicators.wma(df,99)
        df.loc[:, "BOLL_UP"],df.loc[:, "BOLL_DOWN"] =  technical_indicators.boll(df,21)
        df.loc[:, "VWAP"] =  technical_indicators.vwap(df,14)
        df.loc[:, "TEMA"] =  technical_indicators.tema(df,9)
        df.loc[:, "SAR"] =  technical_indicators.sar(df)
        df.loc[:, "MACD"],df.loc[:, "MACD_SIGNAL"] =  technical_indicators.macd(df)
        df.loc[:, "RSI6"] =  technical_indicators.rsi(df,6)
        df.loc[:, "RSI12"] =  technical_indicators.rsi(df,12)
        df.loc[:, "RSI24"] =  technical_indicators.rsi(df,24)
        #df.loc[:, "K"],df.loc[:, "D"],df.loc[:, "J"] =  technical_indicators.kdj(df)
        df.loc[:, "OBV"] =  technical_indicators.obv(df)
        df.loc[:, "WR"] =  technical_indicators.wr(df,14)
        #df.loc[:, "DI+"],df.loc[:, "DI-"],df.loc[:, "ADX"] =  technical_indicators.dmi(df)
        #df.loc[:, "MTM"] =  technical_indicators.mtm(df)
        #df.loc[:, "EMV"] =  technical_indicators.emv(df)
        df.loc[:, "PL"] =  technical_indicators.pl(df,12)
        #df = df.append(klines)
    except Exception as e:
        print('Exception : {}'.format(e))
        df = None

    return df



