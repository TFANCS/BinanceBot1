from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from time import sleep
import tensorflow as tf
import const
import os
import make_dataset


def test(binance):
    df = pd.DataFrame(columns=["Open","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount"])
    klines = binance.get_historical_klines("BTCUSDT", "1 days ago UTC", "now UTC");
    df = df.append(klines)
    df = df.applymap(lambda x: float(x))
    print(df)
    mpf.plot(df, type='candle')

    indicator = ma(df,7)

    x = range(len(df)-7)
    plt.plot(x, indicator)
    plt.show()



def ma(df,length):
    output = []
    for i in range(len(df)-length):
        average = sum(df.iloc[i:i+length,1]) / length
        output.append(average)
    return output





def ema(df,length):
    output = []
    first = sum(df.iloc[0:length,1]) / length
    for i in range(len(df)-length):
        value = first + (2/(length+1))*(df.iloc[i,1]-first)
        first = value
        output.append(value)
    return output




def wma(df,length):
    output = []
    for i in range(len(df)-length):
        sum = 0
        for j in range(length):
            sum += df.iloc[i+j,1]*(length-j)
        average = sum / length
        output.append(average)
    return output




def boll(df,length):
    std = np.std(df.iloc[i:i+length,1])
    ma = ma(df,length)
    upper = ma + (2 * std)
    lower = ma - (2 * std)
    return upper, lower



def vwap(df,length):
    output = []
    for i in range(len(df)-length):
        sum = 0
        for j in range(length):
            typical_price = df.iloc[i+j,1] + df.iloc[i+j,2] + df.iloc[i+j,3] / 3   #Typical Price = High + Low + Close / 3
            sum += typical_price * df.iloc[i+j,4]   #VWAP = ∑ (Typical Price * Volume ) / ∑ Volume
        value = sum / sum(df.iloc[i:i+length,4])
        output.append(value)
    return output



def ema_array(arr,length):
    output = []
    first = sum(arr[0:length]) / length
    for i in range(len(arr)-length):
        value = first + (2/(length+1))*(arr[i]-first)
        first = value
        output.append(value)
    return output
def tema(df,length):
    ema1 = ema(df,length)
    ema2 = ema_array(ema1,length)
    ema3 = ema_array(ema2,length)
    tema = (3*np.array(ema1))-(3*np.array(ema2))+ema3
    return tema




def sar(df,length):
    sar = 0
    af = 0.02
    ep_min = float("inf")
    ep_max = 0
    for i in range(length):
        if sar > df.iloc[i,1]:  #down trend
            if df.iloc[i,3]<ep_min:
                ep_min = df.iloc[i,3]
            sar = sar + af * (ep_min-sar)
        else :         #up trend
            if df.iloc[i,2]>ep_max:
                ep_max = df.iloc[i,2]
            sar = sar + af * (ep_max-sar)
        output.append(sar)
        
    return sar




def macd(df):
    short_period_ema = ema(df,12)
    long_period_ema = ema(df,26)
    macd = short_period_ema - long_period_ema
    signal = ema_array(df,9)
    return output,signal


def rsi(df,length):
    output = []
    gain_sum = 0
    loss_sum = 0
    for i in range(length):    #first rsi value
        dif = df.iloc[i+1,1]-df.iloc[i,1]
        if dif > 0:
            gain_sum += dif
        else:
            loss_sum += dif
    avg_gain = gain_sum / length
    avg_loss = loss_sum / length
    rsi = 100-(100/(1+(avg_gain/avg_loss)))
    output.append(rsi)
    for i in range(length,len(df)):
        dif = df.iloc[i+1,1]-df.iloc[i,1]
        if dif > 0:
            avg_gain = ((avg_gain*(length-1)) + dif)/length
        else:
            avg_loss = ((avg_loss*(length-1)) + dif)/length
        rsi = 100-(100/(1+(avg_gain/avg_loss)))
        output.append(rsi)
    return output




def kdj(df):
    length = 9
    k = []
    d = []
    j = []
    k1 = 50
    d1 = 50
    for i in range(len(df)-length):
        lowest = float("inf")
        highest = 0
        for j in range(length):
            if df.iloc[j,3] < lowest:
                lowest = df.iloc[j,3]
            if df.iloc[j,2] > highest:
                highest = df.iloc[j,2]
        k.append((df.iloc[i+length,1]-lowest)/(highest-lowest)*100)
        if k >= 3:
            d.append((k[-1] + k[-2] + k[-3])/3)
        else:
            d.append(0)
        j.append((3*k[-1])-(2*d[-1]))
    return k,d,j





def obv(df):
    output = []
    obv = 0
    for i in range(len(df)):
        dif = df.iloc[i+1,1]-df.iloc[i,1]
        if dif > 0:
            obv + df.iloc[i,4]
        else:
            obv - df.iloc[i,4]
        output.append(obv)
    return output




def cci(df,length):
    tp = [0]*(length-1)
    ma = [0]*(length-1)
    cci = []
    for i in range(len(df)-length):
        tp.append((df.iloc[i,1] + df.iloc[i,2] + df.iloc[i,3])/3)
        ma.append((df.iloc[-length-1:-1,1])/length)
        sum = 0
        for j in range(length):
            sum += tp[-j-1]-ma[-j-1]
        md = sum/length
        cci.append((tp[-1]-ma[-1])/(md+0.015))
    return cci




def stochRSI(df):
    output = []
    return output





def wr(df,length):
    lowest = float("inf")
    highest = 0
    r = []
    for j in range(length):
        if df.iloc[j,3] < lowest:
            lowest = df.iloc[j,3]
        if df.iloc[j,2] > highest:
            highest = df.iloc[j,2]
        r.append((highest-df.iloc[j,1])/(highest-lowest)*100)







def dmi(df):
    length = 14
    dm_plus = []
    dm_minus = []
    di_plus = []
    di_minus = []
    tr = []
    dx = []
    for i in range(len(df)-1):
        dm_plus_temp = df.iloc[i+1,2] - df.iloc[i,2]
        dm_minus_temp = df.iloc[i,3] - df.iloc[i+1,3]
        if dm_plus_temp < 0:
            dm_plus_temp = 0
        if dm_minus_temp < 0:
            dm_minus_temp = 0
        if dm_plus_temp > dm_minus_temp:
            dm_minus_temp = 0
        if dm_plus_temp < dm_minus_temp:
            dm_plus_temp = 0
        dm_plus.append(dm_plus_temp)
        dm_minus.append(dm_minus_temp)
        tr.append(max([df.iloc[i+1,2]-df.iloc[i+1,3], df.iloc[i+1,2]-df.iloc[i,1], df.iloc[i,1]-df.iloc[i+1,3]]))
        if i>= length:
            di_plus.append((sum(dm_plus[-length-1:-1])/length) / (sum(tr[-length-1:-1])/length) * 100)
            di_minus.append((sum(dm_minus[-length-1:-1])/length) / (sum(tr[-length-1:-1])/length) * 100)
            dx.append((di_plus-di_minus)/(di_plus+di_minus))
    adx = ema_array(dx)
    return di_plus,di_minus,adx




def mtm(df):
    length = 14
    ma = ma(df,length)
    mtm = []
    for i in range(len(df)-1):
        mtm.append(ma[i+1]-ma[i])
    return mtm



def emv(df):
    length = 14
    emv = []
    for i in range(len(df)-1):
        distance_moved = ((df.iloc[i+1,2]+df.iloc[i+1,3])/2)-((df.iloc[i,2]+df.iloc[i,3])/2)
        box_ratio = (df.iloc[i+1,4]/100000)/(df.iloc[i+1,2]+df.iloc[i+1,3])
        emv.append(distance_moved/box_ratio)


