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
    klines = binance.get_klines("BTCUSDT",200);
    df = df.append(klines)
    df = df.applymap(lambda x: float(x))
    print(df)
    mpf.plot(df, type='candle')

    indicator = sar(df)
    #a = []
    #for i in range(len(df)):
    #    a.append(df.iloc[i,1])

    x = range(len(df))
    plt.plot(x, indicator)
    plt.show()



def pad_df(df,length):
    pad = pd.DataFrame([df.iloc[0,:]]*(length-1), columns=df.columns)
    df = pad.append(df, ignore_index=True)
    return df


def ma(df,length):
    output = []
    df = pad_df(df,length)
    for i in range(len(df)-(length-1)):
        average = sum(df.iloc[i:i+length,1]) / length
        output.append(average)
    return output



def ema(df,length):
    first = sum(df.iloc[0:length,1]) / length
    output = [first]*(length-2)
    output.append(first)
    for i in range(len(df)-(length-1)):
        value = first + (2/(length+1))*(df.iloc[i+length-1,1]-first)
        first = value
        output.append(value)
    return output




def wma(df,length):
    output = []
    df = pad_df(df,length)
    for i in range(len(df)-(length-1)):
        sum = 0
        weight_sum = 0
        for j in range(length):
            sum += df.iloc[i+j,1]*(length-j)
            weight_sum += length-j
        average = sum / weight_sum
        output.append(average)
    return output




def boll(df,length):
    std = []
    ma = []
    df = pad_df(df,length)
    for i in range(len(df)-(length-1)):
        std.append(np.std(df.iloc[i:i+length,1]))
        ma.append(sum(df.iloc[i:i+length,1])/length)
    upper = np.array(ma) + (2 * np.array(std))
    lower = np.array(ma) - (2 * np.array(std))
    return upper, lower



def vwap(df,length):
    output = []
    df = pad_df(df,length)
    for i in range(len(df)-(length-1)):
        sum_val = 0
        for j in range(length):
            typical_price = (df.iloc[i+j,1] + df.iloc[i+j,2] + df.iloc[i+j,3]) / 3   #Typical Price = High + Low + Close / 3
            sum_val += typical_price * df.iloc[i+j,4]   #VWAP = ∑ (Typical Price * Volume ) / ∑ Volume
        value = sum_val / sum(df.iloc[i:i+length,4])
        output.append(value)
    return output





def ema_array(arr,length):
    first = sum(arr[0:length]) / length
    output = [first]*(length-2)
    output.append(first)
    for i in range(len(arr)-(length-1)):
        value = first + (2/(length+1))*(arr[i+length-1]-first)
        first = value
        output.append(value)
    return output
def tema(df,length):
    ema1 = ema(df,length*6)
    ema2 = ema_array(ema1,length*3)
    ema3 = ema_array(ema2,length)
    tema = (3*np.array(ema1))-(3*np.array(ema2))+np.array(ema3)
    return tema







def sar(df):
    output = []
    trend = 0
    sar = 0
    af = 0.02
    ep_min = float("inf")
    ep_max = 0
    for i in range(len(df)):
        if trend == 0:
            sar = 0
            if df.iloc[i,2] > df.iloc[i-1,2] and df.iloc[i,3] > df.iloc[i-1,3]:
                trend = 1
                sar = df.iloc[i,3]
            elif df.iloc[i,2] < df.iloc[i-1,2] and df.iloc[i,3] < df.iloc[i-1,3]:
                trend = -1
                sar = df.iloc[i,2]
        elif trend == -1:  #down trend
            if df.iloc[i,3]<ep_min:
                ep_min = df.iloc[i,3]
                if af < 0.2:
                    af += 0.02
            sar = sar + af * (ep_min-sar)
            if i+1 < len(df) and df.iloc[i+1,2] > sar:
                trend = 1
                af = 0.02
                ep_max = df.iloc[i+1,2]
                sar = ep_min
        elif trend == 1 :  #up trend
            if df.iloc[i,2]>ep_max:
                ep_max = df.iloc[i,2]
                if af < 0.2:
                    af += 0.02
            sar = sar + af * (ep_max-sar)
            if i+1 < len(df) and df.iloc[i+1,3] < sar:
                trend = -1
                af = 0.02
                ep_min = df.iloc[i+1,3]
                sar = ep_max
        output.append(sar)
        
    return output



def ma_array(arr,length):
    output = []
    pad = [arr[0]]*(length-1)
    arr = pad + arr
    for i in range(len(arr)-(length-1)):
        average = sum(arr[i:i+length]) / length
        output.append(average)
    return output
def macd(df):
    short_period_ema = ema(df,12)
    long_period_ema = ema(df,26)
    macd = np.array(short_period_ema) - np.array(long_period_ema)
    macd = (macd*1000).tolist()
    signal = ma_array(macd,9)
    return macd,signal





def rsi(df,length):
    gain_sum = 0
    loss_sum = 0
    for i in range(length):    #first rsi value
        dif = df.iloc[i,1]-df.iloc[i,0]
        if dif > 0:
            gain_sum += dif
        else:
            loss_sum -= dif
    avg_gain = gain_sum / length
    avg_loss = loss_sum / length
    rsi = 100-(100/(1+(avg_gain/avg_loss)))
    output = [rsi/1000] * (length-1)
    output.append(rsi)
    for i in range(length,len(df)):
        dif = df.iloc[i,1]-df.iloc[i,0]
        if dif > 0:
            avg_gain = ((avg_gain*(length-1)) + dif)/length
        else:
            avg_loss = ((avg_loss*(length-1)) - dif)/length
        rsi = 100-(100/(1+(avg_gain/avg_loss)))
        output.append(rsi/1000)
    return output




def kdj(df):
    length = 3
    k = []
    d = []
    j = []
    for i in range(len(df)-length):
        lowest = float("inf")
        highest = 0
        for l in range(i+length):
            if df.iloc[l,3] < lowest:
                lowest = df.iloc[l,3]
            if df.iloc[l,2] > highest:
                highest = df.iloc[l,2]
        k.append((df.iloc[i+length-1,1]-lowest)/(highest-lowest)*100)
        if len(k) >= 3:
            d.append((k[-1] + k[-2] + k[-3])/3)
        else:
            d.append(k[-1])
        j.append((3*k[-1])-(2*d[-1]))
    k = ([k[0]] * length) + k
    d = ([d[0]] * length) + d
    j = ([j[0]] * length) + j
    return k,d,j





def obv(df):
    output = []
    obv = 0
    for i in range(len(df)-1):
        dif = df.iloc[i+1,1]-df.iloc[i,1]
        if dif > 0:
            obv += df.iloc[i,4]
        else:
            obv -= df.iloc[i,4]
        output.append(obv)
    pad = [output[0]]
    output = pad + output
    return output



"""
def cci(df,length):
    tp = [0]*(length-1)
    ma = [0]*(length-1)
    cci = []
    for i in range(len(df)-length):
        tp.append((df.iloc[i+length,1] + df.iloc[i+length,2] + df.iloc[i+length,3])/3)
        ma.append(sum(df.iloc[i-length+1:i+1,1])/length)
        sum_val = 0
        for j in range(length):
            sum_val += tp[-j-1]-ma[-j-1]
        md = sum_val/length
        cci.append((tp[-1]-ma[-1])/(md*0.015))
    pad = [cci[0]] * length
    cci = pad + cci
    return cci
"""


"""
def stochRSI(df):
    output,_,_ = kdj(df)

    return output
"""




def wr(df,length):
    lowest = float("inf")
    highest = 0
    r = []
    for j in range(len(df)):
        if df.iloc[j,3] < lowest:
            lowest = df.iloc[j,3]
        if df.iloc[j,2] > highest:
            highest = df.iloc[j,2]
        r.append((highest-df.iloc[j,1])/(highest-lowest)/10)
    return r







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
        tr.append(max([df.iloc[i+1,2]-df.iloc[i+1,3], df.iloc[i+1,2]-df.iloc[i,1], df.iloc[i+1,3]-df.iloc[i,1]]))
        if i>= length:
            di_plus.append(sum(dm_plus[-length-1:-1]) / sum(tr[-length-1:-1]) * 100)
            di_minus.append(sum(dm_minus[-length-1:-1]) / sum(tr[-length-1:-1]) * 100)
            dx.append((di_plus[-1]-di_minus[-1])/(di_plus[-1]+di_minus[-1]) * 100)
    adx = ma_array(dx,length)
    di_plus = ([di_plus[0]]*(length+1))+di_plus
    di_minus = ([di_minus[0]]*(length+1))+di_minus
    adx = ([adx[0]]*(length+1))+adx
    return di_plus,di_minus,adx




def mtm(df):
    length = 14
    ma_val = ma(df,length)
    mtm = []
    for i in range(len(df)-1):
        mtm.append(ma_val[i+1]-ma_val[i])
    mtm = [mtm[0]] + mtm
    return mtm



def emv(df):
    length = 14
    emv = []
    for i in range(len(df)-1):
        distance_moved = ((df.iloc[i+1,2]+df.iloc[i+1,3])/2)-((df.iloc[i,2]+df.iloc[i,3])/2)
        box_ratio = (df.iloc[i+1,4]/1000)/(df.iloc[i+1,2]+df.iloc[i+1,3])
        emv.append(distance_moved/box_ratio*0.01)
    emv = [emv[0]] + emv
    return emv




def pl(df,length): #psichological line
    output = []
    df = pad_df(df,length)
    for i in range(len(df)-(length-1)):
        up_day = 0
        for j in range(length):
            if df.iloc[i+j,1]-df.iloc[i+j,0]>0:
                up_day += 1
        output.append(up_day/length/10)
    return output




