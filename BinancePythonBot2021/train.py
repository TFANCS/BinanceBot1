from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from time import sleep
import tensorflow as tf
import const
import os

def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)
def test_sin(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    value = sin(x)+noise
    value = tf.keras.utils.normalize(value, axis=0, order=2)
    return value[0] #it changes shape so takes [0]
def make_sin_dataset(orig_data):

    data, target = [], []
    time_length = 25

    for i in range(len(orig_data)-time_length):  #get [i]~[i+time_length] as data and get [i+time_length] as target
        data.append(orig_data[i:i + time_length])
        target.append(orig_data[i + time_length])

    re_data = np.array(data).reshape(len(data), time_length, 1)
    re_target = np.array(target).reshape(len(data), 1)


    return re_data, re_target





def make_dataset(orig_data):

    data, target = [], []
    time_length = 50


    for i in range(len(orig_data)-time_length):  #get [i]~[i+time_length] as data and get [i+time_length] as target
        data.append(orig_data.iloc[i:i + time_length,:]) #row i to i + time_length-1
        target.append(orig_data.iloc[i + time_length,1]) #row i + time_length of column 1(Close price)

    re_data = np.array(data).reshape(len(data), time_length, len(orig_data.columns))
    re_target = np.array(target).reshape(len(data), 1)


    return re_data, re_target




def train(binance):


    df_list = {}

    for symbol in const.PAIR_SYMBOLS:
        df_list[symbol] = pd.read_csv("..\\Data\\" + symbol + "_data.csv", index_col=0, parse_dates=True)
        #mpf.plot(df_list[symbol], type='candle')
        df_list[symbol] = tf.keras.utils.normalize(df_list[symbol], axis=0, order=2)


    #data, target = make_sin_dataset(test_sin())
    data, target = make_dataset(df_list["BTCUSDT"])

    checkpoint_path = "training/ckpt-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=50)


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128, activation="tanh", recurrent_activation="sigmoid"))
    model.add(tf.keras.layers.Dense(512, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0015),
              loss="mean_squared_error")

    epochs = 100
    batch_size = 20

    #model.fit(x_train, y_train, epochs=5)
    stack = model.fit(data, target,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              )

    x = range(epochs)
    plt.plot(x, stack.history["loss"])
    plt.title("loss")
    plt.show()
    
    p_data, _ = make_dataset(df_list["ETHUSDT"])
    predicted = model.predict(p_data)
    predicted = np.pad(predicted,[[0,50],[0,0]],"edge")
    df = pd.DataFrame(predicted)
    addplot = mpf.make_addplot(df)
    mpf.plot(df_list["BTCUSDT"], type='candle', addplot = addplot)



    """
    plt.figure()
    plt.plot(range(25,len(predicted)+25),predicted, color="r", label="predict_data")
    plt.plot(range(0, len(f)), f, color="b", label="orig_data")
    plt.legend()
    plt.show()
    """

    sleep(1)




