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
from functools import reduce



def scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr * tf.math.exp(-0.1)



def train(binance,model):


    df_list = {}
    df_list_test = {}

    for symbol in const.PAIR_SYMBOLS:
        df_list[symbol] = pd.read_csv("..\\Data\\" + symbol + "_data.csv", index_col=0, parse_dates=True)
        #df_list[symbol] = tf.keras.utils.normalize(df_list[symbol], axis=0, order=2)
        df_list_test[symbol] = pd.read_csv("..\\Data\\" + symbol + "_data_test.csv", index_col=0, parse_dates=True)
        #df_list_test[symbol] = tf.keras.utils.normalize(df_list_test[symbol], axis=0, order=2)

    



    data, target = make_dataset.make_dataset(df_list["BTCUSDT"])
    data_test, target_test = make_dataset.make_dataset(df_list_test["BTCUSDT"])

    checkpoint_dir = os.path.dirname(const.CHECKPOINT_PATH)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        const.CHECKPOINT_PATH, 
        verbose=1, 
        save_weights_only=True,
        save_freq=3000)
    lr_change_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    epochs = 20
    batch_size = 4

    stack = model.fit(data, target,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[checkpoint_callback,lr_change_callback],
              validation_data=(data_test, target_test)
              )

    x = range(epochs)
    plt.plot(x, stack.history["loss"])
    plt.title("loss")
    plt.show()
    

    #p_data, _ = make_dataset.make_dataset(df_list_test)
    #predicted = model.predict(p_data)
    #predicted = np.pad(predicted,[[0,const.TIME_LENGTH],[0,0]],"edge")
    #df = pd.DataFrame(predicted)
    #for i in range(100):
    #    print(df.iloc[i,:])
    #df = df.idxmax(axis=1)
    #df = df.columns.get_loc(df.idxmax(axis=1))
    #addplot = mpf.make_addplot(df)
    #mpf.plot(df_list["BTCUSDT"], type='candle', addplot = addplot)
    

    test_data, test_target = make_dataset.make_dataset(df_list_test["BTCUSDT"])
    test_loss, test_acc = model.evaluate(test_data,  test_target, verbose=2)
    print("Test accuracy:", test_acc)

    
    #plt.figure()
    #plt.plot(range(25,len(predicted)+25),predicted, color="r", label="predict_data")
    #plt.plot(range(0, len(f)), f, color="b", label="orig_data")
    #plt.legend()
    #plt.show()
    

    sleep(1)




