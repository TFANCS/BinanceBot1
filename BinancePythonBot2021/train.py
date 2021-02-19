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
import kerastuner as kt


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)



"""
def model_builder(hp): #for hyperparameter tuning
    hp_dense_units1 = hp.Int('dense_units1', min_value = 20, max_value = 800, step = 20)
    hp_dense_units2 = hp.Int('dense_units2', min_value = 20, max_value = 500, step = 20)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hp_dense_units1, activation="tanh"), input_shape=(const.TIME_LENGTH, 31)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hp_dense_units2, activation="swish"), input_shape=(const.TIME_LENGTH, 31)))
    model.add(tf.keras.layers.LSTM(80, activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.5, return_sequences = True))
    model.add(tf.keras.layers.LSTM(80, activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.5))
    model.add(tf.keras.layers.Dense(800, activation="swish"))
    model.add(tf.keras.layers.Dropout(0.6))
    model.add(tf.keras.layers.Dense(400, activation="tanh"))
    model.add(tf.keras.layers.Dense(200, activation="swish"))
    model.add(tf.keras.layers.Dropout(0.6))
    model.add(tf.keras.layers.Dense(75, activation="tanh"))
    model.add(tf.keras.layers.Dense(50, activation="swish"))
    model.add(tf.keras.layers.Dropout(0.6))
    model.add(tf.keras.layers.Dense(25, activation="tanh"))
    model.add(tf.keras.layers.Dense(3, activation="swish"))
    optimizer = tf.keras.optimizers.Adam(lr=0.00005)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
"""




def train(binance,model):


    df_list = {}
    df_list_test = {}

    for symbol in const.PAIR_SYMBOLS:
        df_list[symbol] = pd.read_csv("..\\Data\\" + symbol + "_data.csv", index_col=0, parse_dates=True)
        #df_list[symbol] = tf.keras.utils.normalize(df_list[symbol], axis=0, order=2)

    #df_list_merged = pd.concat(df_list, axis=1)

    df_test = make_dataset.make_current_data(binance,symbol,0,0)

    data, target = make_dataset.make_dataset(df_list["BTCUSDT"])
    data_test, target_test = make_dataset.make_dataset(df_test)

    checkpoint_dir = os.path.dirname(const.CHECKPOINT_PATH.format(time_length=const.TIME_LENGTH))
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        const.CHECKPOINT_PATH.format(time_length=const.TIME_LENGTH), 
        verbose=1, 
        save_weights_only=True,
        save_freq=3000)
    lr_change_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    epochs = 30
    batch_size = 5


    #tuner = kt.Hyperband(model_builder,
    #                 objective = "val_accuracy", 
    #                 max_epochs = 10,
    #                 factor = 3,
    #                 directory = "HyperparameterTunerData",
    #                 project_name = "tunerData")
    #tuner.search(data, target, epochs=3, validation_data=(data_test, target_test))
    #tuner.results_summary()

    stack = model.fit(data, target,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[checkpoint_callback,lr_change_callback],
              validation_data=(data_test, target_test)
              )
    model.summary()

    x = range(epochs)
    plt.plot(x, stack.history["loss"])
    plt.plot(x, stack.history["val_loss"])
    plt.legend(["loss", "val_loss"], loc="upper left")
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
    
    
    test_data, test_target = make_dataset.make_dataset(df_test)
    test_loss, test_acc = model.evaluate(test_data,  test_target, verbose=2)
    print("Test accuracy:", test_acc)

    




