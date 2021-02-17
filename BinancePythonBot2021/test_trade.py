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



def test_trade(binance,model):


    while True:
        df = make_dataset.make_current_data(binance,symbol,0,0)
        print(df)

        input_data = np.array(df).reshape(1, const.TIME_LENGTH, len(input_data.columns))
        predicted = model.predict(input_data)
        print(predicted)





        sleep(5)


