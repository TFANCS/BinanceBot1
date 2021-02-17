from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime
import mplfinance as mpf
import tensorflow as tf
import os
import const
import collect_info
import train
import simulation
import make_dataset
import technical_indicators
import test_trade





class BinanceAPI:

    def __init__(self, api_key, api_secret):
        API_KEY = api_key;
        API_SECRET = api_secret;

        self.client = Client(API_KEY, API_SECRET);


    def get_ticker(self, pair):
        try:
            value = self.client.get_ticker(symbol=pair);
            return value;
        except Exception as e:
            print("Exception : " + str(e));


    def get_klines(self, pair, number):
        try:
            klines = pd.DataFrame(self.client.get_klines(symbol=pair, interval=Client.KLINE_INTERVAL_15MINUTE, limit=number),columns = ["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVolume","TradeCount","TakerVolume","TakerQuoteVolume","Ignore"])
            value = klines[["OpenTime","Open","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount"]].copy()
            value.loc[:, "OpenTime"] = pd.to_datetime(value["OpenTime"].apply(lambda x: datetime.fromtimestamp(int(x/1000))))
            value = value.set_index("OpenTime")
            return value;
        except Exception as e:
            print("Exception : " + str(e));


    def get_historical_klines(self, pair, start, end):
        try:
            klines = pd.DataFrame(self.client.get_historical_klines(start_str = start,end_str = end,symbol=pair, interval=Client.KLINE_INTERVAL_15MINUTE, limit=500),columns = ["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVolume","TradeCount","TakerVolume","TakerQuoteVolume","Ignore"])
            value = klines[["OpenTime","Open","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount"]].copy()
            value.loc[:, "OpenTime"] = pd.to_datetime(value["OpenTime"].apply(lambda x: datetime.fromtimestamp(int(x/1000))))
            value = value.set_index("OpenTime")
            return value;
        except Exception as e:
            print("Exception : " + str(e));


    def get_balance(self, symbol):
        try:
            value = self.client.get_asset_balance(asset=symbol)
            return value
        except Exception as e:
            print('Exception : {}'.format(e))


    def create_order(self, symbol, price, quantity, side_str, type_str):
        try:
            if side_str == "BUY":
                side = self.client.SIDE_BUY
            elif side_str == "SELL":
                side = self.client.SIDE_SELL
            if type_str == "LIMIT":
                type = self.client.ORDER_TYPE_LIMIT
            elif type_str == "MARKET":
                type = self.client.ORDER_TYPE_MARKET
            order = self.client.create_test_order(
            symbol=symbol,
            side=side,
            type=type,
            timeInForce=self.client.TIME_IN_FORCE_IOC,
            price=price,
            quantity=quantity)
            print(order)
            print("buy order created.\nSymbol:{0:5}\nPrice:{1:5}\nQuantity:{2:5}",symbol,price,quantity)
        except Exception as e:
            print("Exception : " + str(e));




    def get_base_asset(self, symbol):
        try:
            return self.client.get_symbol_info(symbol)["baseAsset"];
        except Exception as e:
            print("Exception : " + str(e));


    def get_quote_asset(self, symbol):
        try:
            return self.client.get_symbol_info(symbol)["quoteAsset"];
        except Exception as e:
            print("Exception : " + str(e));

    def get_all_tickers(self):
        try:
            return self.client.get_all_tickers();
        except Exception as e:
            print("Exception : " + str(e));

    def get_all_orders(self):
        try:
            return self.client.get_all_orders();
        except Exception as e:
            print("Exception : " + str(e));




def main():

    with open("ApiKey.txt") as f:
        api_key = f.readline().rstrip('\n')
        api_secret = f.readline().rstrip('\n')


    binance = BinanceAPI(api_key, api_secret);


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(100, activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(500, activation="swish"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(250, activation="tanh"))
    model.add(tf.keras.layers.Dense(125, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(75, activation="tanh"))
    model.add(tf.keras.layers.Dense(50, activation="swish"))
    model.add(tf.keras.layers.Dense(25, activation="tanh"))
    model.add(tf.keras.layers.Dense(3, activation="swish"))
    optimizer = tf.keras.optimizers.Adam(lr=0.000075)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


    print("0:CollectData 1:Train 2:Simulation 3:TestTrade 4:Trade")
    print("A:TestIndicators")
    mode = input(">")
    if mode == "0":
        collect_info.collect_info(binance)
    elif mode == "1":
        train.train(binance,model)
    elif mode == "2":
        simulation.simulation(binance,model)
    elif mode == "3":
        test_trade.test_trade(binance,model)
    elif mode == "A":
        technical_indicators.test(binance)








if __name__ == '__main__':
    main()

