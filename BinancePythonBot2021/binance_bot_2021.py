from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime
import mplfinance as mpf
import os
import const
import collect_info
import train


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


    def get_klines(self, pair):
        try:
            klines = pd.DataFrame(self.client.get_klines(symbol=pair, interval=Client.KLINE_INTERVAL_5MINUTE, limit=500),columns = ["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVolume","TradeCount","TakerVolume","TakerQuoteVolume","Ignore"])
            value = klines[["OpenTime","Open","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount"]].copy()
            value.loc[:, "OpenTime"] = pd.to_datetime(value["OpenTime"].apply(lambda x: datetime.fromtimestamp(int(x/1000))))
            value = value.set_index("OpenTime")
            return value;
        except Exception as e:
            print("Exception : " + str(e));


    def get_historical_klines(self, pair, start, end):
        try:
            klines = pd.DataFrame(self.client.get_historical_klines(start_str = start,end_str = end,symbol=pair, interval=Client.KLINE_INTERVAL_5MINUTE, limit=500),columns = ["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVolume","TradeCount","TakerVolume","TakerQuoteVolume","Ignore"])
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


    def create_buy_order(self, symbol, price, quantity):
        try:
            order = self.client.create_test_order(
            symbol=symbol,
            side=self.client.SIDE_BUY,
            type=self.client.ORDER_TYPE_LIMIT,
            timeInForce=self.client.TIME_IN_FORCE_IOC,
            price=price,
            quantity=quantity)
            print(order)
            print("buy order created.\nSymbol:{0:5}\nPrice:{1:5}\nQuantity:{2:5}",symbol,price,quantity)
        except Exception as e:
            print("Exception : " + str(e));


    def create_sell_order(self, symbol, price, quantity):
        try:
            order = self.client.create_test_order(
            symbol=symbol,
            side=self.client.SIDE_SELL,
            type=self.client.ORDER_TYPE_LIMIT,
            timeInForce=self.client.TIME_IN_FORCE_IOC,
            price=price,
            quantity=quantity)
            print("sell order created.\nSymbol:{0:5}\nPrice:{1:5}\nQuantity:{2:5}",symbol,price,quantity)
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

    print("0:CollectInfo 1:Train 2:TestTrade 3:Trade")
    mode = input(">")
    if mode == "0":
        collect_info.collect_info(binance)
    elif mode == "1":
        train.train(binance)










if __name__ == '__main__':
    main()

