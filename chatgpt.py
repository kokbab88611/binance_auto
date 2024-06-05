import numpy as np
import ta
import pandas as pd
import matplotlib.pyplot as plt
import requests
from threading import Thread
from binance.cm_futures import CMFutures
import websocket as wb
import json
import time
import os
import logging
from binance.um_futures import UMFutures
from binance.error import ClientError

class DataCollector:
    def __init__(self):
        self.symbol = "btcusdt"
        self.interval = "3m"
        self.volstream = "wss://fstream.binance.com/ws/btcusdt@aggTrade"
        self.websocket_url = f"wss://fstream.binance.com/ws/{self.symbol}@kline_{self.interval}"
        self.sell_volume = 0
        self.buy_volume = 0
        self.countdown = 0
        self.main_df = self.get_prev_data()
        self.is_candle_closed = False
        self.position_status = False
        self.position = None
        self.enter_price = None
        self.quantity = None
        self.trade = BinanceTrade()
        self.balance = float(self.trade.balance())

    def on_message_vol(self, ws, message):
        data = json.loads(message)
        market_maker = data['m']
        quantity = float(data['q'])
        if market_maker:
            self.sell_volume += quantity
        else:
            self.buy_volume += quantity
        vol_ratio = self.buy_volume / self.sell_volume
        self.countdown += 1
        print(f"{self.countdown} - Buy: {round(self.buy_volume, 4)}, Sell: {round(self.sell_volume, 4)}, BuyRatio: {round(vol_ratio, 4)}")

    def on_message_kline(self, ws, message):
        data = json.loads(message)
        openTime = data['k']['t']
        Open = data['k']['o']
        High = data['k']['h']
        Low = data['k']['l']
        Close = data['k']['c']
        Volume = data['k']['v']
        isClosed = data['k']['x']
        df2 = {'openTime': openTime, 'Open': Open, 'High': High, 'Low': Low, 'Close': Close, 'Volume': Volume}
        self.live_edit(df2)
        if self.position_status == False:
            self.open_position(float(df2['Close']), (self.main_df['Close'].tail(5)).to_list(), self.main_df['Open'].to_list(), self.main_df['High'].to_list(), self.main_df['Low'].to_list())
        elif self.position_status == True:
            self.close_position(float(df2['Close']))
        if isClosed:
            self.main_df = self.add_frame(df2)
            print("__________________3 Min closed__________________")
            self.buy_volume = 0
            self.sell_volume = 0
            self.countdown = 0
        self.balance = float(self.trade.balance())

    def on_close(self, ws):
        print("closed")

    def on_error(self, ws, error):
        print(error)

    def websocket_thread_vol(self):
        ws = wb.WebSocketApp(url=self.volstream, on_message=self.on_message_vol, on_close=self.on_close, on_error=self.on_error)
        ws.run_forever()

    def websocket_thread_kline(self):
        ws = wb.WebSocketApp(url=self.websocket_url, on_message=self.on_message_kline, on_close=self.on_close, on_error=self.on_error)
        ws.run_forever()

    def get_prev_data(self) -> pd.DataFrame:
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {'symbol': self.symbol, 'interval': self.interval, 'limit': 100}
        response = requests.get(url, params=params).json()
        df = pd.DataFrame(response, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
        df = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
        return df

    def add_frame(self, df2):
        self.main_df.loc[len(self.main_df)] = df2
        return self.main_df

    def live_edit(self, df2):
        df2 = list(df2.values())
        self.main_df.iloc[-1] = df2
        if len(self.main_df) == 55:
            self.main_df = self.main_df.drop(self.main_df.index[:15]).reset_index(drop=True)

    def EMA(self):
        ema_short = ta.trend.EMAIndicator(self.main_df['Close'], window=9).ema_indicator()
        ema_medium = ta.trend.EMAIndicator(self.main_df['Close'], window=21).ema_indicator()
        ema_long = ta.trend.EMAIndicator(self.main_df['Close'], window=50).ema_indicator()
        return ema_short, ema_medium, ema_long

    def RSI(self):
        return ta.momentum.RSIIndicator(self.main_df['Close'], window=14).rsi()

    def VWAP(self):
        return ta.volume.VolumeWeightedAveragePrice(self.main_df['High'], self.main_df['Low'], self.main_df['Close'], self.main_df['Volume']).volume_weighted_average_price()

    def ATR(self):
        return ta.volatility.AverageTrueRange(self.main_df['High'], self.main_df['Low'], self.main_df['Close']).average_true_range()

    def volume_profile(self):
        vp = self.main_df.groupby('Close')['Volume'].sum()
        poc = vp.idxmax()
        return vp, poc

    def decision(self, current_price):
        ema_short, ema_medium, ema_long = self.EMA()
        rsi = self.RSI()
        vwap = self.VWAP()
        atr = self.ATR().iloc[-1]
        vp, poc = self.volume_profile()
        volume_threshold = atr * 1.5  # Example threshold, can be adjusted

        if current_price > vwap.iloc[-1] and ema_short.iloc[-1] > ema_medium.iloc[-1] and ema_medium.iloc[-1] > ema_long.iloc[-1] and rsi.iloc[-1] > 40 and self.buy_volume > volume_threshold:
            return "long"
        elif current_price < vwap.iloc[-1] and ema_short.iloc[-1] < ema_medium.iloc[-1] and ema_medium.iloc[-1] < ema_long.iloc[-1] and rsi.iloc[-1] < 60 and self.sell_volume > volume_threshold:
            return "short"
        else:
            return "pass"

    def close_position(self, current_price):
        if self.position == "long" and self.position_status:
            if current_price >= self.price_profit or current_price <= self.price_stoploss:
                self.trade.order("BTCUSDT", "SELL", True, self.quantity)
                self.position_status = False
                self.position = None
                print("Position closed with profit" if current_price >= self.price_profit else "Position closed with loss")
        elif self.position == "short" and self.position_status:
            if current_price <= self.price_profit or current_price >= self.price_stoploss:
                self.trade.order("BTCUSDT", "BUY", True, self.quantity)
                self.position_status = False
                self.position = None
                print("Position closed with profit" if current_price <= self.price_profit else "Position closed with loss")

    def open_position(self, current_price, close_list, open_list, high_list, low_list):
        status = self.decision(current_price)
        self.balance = float(self.trade.balance())
        self.quantity = round(((self.balance * 25) / current_price) * 0.85, 3)
        if status == "long" and self.position_status == False:
            self.trade.order("BTCUSDT", "BUY", False, self.quantity)
            self.long(current_price)
        elif status == "short" and self.position_status == False:
            self.trade.order("BTCUSDT", "SELL", False, self.quantity)
            self.short(current_price)
        else:
            pass

    def long(self, current_price):
        self.in_atr = self.ATR()
        self.in_atr = round(self.in_atr.iloc[-1], 2)
        self.enter_price = current_price
        self.price_profit = round(self.enter_price + (self.in_atr * 2), 1)
        self.price_stoploss = round(self.enter_price - (self.in_atr * 1), 1)
        self.position = "long"
        self.position_status = True
        print(f"Long: {current_price} (Entry price may vary)")

    def short(self, current_price):
        self.in_atr = self.ATR()
        self.in_atr = round(self.in_atr.iloc[-1], 2)
        self.enter_price = current_price
        self.price_profit = round(self.enter_price - (self.in_atr * 2), 1)
        self.price_stoploss = round(self.enter_price + (self.in_atr * 1), 1)
        self.position = "short"
        self.position_status = True
        print(f"Short: {current_price} (Entry price may vary)")

class BinanceTrade:
    def __init__(self):
        self.api_key = os.getenv('Bin_API_KEY')
        self.api_secret = os.getenv('Bin_SECRET_KEY')
        self.um_futures_client = UMFutures(key=self.api_key, secret=self.api_secret)

    def balance(self):
        response = self.um_futures_client.balance()
        balance = next(x for x in response if x['asset'] == "USDT")['balance']
        return balance

    def order(self, symbol, side, reduceOnly, quantity):
        try:
            quantity = float(round(quantity, 3))
            response = self.um_futures_client.new_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                reduceOnly=reduceOnly
            )
            print(response)
        except ClientError as error:
            print(f"Error: {error.status_code} Error code: {error.error_code} Error message: {error.error_message}")

if __name__ == "__main__":
    bot = DataCollector()
    websocket_thread_vol = Thread(target=bot.websocket_thread_vol)
    websocket_thread_vol.start()

    websocket_thread_kline = Thread(target=bot.websocket_thread_kline)
    websocket_thread_kline.start()
