import numpy as np
import ta
import pandas as pd
import requests
from threading import Thread
from binance.cm_futures import CMFutures
import websocket as wb
import json
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
        self.main_df = self.get_prev_data()
        self.is_candle_closed = False
        self.position_status = False
        self.position = None
        self.enter_price = None
        self.quantity = None
        self.trade = BinanceTrade()
        self.balance = float(self.trade.balance())
        self.results_file = "trade_results.txt"
        self.price_profit = None
        self.price_stoploss = None

    def on_message_vol(self, ws, message):
        data = json.loads(message)
        market_maker = data['m']
        quantity = float(data['q'])
        if market_maker:
            self.sell_volume += quantity
        else:
            self.buy_volume += quantity

    def on_message_kline(self, ws, message):
        data = json.loads(message)
        openTime = data['k']['t']
        Open = float(data['k']['o'])
        High = float(data['k']['h'])
        Low = float(data['k']['l'])
        Close = float(data['k']['c'])
        Volume = float(data['k']['v'])
        isClosed = data['k']['x']
        df2 = {'openTime': openTime, 'Open': Open, 'High': High, 'Low': Low, 'Close': Close, 'Volume': Volume}
        self.live_edit(df2)
        if isClosed:
            self.main_df = self.add_frame(df2)
            self.buy_volume = 0
            self.sell_volume = 0
        if self.position_status:
            self.close_position(Close)
        else:
            self.open_position(Close)

    def websocket_thread_vol(self):
        ws = wb.WebSocketApp(url=self.volstream, on_message=self.on_message_vol)
        ws.run_forever()

    def websocket_thread_kline(self):
        ws = wb.WebSocketApp(url=self.websocket_url, on_message=self.on_message_kline)
        ws.run_forever()

    def get_prev_data(self) -> pd.DataFrame:
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {'symbol': self.symbol, 'interval': self.interval, 'limit': 100}
        response = requests.get(url, params=params).json()
        df = pd.DataFrame(response, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
        df = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
        df = df.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'float'})
        return df

    def live_edit(self, df2):
        df2 = list(df2.values())
        self.main_df.iloc[-1] = df2

    def open_position(self, current_price):
        if not self.position_status:
            status = self.decision(current_price)
            if status != "pass":
                self.balance = float(self.trade.balance())
                self.quantity = round(((self.balance * 25) / current_price) * 0.85, 3)
                if status == "long":
                    self.long(current_price)
                elif status == "short":
                    self.short(current_price)

    def close_position(self, current_price):
        if self.position_status:
            if (self.position == "long" and (current_price >= self.price_profit or current_price <= self.price_stoploss)) or \
               (self.position == "short" and (current_price <= self.price_profit or current_price >= self.price_stoploss)):
                self.trade.order(symbol=self.symbol.upper(), side="SELL" if self.position == "long" else "BUY", quantity=self.quantity, reduce_only=True)
                result = "profit" if ((self.position == "long" and current_price >= self.price_profit) or
                                      (self.position == "short" and current_price <= self.price_profit)) else "loss"
                profit_loss_percent = ((current_price - self.enter_price) / self.enter_price) * 100 * (1 if self.position == "long" else -1)
                self.position_status = False
                self.save_result(f"Closed {self.position} position at {current_price} with {result} ({profit_loss_percent:.2f}%)")
                print(f"Closed {self.position} position at {current_price} with {result} ({profit_loss_percent:.2f}%)")

    def long(self, current_price):
        self.enter_price = current_price
        self.in_atr = self.ATR().iloc[-1]
        self.price_profit = round(self.enter_price + (self.in_atr * 2), 1)
        self.price_stoploss = round(self.enter_price - (self.in_atr * 1), 1)
        self.position = "long"
        self.position_status = True
        self.trade.order(symbol=self.symbol.upper(), side="BUY", quantity=self.quantity)
        print(f"Opened long position at {current_price}, Profit Target: {self.price_profit}, Stop Loss: {self.price_stoploss}")

    def short(self, current_price):
        self.enter_price = current_price
        self.in_atr = self.ATR().iloc[-1]
        self.price_profit = round(self.enter_price - (self.in_atr * 2), 1)
        self.price_stoploss = round(self.enter_price + (self.in_atr * 1), 1)
        self.position = "short"
        self.position_status = True
        self.trade.order(symbol=self.symbol.upper(), side="SELL", quantity=self.quantity)
        print(f"Opened short position at {current_price}, Profit Target: {self.price_profit}, Stop Loss: {self.price_stoploss}")

    def save_result(self, message):
        with open(self.results_file, "a") as file:
            file.write(message + "\n")

class BinanceTrade:
    def __init__(self):
        self.api_key = os.getenv('Bin_API_KEY')
        self.api_secret = os.getenv('Bin_SECRET_KEY')
        self.um_futures_client = UMFutures(key=self.api_key, secret=self.api_secret)

    def balance(self):
        try:
            response = self.um_futures_client.balance()
            balance = next(x for x in response if x['asset'] == "USDT")['balance']
            return balance
        except ClientError as e:
            print(f"Error fetching balance: {e}")
            return None

    def order(self, symbol, side, quantity, reduce_only=False):
        try:
            order = self.um_futures_client.new_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                reduce_only=reduce_only
            )
            return order
        except ClientError as e:
            print(f"Error placing order: {e}")
            return None

if __name__ == "__main__":
    bot = DataCollector()
    websocket_thread_vol = Thread(target=bot.websocket_thread_vol)
    websocket_thread_vol.start()

    websocket_thread_kline = Thread(target=bot.websocket_thread_kline)
    websocket_thread_kline.start()
