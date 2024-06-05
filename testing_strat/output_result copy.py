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
import math
import logging
from binance.um_futures import UMFutures
# from binance.lib.utils import config_logging
from binance.error import ClientError
# config_logging(logging, logging.DEBUG)
# from numba import jit, cuda

# https://www.google.com/search?q=use+graphics+card+to+run+python&oq=use+graphics+card+to+run+python&aqs=chrome..69i57j33i160l2.7968j0j4&sourceid=chrome&ie=UTF-8
class Data_collector:
    def __init__ (self):
        self.symbol = "btcusdt"
        self.interval = "3m"
        self.volstream = "wss://fstream.binance.com/ws/btcusdt@aggTrade"
        self.websocket_url = f"wss://fstream.binance.com/ws/{self.symbol}@kline_{self.interval}"
        self.sell_volume = 0
        self.buy_volume = 0
        self.countdown = 0

    def on_message_vol(self, ws, message):
        data = json.loads(message)
        market_maker = data['m']
        quantity = float(data['q'])
        if market_maker:
            self.sell_volume += quantity
        else:
            self.buy_volume += quantity
        vol_ratio = self.buy_volume/self.sell_volume
        self.countdown += 1
        print(f"{self.countdown} - Buy: {round(self.buy_volume, 4)}, Sell: {round(self.sell_volume, 4)}, BuyRatio: {round(vol_ratio, 4)}")

    def on_message_kline(self, ws, message):
        data = json.loads(message)
        is_closed = data['k']['x']
        if is_closed:
            self.is_candle_closed = True
            print("__________________3 Min closed__________________")
            # Reset the buy and sell volumes for the next candle
            self.buy_volume = 0
            self.sell_volume = 0
            self.countdown = 0
       
    def websocket_thread_vol(self):
        ws = wb.WebSocketApp(url=self.volstream, on_message=self.on_message_vol, on_close=self.on_close, on_error=self.on_error)
        ws.run_forever()

    def websocket_thread_kline(self):
        ws = wb.WebSocketApp(url=self.websocket_url, on_message=self.on_message_kline, on_close=self.on_close, on_error=self.on_error)
        ws.run_forever()


    def on_close(self, ws):
        print("closed")

    def on_error(self, ws, error):
        print(error)

if __name__ == "__main__":
    bot = Data_collector()
    websocket_thread_vol = Thread(target=bot.websocket_thread_vol)
    websocket_thread_vol.start()

    # Start WebSocket thread for kline updates
    websocket_thread_kline = Thread(target=bot.websocket_thread_kline)
    websocket_thread_kline.start()