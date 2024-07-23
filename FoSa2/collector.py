from binance.um_futures import UMFutures
from binance.error import ClientError
from threading import Timer
from threading import Thread
from datetime import datetime
import numpy as np
import ta
import pandas as pd
import requests
import websocket as wb
import json
import os 

class CollectData:
    def __init__(self, symbol, interval) -> None:
        # 차트 분봉 
        self.symbol = symbol
        self.interval = interval
        self.volstream = f"wss://fstream.binance.com/ws/{self.symbol}@aggTrade"
        self.websocket_url = f"wss://fstream.binance.com/ws/{self.symbol}@kline_{self.interval}"  

    def on_message_kline(self, ws, message):
        data = json.loads(message)
        kline_data = data['k']
        df2 = {
            'openTime': kline_data['t'],
            'open': float(kline_data['o']),
            'high': float(kline_data['h']),
            'low': float(kline_data['l']),
            'close': float(kline_data['c']),
            'volume': float(kline_data['v'])
        }
        self.live_edit(df2)
        if kline_data['x']:  # If the candle is closed
            self.main_df = self.add_frame(df2)
            self.reset_volumes()

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed with code:", close_status_code, "and message:", close_msg)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def websocket_thread_kline(self):
        ws = wb.WebSocketApp(
            url=self.websocket_url,
            on_message=self.on_message_kline,
            on_close=self.on_close,
            on_error=self.on_error
        )
        ws.run_forever()

    def get_prev_data(self) -> pd.DataFrame:
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {'symbol': self.symbol, 'interval': self.interval, 'limit': self.prev_limit}
        response = requests.get(url, params=params).json()
        columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
        return df[columns].astype(float)

    def add_frame(self, df2):
        self.main_df = pd.concat([self.main_df, pd.DataFrame([df2])], ignore_index=True)
        return self.main_df

    def live_edit(self, df2):
        self.main_df.iloc[-1] = list(df2.values())
        if len(self.main_df) >= self.prev_limit * 1.5:
            self.main_df = self.main_df.iloc[int(self.prev_limit / 2):].reset_index(drop=True)

    def reset_volumes(self):
        self.buy_volume = 0
        self.sell_volume = 0

    def update_trade_status(self):
        self.previous_active = self.current_active
        self.current_active = self.trade.check_open_orders()
