import json
import time
import logging
import asyncio
import pandas as pd
import numpy as np
import requests
import websocket as wb
from threading import Thread

import config
from src.client import BinanceClient
from src.strategy import TechnicalStrategy
from src.ml_models import PredictionModel

class TradingBot:
    def __init__(self):
        self.main_df = pd.DataFrame()
        self.symbol = config.SYMBOL.lower()
        self.interval = config.INTERVAL
        self.price_profit = None
        self.price_stoploss = None
        self.position_status = False 
        self.position = None 
        self.enter_price = None
        self.quantity = None
        self.ai_result = None
        
        self.websocket_url = config.WS_URL
        self.url = config.API_URL
        self.params = {
            'symbol': config.SYMBOL.lower(),
            'interval': config.INTERVAL,
            'limit': config.LIMIT
        }
        
        self.main_df = self.get_prev_data()
        self.trade = BinanceClient()
        self.strategy = TechnicalStrategy()
        self.ml_model = PredictionModel()
        self.balance = self.trade.get_balance()
        self.win = None
        self.lose = None

    def get_prev_data(self):
        response = requests.get(self.url, params=self.params)
        response = response.json()
        self.main_df = pd.DataFrame(response, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime',
                                            'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore']) 
        self.main_df = self.main_df.drop(self.main_df.columns[[6,7,8,9,10,11]], axis=1)
        self.main_df.loc[len(self.main_df)] = pd.Series()
        self.main_df = self.main_df.iloc[:-1]
        self.main_df['Category'] = np.where(self.main_df['Close'] > self.main_df['Open'], 'bullish', 'bearish')
        return self.main_df

    def add_frame(self, df2):
        self.main_df.loc[len(self.main_df)] = df2
        return self.main_df

    def live_edit(self, df2):    
        df2 = list(df2.values())
        self.main_df.iloc[-1] = df2
        length_df = len(self.main_df)
        if length_df == 80:
            self.main_df = self.main_df.drop(self.main_df.index[:11])
            self.main_df = self.main_df.reset_index(drop=True)

    async def prediction_model(self):
        self.ai_result = await self.ml_model.predict_trend(self.main_df)

    def decision(self, current_price, close_list, open_list, high_list, low_list):
        self.main_df['High'] = pd.to_numeric(self.main_df['High'], errors='coerce')
        self.main_df['Low'] = pd.to_numeric(self.main_df['Low'], errors='coerce')
        self.main_df['Close'] = pd.to_numeric(self.main_df['Close'], errors='coerce')
        self.main_df['Volume'] = pd.to_numeric(self.main_df['Volume'], errors='coerce')     
        ema_fourteen_list, ema_eight_list = self.strategy.EMA(self.main_df)
        two_d, two_k = self.strategy.stochRSI(self.main_df)
        
        prev_d, curr_d = two_d[0], two_d[1]
        prev_k, curr_k = two_k[0], two_k[1]
        prev_open, curr_open = float(open_list[-2]), float(open_list[-1])
        kd_prev_diff, kd_curr_diff = prev_k - prev_d , curr_k - curr_d

        prev_grad_ema_fourteen = ema_fourteen_list[-2] - ema_fourteen_list[-3]
        curernt_grad_ema_fourteen = ema_fourteen_list[-1] - ema_fourteen_list[-2]    
        prev_grad_ema_eight = ema_eight_list[-2] - ema_eight_list[-3]
        curernt_grad_ema_eight = ema_eight_list[-1] - ema_eight_list[-2]  
    
        print('__________________________________________________________________________________________________________________________________________________\n'
              f'kd_prev_diff > 0: {kd_prev_diff > 0}\nkd_curr_diff > 0: {kd_curr_diff > 0}\n'
              f'prev_grad_ema_fourteen > 0: {prev_grad_ema_fourteen > 0}\n'
              f'curernt_grad_ema_fourteen > 0 : {curernt_grad_ema_fourteen > 0 }\nprev_grad_ema_eight > 0: {prev_grad_ema_eight > 0}\n'
              f'curernt_grad_ema_eight > 0: {curernt_grad_ema_eight > 0}\n'
              f'current_price > curr_open: {current_price > curr_open}\ndanger_check: {self.strategy.danger_check(high_list, low_list)}\n'
              f'AI: {self.ai_result}\n')

        if ((kd_prev_diff > 0 and kd_curr_diff > 0) and prev_grad_ema_fourteen > 0 and 
            curernt_grad_ema_fourteen > 0 and prev_grad_ema_eight > 0 and curernt_grad_ema_eight > 0 and 
            current_price > curr_open and 
            self.strategy.danger_check(high_list, low_list) and
            self.ai_result == "bullish"):
            return "long"
        elif ((kd_prev_diff < 0 and kd_curr_diff < 0) and prev_grad_ema_fourteen < 0 and 
            curernt_grad_ema_fourteen < 0 and prev_grad_ema_eight < 0 and curernt_grad_ema_eight < 0 and 
            current_price < curr_open and 
            self.strategy.danger_check(high_list, low_list) and
            self.ai_result == "bearish"):
            return "short"
        else:
            return "pass"

    def close_position(self, current_price):
        gain = 40
        loss = 70
        if self.position == "long" and self.position_status:
            if current_price >= self.enter_price + gain or current_price <= self.enter_price - loss:
                self.trade.place_order(config.SYMBOL, "SELL", True, quantity=self.quantity)
                self.position_status = False 
                self.position = None
                time.sleep(30)
        elif self.position == "short" and self.position_status:
            if current_price <= self.enter_price - gain or current_price >= self.enter_price + loss:
                self.trade.place_order(config.SYMBOL, "BUY", True, quantity=self.quantity)
                self.position = None
                self.position_status = False
                time.sleep(30)

    def open_position(self, current_price, close_list, open_list, high_list, low_list):
        status = self.decision(current_price, close_list, open_list, high_list, low_list)
        self.balance = self.trade.get_balance()
        self.quantity = round(((self.balance * config.LEVERAGE) / current_price) * config.QUANTITY_MULTIPLIER, 3)
        if status == "long" and self.position_status == False:
            self.trade.place_order(config.SYMBOL, "BUY", False, quantity=self.quantity)
            self.long(current_price)
        elif status == "short" and self.position_status == False:
            self.trade.place_order(config.SYMBOL, "SELL", False, quantity=self.quantity)
            self.short(current_price)
        else:
            pass

    def long(self, current_price):
        self.enter_price = current_price
        self.price_profit = round(self.enter_price + 5, 1)
        self.price_stoploss = round(self.enter_price - 10, 1)
        print(f"익절: {self.price_profit}\n손절: {self.price_stoploss}") 
        self.amount = self.balance/current_price
        self.position = "long"
        self.position_status = True
        print(f"롱: {current_price} (실제 진입 가격은 다를 수 있음)")

    def short(self, current_price):
        self.enter_price = current_price
        self.price_profit = round(self.enter_price - 5, 1)
        self.price_stoploss = round(self.enter_price + 10, 1)
        print(f"익절: {self.price_profit}\n손절: {self.price_stoploss}") 
        self.amount = self.balance/current_price
        self.position = "short"
        self.position_status = True
        print(f"숏: {current_price} (실제 진입 가격은 다를 수 있음)")    

    def on_message(self, ws, message):
        data = json.loads(message)
        openTime = data['k']['t']
        Open = data['k']['o']
        High = data['k']['h']
        Low = data['k']['l']
        Close = data['k']['c']
        Volume = data['k']['v']
        isClosed = data['k']['x']
        Status = (float(Open) - float(Close)) > 0 
        if Status == True:
            Status = "bearish"
        else:
            Status = "bullish"
        self.balance = self.trade.get_balance()
        df2 = {'openTime': openTime, 'Open': Open, 'High': High, 'Low': Low, 'Close': Close, 'Volume': Volume, 'Category': Status}
        self.live_edit(df2) 
        if self.position_status == False:
            asyncio.run(self.prediction_model())
            self.open_position(float(df2['Close']), (self.main_df['Close'].tail(5)).to_list(), self.main_df['Open'].to_list(), self.main_df['High'].to_list(), self.main_df['Low'].to_list())
        elif self.position_status == True:
            self.close_position(float(df2['Close']))
        if isClosed == True:
            self.main_df = self.add_frame(df2)
            time.sleep(5)

    def on_close(self, ws):
        print("closed")

    def on_error(self, ws, error):
        print(error)

    def websocket_thread(self):
        ws = wb.WebSocketApp(url=self.websocket_url, on_message=self.on_message, on_close=self.on_close, on_error=self.on_error)
        ws.run_forever()
