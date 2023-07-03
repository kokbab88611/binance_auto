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
# from binance.lib.utils import config_logging
from binance.error import ClientError
# config_logging(logging, logging.DEBUG)
# from numba import jit, cuda

# https://www.google.com/search?q=use+graphics+card+to+run+python&oq=use+graphics+card+to+run+python&aqs=chrome..69i57j33i160l2.7968j0j4&sourceid=chrome&ie=UTF-8
class Data_collector:
    def __init__ (self):
        self.main_df = pd.DataFrame()
        self.symbol = "btcusdt"
        self.interval = "3m"
        self.price_profit = None
        self.price_stoploss = None
        self.position_status = False 
        self.position = None 
        self.enter_price = None
        self.quantity = None
        self.websocket_url = f"wss://fstream.binance.com/ws/{self.symbol}@kline_{self.interval}"
        self.url = 'https://fapi.binance.com/fapi/v1/klines'
        self.params = {
        'symbol': 'btcusdt',
        'interval': '3m',
        'limit': "70"
            }
        self.main_df = self.get_prev_data()
        self.trade = BinanceTrade()
        self.balance = float(self.trade.balance())
        self.win = None
        self.lose = None

    def on_message(self, ws, message):
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
            self.open_position(float(df2['Close']), (self.main_df['Close'].tail(5)).to_list() , self.main_df['Open'].to_list(), self.main_df['High'].to_list(), self.main_df['Low'].to_list())
            #close 리스트는 element 5개
        elif self.position_status == True:
            self.close_position(float(df2['Close']))
        if isClosed == True:
            self.main_df = self.add_frame(df2)
            time.sleep(5)
            # os.system('w32tm /resync')  
        self.balance = float(self.trade.balance())

    def on_close(self, ws):
        print("closed")

    def on_error(self, ws, error):
        print(error)

    def websocket_thread(self):
        ws = wb.WebSocketApp(url=self.websocket_url, on_message= self.on_message, on_close= self.on_close, on_error= self.on_error)
        ws.run_forever()

    def get_prev_data(self) -> pd.DataFrame:
        response = requests.get(self.url, params=self.params)
        response = response.json()
        self.main_df = pd.DataFrame(response, columns =['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime',
                                            'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore']) 
        self.main_df = self.main_df.drop(self.main_df.columns[[6,7,8,9,10,11]], axis=1)
        self.main_df.loc[len(self.main_df)] = pd.Series()
        self.main_df = self.main_df.iloc[:-1]
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

    def EMA(self):
        # ema_fifty = ta.trend.EMAIndicator(df_close, window=50)
        ema_fourteen = ta.trend.EMAIndicator(self.main_df['Close'], window=14)
        ema_eight = ta.trend.EMAIndicator(self.main_df['Close'], window=8)
        # ema_hundred = ta.trend.EMAIndicator(df_close, window=100) 
        # ema_fifty_indicator = ema_fifty.ema_indicator()
        ema_fourteen_indicator = ema_fourteen.ema_indicator()
        ema_eight_indicator = ema_eight.ema_indicator()
        # ema_hundred_indicator = ema_hundred.ema_indicator()
        # ema_fifty_list = (ema_fifty_indicator.tail(5)).tolist()
        ema_fourteen_list = (ema_fourteen_indicator.tail(5)).tolist()
        ema_eight_list = (ema_eight_indicator.tail(5)).tolist()
        # ema_hundred_list = (ema_hundred_indicator.tail(5)).tolist()
        return ema_fourteen_list, ema_eight_list

    def peak_check(self):
        self.main_df['High'] = pd.to_numeric(self.main_df['High'], errors='coerce')
        self.main_df['Low'] = pd.to_numeric(self.main_df['Low'], errors='coerce')
        bhi = ta.volatility.bollinger_hband_indicator(self.main_df['High'], window = 20)
        bli = ta.volatility.bollinger_lband_indicator(self.main_df['Low'], window = 20)
        bhi = np.array(bhi.tail(3).tolist())
        bli = np.array(bli.tail(3).tolist())
        bhi = bhi.astype('int')
        bli = bli.astype('int')
        if 1 in bhi:
            return "nl" #no long
        elif 1 in bli:
            return "ns" #no short
        else:
            return "safe"

    def stochRSI(self):
        df_close = pd.to_numeric(self.main_df['Close'], errors='coerce')
        rsi = ta.momentum.StochRSIIndicator(df_close, window = 14)
        d = rsi.stochrsi_d()
        k = rsi.stochrsi_k()
        d_two = d.tail(2).tolist()
        k_two = k.tail(2).tolist()
        return d_two, k_two

    def decision(self, current_price, close_list, open_list, high_list, low_list):
        self.main_df['High'] = pd.to_numeric(self.main_df['High'], errors='coerce')
        self.main_df['Low'] = pd.to_numeric(self.main_df['Low'], errors='coerce')
        self.main_df['Close'] = pd.to_numeric(self.main_df['Close'], errors='coerce')
        self.main_df['Volume'] = pd.to_numeric(self.main_df['Volume'], errors='coerce')     
        ema_fourteen_list, ema_eight_list = self.EMA()
        two_d, two_k = self.stochRSI()
        # current_atr = ATR(self.main_df)
        prev_d, curr_d = two_d[0], two_d[1]
        prev_k, curr_k = two_k[0], two_k[1]
        prev_open, curr_open = float(open_list[-2]), float(open_list[-1])
        kd_prev_diff, kd_curr_diff = prev_k - prev_d , curr_k - curr_d #양수면 k가 위 음수면 k가 위, if variable > 0 k is above d        

        prev_grad_ema_fourteen = ema_fourteen_list[-2] - ema_fourteen_list[-3]
        curernt_grad_ema_fourteen = ema_fourteen_list[-1] - ema_fourteen_list[-2]    
        prev_grad_ema_eight = ema_eight_list[-2] - ema_eight_list[-3]
        curernt_grad_ema_eight = ema_eight_list[-1] - ema_eight_list[-2]  

        print('__________________________________________________________________________________________________________________________________________________\n'
              f'kd_prev_diff > 0: {kd_prev_diff > 0}\nkd_curr_diff > 0: {kd_curr_diff > 0}\n'
              f'prev_grad_ema_fourteen > 0: {prev_grad_ema_fourteen > 0}\n'
              f'curernt_grad_ema_fourteen > 0 : {curernt_grad_ema_fourteen > 0 }\nprev_grad_ema_eight > 0: {prev_grad_ema_eight > 0}\n'
              f'curernt_grad_ema_eight > 0: {curernt_grad_ema_eight > 0}\n'
              f'current_price > curr_open: {current_price > curr_open}\ndanger_check: {self.danger_check(high_list, low_list)}\n')
            #   f'peak: {self.peak_check()}')

        if ((kd_prev_diff > 0 and kd_curr_diff > 0) and prev_grad_ema_fourteen > 0 and 
            curernt_grad_ema_fourteen > 0 and prev_grad_ema_eight > 0 and curernt_grad_ema_eight > 0 and 
            current_price > curr_open and 
            self.danger_check(high_list, low_list)): #and self.peak_check() != "nl" #prev_open >= prev_ema_fifty and (curr_open >= curr_ema_fifty or current_price > curr_ema_fifty) and prev_open > prev_ema_hundred and curr_open > curr_ema_hundred 
            return "long"
        elif ((kd_prev_diff < 0 and kd_curr_diff < 0) and prev_grad_ema_fourteen < 0 and 
            curernt_grad_ema_fourteen < 0 and prev_grad_ema_eight < 0 and curernt_grad_ema_eight < 0 and 
            current_price < curr_open and 
            self.danger_check(high_list, low_list)): # and self.peak_check() != "ns" # prev_open <= prev_ema_fifty and (curr_open <= curr_ema_fifty or current_price < curr_ema_fifty) and prev_open < prev_ema_hundred and curr_open < curr_ema_hundred 
            return "short"
        else:
            return "pass"

    def danger_check(self, high, low):
        prev_high, prev_low = float(high[-2]), float(low[-2]) 
        prev_percentage_change = (prev_high-prev_low)/prev_low
        if abs(prev_percentage_change)>= 0.01:#0.01 1%
            return False
        else:
            return True
        
    def close_position(self, current_price):
        gain = 40
        loss = 70
        if self.position == "long" and self.position_status:
            if current_price >= self.enter_price + gain or current_price <= self.enter_price - loss:
                self.trade.order("BTCUSDT", "SELL", True, quantity=self.quantity)
                self.position_status = False 
                self.position = None
                time.sleep(30) #분봉보다 쉬면 안됨.
        elif self.position == "short" and self.position_status:
            if current_price <= self.enter_price - gain or current_price >= self.enter_price + loss:
                self.trade.order("BTCUSDT", "BUY", True, quantity=self.quantity)
                self.position = None
                self.position_status = False
                time.sleep(30)


    def open_position(self, current_price, close_list, open_list, high_list, low_list):
        status = self.decision(current_price, close_list, open_list, high_list, low_list)
        self.balance = float(self.trade.balance())
        self.quantity = round(((self.balance * 25) / current_price) * 0.85,3)
        if status == "long" and self.position_status == False: #롱
            self.trade.order("BTCUSDT", "BUY", False, quantity=self.quantity)
            self.long(current_price)
        elif status == "short" and self.position_status == False: #숏
            self.trade.order("BTCUSDT", "SELL", False, quantity=self.quantity)
            self.short(current_price)
        else: #관망 
            pass

    def long(self, current_price):
        self.enter_price = current_price
        self.price_profit = round(self.enter_price + 40, 1)
        self.price_stoploss = round(self.enter_price - 70, 1)
        print(f"익절: {self.price_profit}\n손절: {self.price_stoploss}") 
        self.amount = self.balance/current_price
        self.position = "long"
        self.position_status = True
        print(f"롱: {current_price} (실제 진입 가격은 다를 수 있음)")

    # sleep을하는 순간 프레임이 업데이트가 안됨
    def short(self, current_price):
        self.enter_price = current_price
        self.price_profit = round(self.enter_price - 40, 1)
        self.price_stoploss = round(self.enter_price + 70, 1)
        print(f"익절: {self.price_profit}\n손절: {self.price_stoploss}") 
        self.amount = self.balance/current_price
        self.position = "short"
        self.position_status = True
        print(f"숏: {current_price} (실제 진입 가격은 다를 수 있음)" )    


class BinanceTrade:
    def __init__(self) -> None:
        self.api_key = os.getenv('Bin_API_KEY')
        self.api_secret = os.getenv('Bin_SECRET_KEY')
        self.um_futures_client = UMFutures(key=self.api_key, secret=self.api_secret)
    # get server time

    def change_leverage(self, leverage):
        try:
            response = self.um_futures_client.change_leverage(symbol="BTCUSDT", leverage=leverage)
            logging.info(response)

        except ClientError as error:
            logging.error(f"에러:{error.status_code} 에러코드:{error.error_code} 에러 메세지:{error.error_message}")

    def balance(self):
        try:
            response = self.um_futures_client.balance()
            balance = next(x for x in response if x['asset'] == "USDT")['balance']
            return balance
        except ClientError as error:
            logging.error(f"에러:{error.status_code} 에러코드:{error.error_code} 에러 메세지:{error.error_message}")
    
    def order(self, symbol, side, reduceOnly, quantity): #심볼, BUY SELL, 정리 = true
        try:
            # quantity = (balance*25) / price
            quantity = float(round(quantity, 3))
            stop_price = None
            response = self.um_futures_client.new_order(
                symbol=symbol,
                side=side,
                type= "MARKET",
                quantity=quantity,
                reduceOnly = reduceOnly, #true로 변경하기
            )
            print(response)
        except ClientError as error:
            logging.error(f"에러:{error.status_code} 에러코드:{error.error_code} 에러 메세지:{error.error_message}")

if __name__ == "__main__":
    bot = Data_collector()
    websocket_thread = Thread(target=bot.websocket_thread)
    websocket_thread.start()
