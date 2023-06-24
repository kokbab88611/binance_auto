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

# https://www.google.com/search?q=use+graphics+card+to+run+python&oq=use+graphics+card+to+run+python&aqs=chrome..69i57j33i160l2.7968j0j4&sourceid=chrome&ie=UTF-8
class Data_collector:
    def __init__ (self):
        self.balance = 100
        self.commission = 0.01 #25배 레버리지 수수료 . 기본 수수료는 0.0004
        self.main_df = pd.DataFrame()
        self.symbol = "btcusdt"
        self.interval = "5m"
        self.in_atr = None
        self.price_profit = None
        self.price_stoploss = None
        self.prev_trade = None
        self.win = 0 
        self.lose = 0
        self.position_status = False # 포지션 잡고 있는지 true, false
        self.position = None #숏인지 롱인지 
        self.enter_price = None
        self.amount = None
        self.websocket_url = f"wss://fstream.binance.com/ws/{self.symbol}@kline_{self.interval}"
        self.url = 'https://fapi.binance.com/fapi/v1/klines'
        self.params = {
        'symbol': 'BTCUSDT',
        'interval': '5m',
        'limit': "230"
            }
        self.main_df = self.get_prev_data()

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
        # print((self.main_df['Close'].tail(10)).tolist())
        if self.position_status == False:
            self.open_position(float(df2['Close']), self.main_df['Close'].to_list(), self.main_df['Open'].to_list(), self.main_df['High'].to_list(), self.main_df['Low'].to_list())
        elif self.position_status == True:
            self.close_position(float(df2['Close']))
        if isClosed == True:
            self.main_df = self.add_frame(df2)
            time.sleep(10) # 새 분봉에서 10초 관망   

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
        return self.main_df

    def add_frame(self, df2):
        self.main_df.loc[len(self.main_df)] = df2
        return self.main_df

    def live_edit(self, df2): #dataframe은 row 250개 이하로 고정한다. 만약 row가 250개를 넘으면 30개를 지우고 index를 초기화 시킨다.   
        df2 = list(df2.values())
        self.main_df.iloc[-1] = df2
        length_df = len(self.main_df)
        if length_df == 250:
            self.main_df = self.main_df.drop(self.main_df.index[:30])
            self.main_df = self.main_df.reset_index(drop=True)

    def EMA(self, df_close):
        ema_fifty = ta.trend.EMAIndicator(df_close, window=50)
        ema_fourteen = ta.trend.EMAIndicator(df_close, window=14)
        ema_eight = ta.trend.EMAIndicator(df_close, window=8)
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
        bhi = ta.volatility.bollinger_hband_indicator(self.main_df['High'], window = 30)
        bli = ta.volatility.bollinger_lband_indicator(self.main_df['Low'], window = 30)
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

    def ATR(self):
        self.main_df['High'] = pd.to_numeric(self.main_df['High'], errors='coerce')
        self.main_df['Low'] = pd.to_numeric(self.main_df['Low'], errors='coerce')
        self.main_df['Close'] = pd.to_numeric(self.main_df['Close'], errors='coerce')
        df_high = self.main_df['High']
        df_low = self.main_df['Low']
        df_close = self.main_df['Close']
        pre_atr = ta.volatility.AverageTrueRange(df_high, df_low, df_close)
        atr = pre_atr.average_true_range() 
        return atr

    def decision(self, current_price, close_list, open_list, high_list, low_list):
        df_close = pd.Series(close_list)
        ema_fourteen_list, ema_eight_list = self.EMA(df_close)
        two_d, two_k = self.stochRSI()
        # current_atr = ATR(self.main_df)
        prev_d, curr_d = two_d[0], two_d[1]
        prev_k, curr_k = two_k[0], two_k[1]
        prev_open, curr_open = float(open_list[-2]), float(open_list[-1])
        kd_prev_diff, kd_curr_diff = prev_k - prev_d , curr_k - curr_d #양수면 k가 위 음수면 k가 위, if variable > 0 k is above d
        curr_k_zero, curr_d_zero = curr_k == 0, curr_d == 0
        curr_k_hund, curr_d_hund = curr_k == 100, curr_d == 100  
        prev_k_zero, prev_d_zero = prev_k == 0, prev_d == 0
        prev_k_hund, prev_d_hund = prev_k == 100, prev_d == 100            
        # prev_ema_fifty, curr_ema_fifty = ema_fifty_list[-2], ema_fifty_list[-1]  
        # prev_ema_hundred, curr_ema_hundred = ema_hundred_list[-2], ema_hundred_list[-1]
        # prev_grad_ema_fifty = ema_fifty_list[-2] - ema_fifty_list[-3]         #양수면 ema상승, 음수면 ema하락 if varaible >0 ema is going up
        # curernt_grad_ema_fifty = ema_fifty_list[-1] - ema_fifty_list[-2] 
        prev_grad_ema_fourteen = ema_fourteen_list[-2] - ema_fourteen_list[-3]
        curernt_grad_ema_fourteen = ema_fourteen_list[-1] - ema_fourteen_list[-2]    
        prev_grad_ema_eight = ema_eight_list[-2] - ema_eight_list[-3]
        curernt_grad_ema_eight = ema_eight_list[-1] - ema_eight_list[-2]  

        # danger = danger_check(high, low)
        print('__________________________________________________________________________________________________________________________________________________\n'
              f'kd_prev_diff > 0: {kd_prev_diff > 0}\nkd_curr_diff > 0: {kd_curr_diff > 0}\n'
              f'prev_grad_ema_fourteen > 0: {prev_grad_ema_fourteen > 0}\n'
              f'curernt_grad_ema_fourteen > 0 : {curernt_grad_ema_fourteen > 0 }\nprev_grad_ema_eight > 0: {prev_grad_ema_eight > 0}\n'
              f'curernt_grad_ema_eight > 0: {curernt_grad_ema_eight > 0}\n'
              f'current_price > curr_open: {current_price > curr_open}\ndanger_check: {self.danger_check(high_list, low_list)}\n'
              f'peak: {self.peak_check()}\nkd short: {(kd_prev_diff < 0 and kd_curr_diff < 0) or (curr_k_zero and curr_d_zero and prev_d_zero and prev_d_zero)}\n'
              f'kd long: {(kd_prev_diff > 0 and kd_curr_diff > 0) or (curr_k_hund and curr_d_hund and prev_d_hund and prev_d_hund)}')
        if (((kd_prev_diff > 0 and kd_curr_diff > 0) or (curr_k_hund and curr_d_hund and prev_d_hund and prev_d_hund)) and prev_grad_ema_fourteen > 0 and 
            curernt_grad_ema_fourteen > 0 and prev_grad_ema_eight > 0 and curernt_grad_ema_eight > 0 and 
            current_price > curr_open and self.danger_check(high_list, low_list) and self.peak_check() != "nl"): #prev_open >= prev_ema_fifty and (curr_open >= curr_ema_fifty or current_price > curr_ema_fifty) and prev_open > prev_ema_hundred and curr_open > curr_ema_hundred 
            return "long"
        elif (((kd_prev_diff < 0 and kd_curr_diff < 0) or (curr_k_zero and curr_d_zero and prev_d_zero and prev_d_zero)) and prev_grad_ema_fourteen > 0 and 
            curernt_grad_ema_fourteen < 0 and prev_grad_ema_eight < 0 and curernt_grad_ema_eight < 0 and 
            current_price < curr_open and self.danger_check(high_list, low_list) and self.peak_check() != "ns"): # prev_open <= prev_ema_fifty and (curr_open <= curr_ema_fifty or current_price < curr_ema_fifty) and prev_open < prev_ema_hundred and curr_open < curr_ema_hundred 
            return "short"
        else:
            return "pass"

    def danger_check(self, high, low):
        prev_high, prev_low = float(high[-2]), float(low[-2]) 
        prev_percentage_change = (prev_high-prev_low)/prev_low
        # if abs(prev_percentage_change)>= 0.0025:
        #     print("***WARNING***\n이전 3분봉이 0.25% 이상이였습니다")
        #     return False
        if abs(prev_percentage_change)>= 0.01:#0.01 1%
            # print("***WARNING***\n이전 3분봉이 1% 이상이였습니다")
            return False
        else:
            return True
        
    def close_position(self, current_price):
        gain = self.in_atr * 2.5
        loss = self.in_atr * 3
        if self.position == "long" and self.position_status == True:
            if current_price >= self.enter_price + gain or current_price <= self.enter_price - loss:
                enter_value = self.enter_price * self.amount
                percentage = (current_price - self.enter_price) / self.enter_price
                percentage = (percentage * 25) + 1
                self.balance = enter_value * percentage
                prev_balance = self.balance - (enter_value * self.amount) 
                self.amount = 0
                if (percentage-1)*100 < 0:
                    self.lose += 1
                elif (percentage-1)*100 > 0: 
                    self.win += 1 
                print(f"clear position at {current_price} with {(percentage-1)*100}% profit({prev_balance})\nNew balance is {self.balance}\nWin: {self.win}, Lose: {self.lose}")
                print("sleep")
                self.position_status = False 
                self.position = None
                time.sleep(70) #분봉보다 쉬면 안됨.
        elif self.position == "short" and self.position_status == True:
            if current_price <= self.enter_price - gain or current_price >= self.enter_price + loss:
                enter_value = self.enter_price * self.amount
                percentage = (current_price - self.enter_price) / self.enter_price
                percentage = 1 - (percentage * 25)
                self.balance = enter_value * percentage
                prev_balance = self.balance - (enter_value * self.amount) 
                self.amount = 0    
                if (percentage-1)*100 < 0:
                    self.lose += 1
                elif (percentage-1)*100 > 0: 
                    self.win += 1 
                print(f"clear position at {current_price} with {(percentage-1)*100}% profit({prev_balance})\nNew balance is {self.balance}\nWin: {self.win}, Lose: {self.lose}")
                print("sleep")
                self.position = None
                self.position_status = False
                time.sleep(70)
                # market() #binance close position function

    def open_position(self, current_price, close_list, open_list, high_list, low_list):
        status = self.decision(current_price,close_list, open_list, high_list, low_list)
        print(status)
        if status == "long" and self.position_status == False: #롱
            print(self.balance)
            self.long(current_price)
            self.position_status = True
        elif status == "short" and self.position_status == False: #숏
            print(self.balance)
            self.short(current_price)
            self.position_status = True
        else: #관망 
            pass

    # Create and start the WebSocket thread
    def long(self, current_price):
        self.in_atr = self.ATR()
        self.in_atr = round(self.in_atr.iloc[-1], 2)   
        if self.in_atr > 50:
            self.in_atr = 50
        else:
            pass
        self.enter_price = current_price
        self.balance = self.balance * (1 - self.commission)
        self.amount = self.balance/current_price
        self.balance = 0
        self.position = "long"
        print(f"long: {current_price}")
        print(self.in_atr)

    # sleep을하는 순간 프레임이 업데이트가 안됨
    def short(self, current_price):
        self.in_atr = self.ATR()
        self.in_atr = round(self.in_atr.iloc[-1], 2)  
        if self.in_atr > 50:
            self.in_atr = 50
        else:
            pass
        self.enter_price = current_price
        self.balance = self.balance * (1 - self.commission)
        self.amount = self.balance/current_price
        self.enter_price = current_price
        self.balance = 0
        self.position = "short"
        print(f"short: {current_price}" )    
        print(self.in_atr)

if __name__ == "__main__":
    bot = Data_collector()
    websocket_thread = Thread(target=bot.websocket_thread)
    websocket_thread.start()