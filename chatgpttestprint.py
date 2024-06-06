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

    def on_close(self, ws):
        print("WebSocket closed")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

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
        df = df.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'float'})
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

        # Qualifying conditions
        vwap_qualify = current_price > vwap.iloc[-1]
        ema_short_qualify = ema_short.iloc[-1] > ema_medium.iloc[-2]
        ema_medium_qualify = ema_medium.iloc[-1] > ema_long.iloc[-2]
        ema_long_qualify = ema_long.iloc[-1] > poc
        rsi_qualify = rsi.iloc[-1] > 40
        volume_qualify = self.buy_volume > volume_threshold

        # Volume Profile Condition: Check if current price is near a high-volume node
        high_volume_node_qualify = vp.get(current_price, 0) > (vp.mean() + vp.std())

        print("=======================")
        print(f"vwap_qualify = {vwap_qualify}")
        print(f"ema_short = {ema_short_qualify}")
        print(f"ema_medium = {ema_medium_qualify}")
        print(f"ema_long = {ema_long_qualify}")
        print(f"rsi = {rsi_qualify}, {rsi.iloc[-1]}")
        print(f"volume_qualify = {volume_qualify}")
        print(f"high_volume_node_qualify = {high_volume_node_qualify}")
        print(poc)
        print("=======================")

        if vwap_qualify and ema_short_qualify and ema_medium_qualify and ema_long_qualify and rsi_qualify and volume_qualify and high_volume_node_qualify:
            print("All conditions met for long position.")
            return "long"
        elif not vwap_qualify and not ema_short_qualify and not ema_medium_qualify and rsi.iloc[-1] < 60 and self.sell_volume > volume_threshold and high_volume_node_qualify:
            print("All conditions met for short position.")
            return "short"
        else:
            print("Conditions not met for either position.")
            return "pass"

    def close_position(self, current_price):
        if self.position_status:
            if self.position == "long":
                if current_price >= self.price_profit or current_price <= self.price_stoploss:
                    self.trade.order(symbol=self.symbol.upper(), side="SELL", quantity=self.quantity, reduce_only=True)
                    result = "profit" if current_price >= self.price_profit else "loss"
                    profit_loss_percent = ((current_price - self.enter_price) / self.enter_price) * 100
                    self.position_status = False
                    self.save_result(f"Closed long position at {current_price} with {result} ({profit_loss_percent:.2f}%)")
                    print(f"Closed long position at {current_price} with {result} ({profit_loss_percent:.2f}%)")
                    self.position = None

            elif self.position == "short":
                if current_price <= self.price_profit or current_price >= self.price_stoploss:
                    self.trade.order(symbol=self.symbol.upper(), side="BUY", quantity=self.quantity, reduce_only=True)
                    result = "profit" if current_price <= self.price_profit else "loss"
                    profit_loss_percent = ((self.enter_price - current_price) / self.enter_price) * 100
                    self.position_status = False
                    self.save_result(f"Closed short position at {current_price} with {result} ({profit_loss_percent:.2f}%)")
                    print(f"Closed short position at {current_price} with {result} ({profit_loss_percent:.2f}%)")
                    self.position = None

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

    def set_atr_based_sl_tp(self, entry_price, atr):
        leverage = 25
        fee_percent = 0.0500
        min_profit = 0.01
        
        effective_fee_per_transaction = (fee_percent * leverage) / 100
        total_fee = 2 * effective_fee_per_transaction
        required_return = total_fee + min_profit

        # Calculate initial ATR-based stop-loss and take-profit
        stop_loss_price = entry_price - (atr * 1.2)
        atr_based_tp = entry_price + (atr * 1.75)

        # Adjust take-profit to ensure at least 1% profit after fees
        minimum_profit_tp = entry_price * (1 + required_return)
        take_profit_price = max(atr_based_tp, minimum_profit_tp)

        return take_profit_price, stop_loss_price

    def long(self, current_price):
        self.in_atr = self.ATR()
        self.in_atr = round(self.in_atr.iloc[-1], 2)
        self.enter_price = current_price
        self.price_profit, self.price_stoploss = self.set_atr_based_sl_tp(self.enter_price, self.in_atr)
        self.position = "long"
        self.position_status = True
        self.trade.order(symbol=self.symbol.upper(), side="BUY", quantity=self.quantity)
        self.save_result(f"Opened long position at {current_price}")
        print(f"Opened long position at {current_price}")
        print(f"Target Profit Price: {self.price_profit}")
        print(f"Stop Loss Price: {self.price_stoploss}")

    def short(self, current_price):
        self.in_atr = self.ATR()
        self.in_atr = round(self.in_atr.iloc[-1], 2)
        self.enter_price = current_price
        self.price_profit, self.price_stoploss = self.set_atr_based_sl_tp(self.enter_price, self.in_atr)
        self.position = "short"
        self.position_status = True
        self.trade.order(symbol=self.symbol.upper(), side="SELL", quantity=self.quantity)
        self.save_result(f"Opened short position at {current_price}")
        print(f"Opened short position at {current_price}")
        print(f"Target Profit Price: {self.price_profit}")
        print(f"Stop Loss Price: {self.price_stoploss}")

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
            print(f"Order placed: {order}")
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
