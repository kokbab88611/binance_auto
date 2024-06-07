import numpy as np
import ta
import pandas as pd
import requests
from threading import Thread
import websocket as wb
import json
from datetime import datetime
import os 
class DataCollector:
    def __init__(self):
        self.leverage = 25
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
        self.results_file = "trade_results.log"
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

    def on_close(self, ws, close_status_code, close_msg):
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

    def bollinger_bands(self):
        bb = ta.volatility.BollingerBands(close=self.main_df['Close'], window=20, window_dev=2)
        bb_middle = bb.bollinger_mavg()
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        return bb_middle, bb_upper, bb_lower

    def stochastic_oscillator(self):
        stoch = ta.momentum.StochasticOscillator(self.main_df['High'], self.main_df['Low'], self.main_df['Close'])
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()
        return stoch_k, stoch_d

    def MACD(self):
        macd = ta.trend.MACD(self.main_df['Close'])
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        return macd_line, signal_line

    def ADX(self):
        adx = ta.trend.ADXIndicator(high=self.main_df['High'], low=self.main_df['Low'], close=self.main_df['Close'], window=14)
        return adx.adx()

    def ichimoku(self):
        ichimoku = ta.trend.IchimokuIndicator(high=self.main_df['High'], low=self.main_df['Low'], window1=9, window2=26, window3=52)
        ichimoku_base = ichimoku.ichimoku_base_line()
        ichimoku_conversion = ichimoku.ichimoku_conversion_line()
        ichimoku_span_a = ichimoku.ichimoku_a()
        ichimoku_span_b = ichimoku.ichimoku_b()
        return ichimoku_base, ichimoku_conversion, ichimoku_span_a, ichimoku_span_b

    def decision(self, current_price):
        ema_short, ema_medium, ema_long = self.EMA()
        rsi = self.RSI()
        vwap = self.VWAP()
        atr = self.ATR().iloc[-1]
        vp, poc = self.volume_profile()
        bb_middle, bb_upper, bb_lower = self.bollinger_bands()
        macd_line, signal_line = self.MACD()
        stoch_k, stoch_d = self.stochastic_oscillator()
        adx = self.ADX()
        ichimoku_base, ichimoku_conversion, ichimoku_span_a, ichimoku_span_b = self.ichimoku()
        volume_threshold = atr * 1.5  # Example threshold, can be adjusted

        # Qualifying conditions
        vwap_qualify = current_price > vwap.iloc[-1]
        ema_short_qualify = ema_short.iloc[-1] > ema_medium.iloc[-2]
        ema_medium_qualify = ema_medium.iloc[-1] > ema_long.iloc[-2]
        rsi_qualify = rsi.iloc[-1] > 40
        volume_qualify = self.buy_volume > volume_threshold
        bb_upper_qualify = current_price < bb_upper.iloc[-1]
        bb_lower_qualify = current_price > bb_lower.iloc[-1]
        macd_qualify = macd_line.iloc[-1] > signal_line.iloc[-1]
        stoch_qualify = stoch_k.iloc[-1] > 20 and stoch_k.iloc[-1] < 80  # Not in extreme overbought or oversold
        adx_qualify = adx.iloc[-1] > 25  # ADX above 25 indicates a strong trend

        # Ichimoku Cloud Conditions
        ichimoku_qualify = (current_price > ichimoku_span_a.iloc[-1] and current_price > ichimoku_span_b.iloc[-1]) or \
                        (current_price < ichimoku_span_a.iloc[-1] and current_price < ichimoku_span_b.iloc[-1])

        # Less Restrictive Volume Profile Condition
        high_volume_node_threshold = vp.mean() * 0.9  # Slightly more restrictive than before
        high_volume_node_qualify = vp.get(current_price, 0) > high_volume_node_threshold

        print("=======================")
        print(f"vwap_qualify = {vwap_qualify}")
        print(f"ema_short = {ema_short_qualify}")
        print(f"ema_medium = {ema_medium_qualify}")
        print(f"rsi = {rsi_qualify}, {rsi.iloc[-1]}")
        print(f"volume_qualify = {volume_qualify}")
        print(f"bb_upper_qualify = {bb_upper_qualify} ({current_price} < {bb_upper.iloc[-1]})")
        print(f"bb_lower_qualify = {bb_lower_qualify} ({current_price} > {bb_lower.iloc[-1]})")
        print(f"macd_qualify = {macd_qualify}")
        print(f"stoch_qualify = {stoch_qualify} ({stoch_k.iloc[-1]})")
        print(f"adx_qualify = {adx_qualify} ({adx.iloc[-1]})")
        print(f"ichimoku_qualify = {ichimoku_qualify}")
        print(f"high_volume_node_qualify = {high_volume_node_qualify} ({vp.get(current_price, 0)}/{high_volume_node_threshold})")
        print("=======================")

        if (vwap_qualify and ema_short_qualify and ema_medium_qualify and rsi_qualify and volume_qualify and 
            bb_upper_qualify and bb_lower_qualify and macd_qualify and stoch_qualify and adx_qualify and 
            ichimoku_qualify and high_volume_node_qualify):
            print("All conditions met for long position.")
            return "long"
        elif (not vwap_qualify and not ema_short_qualify and not ema_medium_qualify and rsi.iloc[-1] < 60 and 
            self.sell_volume > volume_threshold and bb_upper_qualify and bb_lower_qualify and macd_qualify and 
            stoch_qualify and adx_qualify and ichimoku_qualify and high_volume_node_qualify):
            print("All conditions met for short position.")
            return "short"
        else:
            print("Conditions not met for either position.")
            return "pass"

    # def decision(self, current_price):
#     ema_short, ema_medium, ema_long = self.EMA()
#     rsi = self.RSI()
#     vwap = self.VWAP()
#     atr = self.ATR().iloc[-1]
#     vp, poc = self.volume_profile()
#     volume_threshold = atr * 1.5  # Example threshold, can be adjusted

#     # Qualifying conditions
#     vwap_qualify = current_price > vwap.iloc[-1]
#     ema_short_qualify = ema_short.iloc[-1] > ema_medium.iloc[-2]
#     ema_medium_qualify = ema_medium.iloc[-1] > ema_long.iloc[-2]
#     rsi_qualify = rsi.iloc[-1] > 40
#     volume_qualify = self.buy_volume > volume_threshold

#     # Less Stringent EMA Long Condition
#     # Allow a buffer of 2% around POC or price above EMA long
#     buffer_percent = 0.02
#     ema_long_qualify = (ema_long.iloc[-1] > poc * (1 - buffer_percent)) or (current_price > ema_long.iloc[-1])

#     # Simplified Volume Profile Condition
#     high_volume_node_qualify = vp.get(current_price, 0) > vp.mean()

#     print("=======================")
#     print(f"vwap_qualify = {vwap_qualify}")
#     print(f"ema_short = {ema_short_qualify}")
#     print(f"ema_medium = {ema_medium_qualify}")
#     print(f"ema_long = {ema_long_qualify}")
#     print(f"rsi = {rsi.iloc[-1]}")
#     print(f"volume_qualify = {volume_qualify}")
#     print(f"high_volume_node_qualify = {high_volume_node_qualify} ({self.buy_volume}/{volume_threshold})")
#     print("=======================")

#     if vwap_qualify and ema_short_qualify and ema_medium_qualify and ema_long_qualify and rsi_qualify and volume_qualify and high_volume_node_qualify:
#         print("All conditions met for long position.")
#         return "long"
#     elif not vwap_qualify and not ema_short_qualify and not ema_medium_qualify and rsi.iloc[-1] < 60 and self.sell_volume > volume_threshold and high_volume_node_qualify:
#         print("All conditions met for short position.")
#         return "short"
#     else:
#         print("Conditions not met for either position.")
#         return "pass"



    def close_position(self, current_price):
        if self.position_status:
            close_status = False
            fee_percent = None
            result = None

            if self.position == "long":
                if current_price >= self.price_profit or current_price <= self.price_stoploss:
                    fee_percent = 0.0500  # Assuming taker fee for a long position market order
                    result = "profit" if current_price >= self.price_profit else "loss"
                    close_status = True
            elif self.position == "short":
                if current_price <= self.price_profit or current_price >= self.price_stoploss:
                    fee_percent = 0.0500  # Assuming taker fee for a short position market order
                    result = "profit" if current_price <= self.price_profit else "loss"
                    close_status = True
            else:
                print("Position not defined or invalid position type")
                return  # Exit if the position type is neither long nor short

            if close_status:
                effective_fee_percent = fee_percent * self.leverage / 100
                fee_paid = (self.enter_price * effective_fee_percent) + \
                           (current_price * effective_fee_percent)

                profit_loss_amount = (current_price - self.enter_price) if self.position == "long" else \
                                     (self.enter_price - current_price)

                profit_loss_percent = ((profit_loss_amount - fee_paid) / self.enter_price) * 100

                log_message = f"Closed {self.position} position at {current_price} with {result}. " \
                              f"Profit/Loss: ({profit_loss_percent:.2f}%), "
                self.save_result(log_message)
                print(log_message)

                self.position_status = False
                self.position = None

    def save_result(self, message):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{current_time} - {message}"
        with open(self.results_file, "a") as file:
            file.write(log_message + "\n")

    def open_position(self, current_price):
        if not self.position_status:
            status = self.decision(current_price)
            if status != "pass":
                if status == "long":
                    self.long(current_price)
                elif status == "short":
                    self.short(current_price)

    def set_atr_based_sl_tp(self, entry_price, atr):
        leverage = 25
        fee_percent = 0.0005  # 0.05% as a decimal
        min_profit = 0.01  # 1% minimum profit

        # Calculate the effective fee per transaction considering leverage
        effective_fee_per_transaction = fee_percent * leverage
        # Total fee for both entry and exit
        total_fee = 2 * effective_fee_per_transaction
        # Total required return to ensure minimum profit after fees
        required_return = total_fee + min_profit

        # Calculate initial ATR-based stop-loss and take-profit
        stop_loss_price = entry_price - (atr * 1.2)
        atr_based_tp = entry_price + (atr * 1.7)

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

        self.save_result(f"Opened short position at {current_price}")
        print(f"Opened short position at {current_price}")
        print(f"Target Profit Price: {self.price_profit}")
        print(f"Stop Loss Price: {self.price_stoploss}")

if __name__ == "__main__":
    bot = DataCollector()
    websocket_thread_vol = Thread(target=bot.websocket_thread_vol)
    websocket_thread_vol.start()

    websocket_thread_kline = Thread(target=bot.websocket_thread_kline)
    websocket_thread_kline.start()
