import numpy as np
import ta
import pandas as pd
import requests
from threading import Thread
import websocket as wb
import json
from datetime import datetime
import os 
from smartmoneyconcepts import smc
from binance.um_futures import UMFutures
from binance.error import ClientError
import time

class DataCollector:
    def __init__(self):
        self.leverage = 20
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
        self.balance = None
        self.smc_df = pd.DataFrame(index=self.main_df.index)
        self.trade = BinanceTrade()

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
        df2 = {'openTime': openTime, 'open': Open, 'high': High, 'low': Low, 'close': Close, 'volume': Volume}
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
        params = {'symbol': self.symbol, 'interval': self.interval, 'limit': 500}
        response = requests.get(url, params=params).json()
        df = pd.DataFrame(response, columns=['openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
        df = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
        df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
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
        ema_short = ta.trend.EMAIndicator(self.main_df['close'], window=9).ema_indicator()
        ema_medium = ta.trend.EMAIndicator(self.main_df['close'], window=21).ema_indicator()
        ema_long = ta.trend.EMAIndicator(self.main_df['close'], window=50).ema_indicator()
        return ema_short, ema_medium, ema_long

    def apply_smc_indicators(self):
        fvg = smc.fvg(self.main_df)
        swing_highs_lows = smc.swing_highs_lows(self.main_df)
        bos_choch = smc.bos_choch(self.main_df, swing_highs_lows)
        ob = smc.ob(self.main_df, swing_highs_lows)
        liquidity = smc.liquidity(self.main_df, swing_highs_lows)
        self.smc_df['swing_highs_lows'] = swing_highs_lows['HighLow']
        self.smc_df['swing_levels'] = swing_highs_lows['Level']

    def RSI(self):
        return ta.momentum.RSIIndicator(self.main_df['close'], window=14).rsi()

    def VWAP(self):
        return ta.volume.VolumeWeightedAveragePrice(self.main_df['high'], self.main_df['low'], self.main_df['close'], self.main_df['volume']).volume_weighted_average_price()

    def ATR(self):
        return ta.volatility.AverageTrueRange(self.main_df['high'], self.main_df['low'], self.main_df['close']).average_true_range()

    def volume_profile(self):
        vp = self.main_df.groupby('close')['volume'].sum()
        poc = vp.idxmax()
        return vp, poc

    def bollinger_bands(self):
        bb = ta.volatility.BollingerBands(close=self.main_df['close'], window=20, window_dev=2)
        bb_middle = bb.bollinger_mavg()
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        return bb_middle, bb_upper, bb_lower

    def stochastic_oscillator(self):
        stoch = ta.momentum.StochasticOscillator(self.main_df['high'], self.main_df['low'], self.main_df['close'])
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()
        return stoch_k, stoch_d

    def ichimoku(self):
        ichimoku = ta.trend.IchimokuIndicator(high=self.main_df['high'], low=self.main_df['low'], window1=9, window2=26, window3=52)
        ichimoku_base = ichimoku.ichimoku_base_line()
        ichimoku_conversion = ichimoku.ichimoku_conversion_line()
        ichimoku_span_a = ichimoku.ichimoku_a()
        ichimoku_span_b = ichimoku.ichimoku_b()
        return ichimoku_base, ichimoku_conversion, ichimoku_span_a, ichimoku_span_b

    def check_uptrend(self, ema_short, ema_medium, ema_long):
        uptrend = (ema_short.iloc[-1] > ema_medium.iloc[-1] > ema_long.iloc[-1]) and (ema_short.iloc[-2] > ema_medium.iloc[-2] > ema_long.iloc[-2])
        return uptrend

    def check_rsi_trend(self, rsi):
        rsi_uptrend = rsi.iloc[-1] > rsi.iloc[-2] and rsi.iloc[-2] > rsi.iloc[-3]
        rsi_downtrend = rsi.iloc[-1] < rsi.iloc[-2] and rsi.iloc[-2] < rsi.iloc[-3]
        return rsi_uptrend, rsi_downtrend

    def decision(self, current_price):
        self.apply_smc_indicators()

        rsi = self.RSI()
        vwap = self.VWAP()
        atr = self.ATR().iloc[-1]
        stoch_k, stoch_d = self.stochastic_oscillator()
        ichimoku_base, ichimoku_conversion, ichimoku_span_a, ichimoku_span_b = self.ichimoku()
        volume_threshold = atr * 1.2  # Example threshold, can be adjusted

        # Qualifying conditions
        vwap_qualify = current_price > vwap.iloc[-1]
        rsi_uptrend, rsi_downtrend = self.check_rsi_trend(rsi)

        rsi_qualify = rsi.iloc[-1] > 40
        bb_upper_qualify = current_price < self.bollinger_bands()[1].iloc[-1]
        bb_lower_qualify = current_price > self.bollinger_bands()[2].iloc[-1]
        stoch_qualify = stoch_k.iloc[-1] > 20 and stoch_k.iloc[-1] < 80  # Not in extreme overbought or oversold

        # Ichimoku Cloud Conditions
        ichimoku_qualify = (current_price > ichimoku_span_a.iloc[-1] and current_price > ichimoku_span_b.iloc[-1]) or \
                        (current_price < ichimoku_span_a.iloc[-1] and current_price < ichimoku_span_b.iloc[-1])

        # Swing High/Low Condition
        # swing_high_low_condition = self.smc_df['swing_highs_lows'].iloc[-1] == 1

        # New Volume Ratio Condition
        volume_ratio_qualify = self.buy_volume > self.sell_volume

        # New condition for high volatility surges
        high_volatility_surge_long = current_price > self.bollinger_bands()[1].iloc[-1] and current_price > (self.main_df['close'].iloc[-1] + atr * 1.5)
        high_volatility_surge_short = current_price < self.bollinger_bands()[2].iloc[-1] and current_price < (self.main_df['close'].iloc[-1] - atr * 1.5)

        # New Candle Comparison Condition
        previous_close = self.main_df['close'].iloc[-2]
        current_close = self.main_df['close'].iloc[-1]
        candle_comparison_long = current_close > previous_close
        candle_comparison_short = current_close < previous_close

        long_safe = [
            rsi.iloc[-1] > 40,
            self.buy_volume > volume_threshold,
            bb_lower_qualify,
            (bb_upper_qualify or high_volatility_surge_long),
            stoch_qualify,
            rsi_uptrend,
            ichimoku_qualify,
            volume_ratio_qualify,
            candle_comparison_long,
            # swing_high_low_condition
        ]

        short_safe = [
            rsi.iloc[-1] < 60,
            self.sell_volume > volume_threshold,
            bb_upper_qualify,
            (bb_lower_qualify or high_volatility_surge_short),
            stoch_qualify,
            rsi_downtrend,
            ichimoku_qualify,
            not volume_ratio_qualify,
            candle_comparison_short,
            # not swing_high_low_condition,
        ] 

        if all(long_safe):
            print("All conditions met for long position.")
            return "long"
        elif all(short_safe):
            print("All conditions met for short position.")
            return "short"
        else:
            return "pass"

    def close_position(self, current_price):
        if self.position_status:
            close_status = False
            result = None

            if self.position == "long":
                if current_price >= self.price_profit or current_price <= self.price_stoploss:
                    result = "profit" if current_price >= self.price_profit else "loss"
                    close_status = True 
                    percent = ((current_price - self.enter_price) / self.enter_price) * self.leverage * 100
            elif self.position == "short":
                if current_price <= self.price_profit or current_price >= self.price_stoploss:
                    result = "profit" if current_price <= self.price_profit else "loss"
                    close_status = True
                    percent = ((self.enter_price - current_price) / self.enter_price) * self.leverage * 100
            else:
                print("Position not defined or invalid position type")
                return

            if close_status:
                log_message = f"Closed {self.position} position at {current_price} with {result}. " \
                            f"{result}: ({percent:.2f}%)"
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
                    self.trade.long(current_price)
                elif status == "short":
                    self.trade.short(current_price)

class BinanceTrade:
    def __init__(self):
        self.api_key = os.getenv('Bin_API_KEY')
        self.api_secret = os.getenv('Bin_SECRET_KEY')
        self.client = UMFutures(key=self.api_key, secret=self.api_secret)
        self.symbol = "BTCUSDT"
        self.leverage = 20
        self.exchange_info = self.client.exchange_info()
        self.symbol_info = self.get_symbol_info(self.symbol.upper())

    def get_symbol_info(self, symbol):
        for s in self.exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s
        return None

    def set_atr_based_sl_tp(self, entry_price, atr, position):
        profit_percentage = 0.000709879 # gain profit from this percentage
        long_profit_percentage = 1.000709879 
        short_profit_percentage = 0.999290121
        long_minimum_tp = entry_price * long_profit_percentage
        short_minimum_tp = entry_price * short_profit_percentage
        if atr > 80:
            atr = 80
        # Total required return to ensure minimum profit after fees
        if position == "long":
            minimum_profit_tp = entry_price * (1 + profit_percentage) 
            stop_loss_price = entry_price - (atr * 1.5)
            atr_based_tp = entry_price + (atr * 1.85)
            if atr_based_tp < long_minimum_tp:
                minimum_profit_tp = entry_price * 1.001116977
        # Adjust take-profit to ensure at least 1% profit after fees
        if position == "short":
            minimum_profit_tp = entry_price * (1 - profit_percentage) 
            stop_loss_price = entry_price + (atr * 1.5)
            atr_based_tp = entry_price - (atr * 1.85)
            if atr_based_tp > short_minimum_tp:
                minimum_profit_tp = entry_price * 0.9988830227

        take_profit_price = max(atr_based_tp, minimum_profit_tp)

        return take_profit_price, stop_loss_price

    def fetch_balance(self):
        try:
            response = self.client.balance()
            balance = next(x for x in response if x['asset'] == "USDT")['balance']
            available_balance = next(x for x in response if x['asset'] == "USDT")['availableBalance']
            print(f"Total Balance: {balance}")
            print(f"Available Balance: {available_balance}")
            return float(balance), float(available_balance)
        except ClientError as e:
            print(f"Error fetching balance: {e}")
            return None, None

    def sync_time(self):
        os.system('w32tm /resync')

    def calculate_quantity(self, available_balance, price):
        max_quantity = round((((available_balance * (1 - (self.leverage * 0.005))) * self.leverage) / price) * 0.9 ,3)
        return round(max_quantity, 3)

    def set_leverage(self):
        try:
            response = self.client.change_leverage(symbol=self.symbol.upper(), leverage=self.leverage)
            print(f"Leverage set to {response['leverage']}")
        except ClientError as e:
            print(f"Error setting leverage: {e}")

    # def check_margin_requirements(self, position_notional, bid_order_value, ask_order_value):
    #     margin_required = self.calculate_margin(position_notional, bid_order_value, ask_order_value)
    #     _ , available_balance = self.fetch_balance()
    #     return available_balance >= margin_required

    def order(self, symbol, side, position_side, quantity, order_type="MARKET", stop_price=None, close_position=False):

        params = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,
            "quantity": quantity,
            "type": order_type,
            "timestamp": int(time.time() * 1000),
        }
        print('================================================================')
        print(stop_price)
        print(quantity)
        print('================================================================')
        if order_type != "MARKET":
            params = {
                "symbol": symbol,
                "side": side,
                "positionSide": position_side,
                "quantity": quantity,
                "type": order_type,
                "timestamp": int(time.time() * 1000),
                "closePosition": True,
                "stopPrice": str(stop_price)
            }

        response = self.client.new_order(**params)
        print(f"Order placed: {response}")
        return response

    def long(self, current_price):
        self.set_leverage()
        balance, available_balance = self.fetch_balance()
        if available_balance is None or available_balance <= 0:
            print("Insufficient available balance to place order")
            return

        calced_quantity = self.calculate_quantity(available_balance, current_price) 
        in_atr = DataCollector().ATR()
        in_atr = round(in_atr.iloc[-1], 2)
        enter_price = current_price
        price_profit, price_stoploss = self.set_atr_based_sl_tp(enter_price, in_atr, "long")

        self.order(symbol=self.symbol.upper(), side="BUY", position_side="LONG", quantity=calced_quantity)
        self.order(symbol=self.symbol.upper(), side="SELL", position_side="LONG", quantity=calced_quantity, order_type="TAKE_PROFIT", stop_price=price_profit, close_position=True)
        self.order(symbol=self.symbol.upper(), side="SELL", position_side="LONG", quantity=calced_quantity, order_type="STOP", stop_price=price_stoploss, close_position=True)

        DataCollector().save_result(f"Opened long position at {current_price}")
        print(f"Opened long position at {current_price}, Target Profit Price: {price_profit}, Stop Loss Price: {price_stoploss}")

    def short(self, current_price):
        self.set_leverage()
        balance, available_balance = self.fetch_balance()
        if available_balance is None or available_balance <= 0:
            print("Insufficient available balance to place order")
            return

        calced_quantity = self.calculate_quantity(available_balance, current_price)
        in_atr = DataCollector().ATR()
        in_atr = round(in_atr.iloc[-1], 2)
        enter_price = current_price
        price_profit, price_stoploss = self.set_atr_based_sl_tp(enter_price, in_atr, "short")

        # position_notional = enter_price * self.quantity
        # bid_order_value = self.quantity * price_stoploss
        # ask_order_value = self.quantity * price_profit

        # if not self.check_margin_requirements(position_notional, bid_order_value, ask_order_value):
        #     print("Insufficient margin to place short order")
        #     return

        self.order(symbol=self.symbol.upper(), side="SELL", position_side="SHORT", quantity=calced_quantity)
        self.order(symbol=self.symbol.upper(), side="BUY", position_side="SHORT", quantity=calced_quantity, order_type="TAKE_PROFIT", stop_price=price_profit, close_position=True)
        self.order(symbol=self.symbol.upper(), side="BUY", position_side="SHORT", quantity=calced_quantity, order_type="STOP", stop_price=price_stoploss, close_position=True)

        DataCollector().save_result(f"Opened short position at {current_price}")
        print(f"Opened short position at {current_price}, Target Profit Price: {price_profit}, Stop Loss Price: {price_stoploss}")

if __name__ == "__main__":
    bot = DataCollector()
    websocket_thread_vol = Thread(target=bot.websocket_thread_vol)
    websocket_thread_vol.start()

    websocket_thread_kline = Thread(target=bot.websocket_thread_kline)
    websocket_thread_kline.start()
