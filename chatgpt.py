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
from binance.cm_futures import CMFutures
from binance.error import ClientError
import time

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
        self.balance = None
        self.smc_df = pd.DataFrame(index=self.main_df.index)

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

        # Ensure the returned data structures match expected lengths

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
        # Check if the current and previous values of shorter EMAs are greater than the next longer EMAs
        uptrend = (ema_short.iloc[-1] > ema_medium.iloc[-1] > ema_long.iloc[-1]) and \
                (ema_short.iloc[-2] > ema_medium.iloc[-2] > ema_long.iloc[-2])

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
        swing_high_low_condition = self.smc_df['swing_highs_lows'].iloc[-1] == 1

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

        # Conditions for Long Position
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
            swing_high_low_condition
        ]
    
        # Conditions for Short Position
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
            not swing_high_low_condition,
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
                return  # Exit if the position type is neither long nor short

            if close_status:
                # Format the log message to include both profit/loss percentage and amount
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
                    self.long(current_price)
                elif status == "short":
                    self.short(current_price)

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
            atr_based_tp = entry_price + (atr * 1.8)
            if atr_based_tp < long_minimum_tp:
                minimum_profit_tp = entry_price * 1.001116977
        # Adjust take-profit to ensure at least 1% profit after fees
        if position == "short":
            minimum_profit_tp = entry_price * (1 - profit_percentage) 
            stop_loss_price = entry_price + (atr * 1.5)
            atr_based_tp = entry_price - (atr * 1.8)
            if atr_based_tp > short_minimum_tp:
                minimum_profit_tp = entry_price * 0.9988830227

        take_profit_price = max(atr_based_tp, minimum_profit_tp)

        return take_profit_price, stop_loss_price

    def long(self, current_price):
        self.in_atr = self.ATR()
        self.in_atr = round(self.in_atr.iloc[-1], 2)
        self.enter_price = current_price
        self.price_profit, self.price_stoploss = self.set_atr_based_sl_tp(self.enter_price, self.in_atr, "long")
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
        self.price_profit, self.price_stoploss = self.set_atr_based_sl_tp(self.enter_price, self.in_atr, "short")
        self.position = "short"
        self.position_status = True

        self.save_result(f"Opened short position at {current_price}")
        print(f"Opened short position at {current_price}")
        print(f"Target Profit Price: {self.price_profit}")
        print(f"Stop Loss Price: {self.price_stoploss}")
class BinanceTrade:
    def __init__(self):
        self.api_key = os.getenv('Bin_API_KEY')
        self.api_secret = os.getenv('Bin_SECRET_KEY')
        self.um_futures_client = UMFutures(key=self.api_key, secret=self.api_secret)
        self.symbol = "btcusdt"
        self.quantity = 0.001  # Example quantity, adjust as necessary
        self.leverage = 25
        self.exchange_info = self.client.futures_exchange_info()
        self.symbol_info = self.get_symbol_info(self.symbol.upper())

    def get_symbol_info(self, symbol):
        for s in self.exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s
        return None

    def validate_order(self, price, quantity, order_type):
        if not self.symbol_info:
            raise ValueError("Symbol information not found.")

        # Validate price against PRICE_FILTER
        price_filter = next(f for f in self.symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')
        if price_filter:
            min_price = float(price_filter['minPrice'])
            max_price = float(price_filter['maxPrice'])
            tick_size = float(price_filter['tickSize'])
            if not (min_price <= price <= max_price):
                raise ValueError(f"Price must be between {min_price} and {max_price}.")
            if (price - min_price) % tick_size != 0:
                raise ValueError(f"Price must be a multiple of {tick_size}.")

        # Validate quantity against LOT_SIZE
        lot_size = next(f for f in self.symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
        if lot_size:
            min_qty = float(lot_size['minQty'])
            max_qty = float(lot_size['maxQty'])
            step_size = float(lot_size['stepSize'])
            if not (min_qty <= quantity <= max_qty):
                raise ValueError(f"Quantity must be between {min_qty} and {max_qty}.")
            if (quantity - min_qty) % step_size != 0:
                raise ValueError(f"Quantity must be a multiple of {step_size}.")

        # Validate quantity against MARKET_LOT_SIZE for market orders
        if order_type == "MARKET":
            market_lot_size = next(f for f in self.symbol_info['filters'] if f['filterType'] == 'MARKET_LOT_SIZE')
            if market_lot_size:
                min_qty = float(market_lot_size['minQty'])
                max_qty = float(market_lot_size['maxQty'])
                step_size = float(market_lot_size['stepSize'])
                if not (min_qty <= quantity <= max_qty):
                    raise ValueError(f"Market order quantity must be between {min_qty} and {max_qty}.")
                if (quantity - min_qty) % step_size != 0:
                    raise ValueError(f"Market order quantity must be a multiple of {step_size}.")

    def fetch_balance(self):
        try:
            response = self.client.futures_account_balance()
            balance = next(x for x in response if x['asset'] == "USDT")['balance']
            available_balance = next(x for x in response if x['asset'] == "USDT")['withdrawAvailable']
            print(f"Total Balance: {balance}")
            print(f"Available Balance: {available_balance}")
            return float(balance), float(available_balance)
        except ClientError as e:
            print(f"Error fetching balance: {e}")
            return None, None

    def calculate_quantity(self, available_balance, price):
        # Calculate the maximum quantity based on available balance and leverage
        max_quantity = (available_balance * self.leverage) / price
        return round(max_quantity, 3)  # Adjust the rounding as needed

    def set_leverage(self):
        try:
            response = self.client.futures_change_leverage(symbol=self.symbol.upper(), leverage=self.leverage)
            print(f"Leverage set to {response['leverage']}")
        except ClientError as e:
            print(f"Error setting leverage: {e}")

    def order(self, symbol, side, position_side, quantity, order_type="MARKET", price=None, stop_price=None):
        """
        Place an order on Binance Futures.

        :param symbol: str - The symbol to trade, e.g., 'BTCUSDT'
        :param side: str - 'BUY' or 'SELL'
        :param position_side: str - 'LONG' or 'SHORT'
        :param quantity: float - The amount of the asset to trade
        :param order_type: str - 'LIMIT', 'STOP_MARKET', or 'MARKET'
        :param price: float - The price at which to execute a LIMIT order
        :param stop_price: float - The stop price for a STOP_MARKET order
        :return: dict - Details of the order placed or None if an error occurred
        """
        try:
            self.validate_order(price if price else 0, quantity, order_type)

            params = {
                "symbol": symbol,
                "side": side,
                "positionSide": position_side,
                "quantity": quantity,
                "type": order_type,
                "timestamp": int(time.time() * 1000)
            }
            if order_type == "LIMIT":
                params.update({
                    "price": str(price),
                    "timeInForce": "GTC"
                })
            elif order_type in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
                params.update({
                    "stopPrice": str(stop_price)
                })

            response = self.client.futures_create_order(**params)
            print(f"Order placed: {response}")
            return response
        except ClientError as e:
            print(f"API error placing order: {e}")
            return None

    def long(self, current_price):
        self.set_leverage()
        balance, available_balance = self.fetch_balance()
        if available_balance is None or available_balance <= 0:
            print("Insufficient available balance to place order")
            return

        self.quantity = self.calculate_quantity(available_balance, current_price)
        self.in_atr = self.ATR()
        self.in_atr = round(self.in_atr.iloc[-1], 2)
        self.enter_price = current_price
        self.price_profit, self.price_stoploss = self.set_atr_based_sl_tp(self.enter_price, self.in_atr, "long")
        self.position = "long"
        self.position_status = True

        # Market order to open the position
        self.order(symbol=self.symbol.upper(), side="BUY", position_side="LONG", quantity=self.quantity)

        # Limit order for take profit
        self.order(symbol=self.symbol.upper(), side="SELL", position_side="LONG", quantity=self.quantity, order_type="LIMIT", price=self.price_profit)

        # Stop market order for stop loss
        self.order(symbol=self.symbol.upper(), side="SELL", position_side="LONG", quantity=self.quantity, order_type="STOP_MARKET", stop_price=self.price_stoploss)

        self.save_result(f"Opened long position at {current_price}")
        print(f"Opened long position at {current_price}, Target Profit Price: {self.price_profit}, Stop Loss Price: {self.price_stoploss}")

    def short(self, current_price):
        self.set_leverage()
        balance, available_balance = self.fetch_balance()
        if available_balance is None or available_balance <= 0:
            print("Insufficient available balance to place order")
            return

        self.quantity = self.calculate_quantity(available_balance, current_price)
        self.in_atr = self.ATR()
        self.in_atr = round(self.in_atr.iloc[-1], 2)
        self.enter_price = current_price
        self.price_profit, self.price_stoploss = self.set_atr_based_sl_tp(self.enter_price, self.in_atr, "short")
        self.position = "short"
        self.position_status = True

        # Market order to open the position
        self.order(symbol=self.symbol.upper(), side="SELL", position_side="SHORT", quantity=self.quantity)

        # Limit order for take profit
        self.order(symbol=self.symbol.upper(), side="BUY", position_side="SHORT", quantity=self.quantity, order_type="LIMIT", price=self.price_profit)

        # Stop market order for stop loss
        self.order(symbol=self.symbol.upper(), side="BUY", position_side="SHORT", quantity=self.quantity, order_type="STOP_MARKET", stop_price=self.price_stoploss)

        self.save_result(f"Opened short position at {current_price}")
        print(f"Opened short position at {current_price}, Target Profit Price: {self.price_profit}, Stop Loss Price: {self.price_stoploss}")

    def save_result(self, message):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{current_time} - {message}"
        with open("trade_results.log", "a") as file:
            file.write(log_message + "\n")

if __name__ == "__main__":
    bot = DataCollector()
    websocket_thread_vol = Thread(target=bot.websocket_thread_vol)
    websocket_thread_vol.start()

    websocket_thread_kline = Thread(target=bot.websocket_thread_kline)
    websocket_thread_kline.start()
