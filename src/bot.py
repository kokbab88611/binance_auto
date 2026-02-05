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
from src.ml_models import TrendPredictor

class TradingBot:
    def __init__(self):
        self.main_df = pd.DataFrame()
        self.client = BinanceClient()
        self.strategy = TechnicalStrategy()
        self.ml_model = TrendPredictor()
        
        # State variables
        self.position_status = False 
        self.position = None 
        self.enter_price = None
        self.quantity = None
        self.trend_signal = None
        self.balance = 0.0
        
        # Load historical context for indicators
        self.hydrate_data()
        self.balance = self.client.get_balance()

    def hydrate_data(self):
        """Fetch historical klines to pre-populate indicators"""
        try:
            params = {
                'symbol': config.SYMBOL.lower(),
                'interval': config.INTERVAL,
                'limit': config.LIMIT
            }
            response = requests.get(config.API_URL, params=params)
            data = response.json()
            
            self.main_df = pd.DataFrame(data, columns=[
                'openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'
            ])
            
            # Clean up response
            self.main_df = self.main_df.drop(self.main_df.columns[[6,7,8,9,10,11]], axis=1)
            
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.main_df[cols] = self.main_df[cols].apply(pd.to_numeric, errors='coerce')
            
            # Remove last incomplete candle so we don't calc on noise
            self.main_df = self.main_df.iloc[:-1]
            
            # Label data for ML training
            self.main_df['Category'] = np.where(
                self.main_df['Close'] > self.main_df['Open'], 'bullish', 'bearish'
            )
            logging.info(f"Hydrated {len(self.main_df)} candles for analysis")
            
        except Exception as e:
            logging.error(f"Failed to load history: {e}")

    def start_socket(self):
        # Using run_forever blocking call for simplicity
        ws = wb.WebSocketApp(
            url=config.WS_URL, 
            on_message=self.on_message, 
            on_close=self.on_close, 
            on_error=self.on_error
        )
        ws.run_forever()

    def on_close(self, ws):
        logging.warning("WebSocket connection dropped. Attempting reconnect...")

    def on_error(self, ws, error):
        logging.error(f"WebSocket Error: {error}")

    def on_message(self, ws, message):
        data = json.loads(message)
        kline = data['k']
        
        new_row = {
            'openTime': kline['t'],
            'Open': float(kline['o']),
            'High': float(kline['h']),
            'Low': float(kline['l']),
            'Close': float(kline['c']),
            'Volume': float(kline['v']),
            'Category': 'bullish' if float(kline['c']) > float(kline['o']) else 'bearish'
        }
        
        is_closed = kline['x']
        current_price = new_row['Close']
        
        # Update balance periodically?
        # self.balance = self.client.get_balance() 
        
        # Main Trading Loop
        if not self.position_status:
            # Re-train model on latest data? 
            # might be too slow for every tick, maybe move to candle close
            self.trend_signal = self.ml_model.predict(self.main_df)
            self.evaluate_entry_conditions(current_price)
        else:
            self.evaluate_exit_conditions(current_price)

        if is_closed:
            self.append_candle(new_row)

    def append_candle(self, new_row):
        """Maintain fixed window size for memory efficiency"""
        self.main_df.loc[len(self.main_df)] = new_row
        if len(self.main_df) > 80:
            self.main_df = self.main_df.iloc[1:].reset_index(drop=True)

    def evaluate_entry_conditions(self, current_price):
        # Calculate Technicals
        ema_slow, ema_fast = self.strategy.get_ema_signals(self.main_df)
        d, k = self.strategy.get_momentum(self.main_df)
        
        # Need context
        if len(ema_slow) < 3 or len(k) < 2: return
        
        # Check crossovers
        kd_prev_diff = k[-2] - d[-2]
        kd_curr_diff = k[-1] - d[-1]
        
        # Check trend direction (gradient)
        ema_slow_grad = ema_slow[-1] - ema_slow[-2]
        ema_fast_grad = ema_fast[-1] - ema_fast[-2]
        
        curr_open = self.main_df['Open'].tolist()[-1]
        
        # Market safety checks
        highs = self.main_df['High'].tolist()
        lows = self.main_df['Low'].tolist()
        is_safe = self.strategy.is_market_safe(highs, lows)
        
        # LOGIC: Momentum + Trend + ML Confirmation
        if (kd_prev_diff > 0 and kd_curr_diff > 0 and 
            ema_slow_grad > 0 and ema_fast_grad > 0 and 
            current_price > curr_open and is_safe and 
            self.trend_signal == "bullish"):
            
            self.execute_trade("long", current_price)
            
        elif (kd_prev_diff < 0 and kd_curr_diff < 0 and 
              ema_slow_grad < 0 and ema_fast_grad < 0 and 
              current_price < curr_open and is_safe and 
              self.trend_signal == "bearish"):
              
            self.execute_trade("short", current_price)

    def execute_trade(self, direction, price):
        self.quantity = round(((self.balance * config.LEVERAGE) / price) * config.QUANTITY_MULTIPLIER, 3)
        side = "BUY" if direction == "long" else "SELL"
        
        logging.info(f"Signal detected: {direction} at {price}")
        response = self.client.place_order(config.SYMBOL, side, False, self.quantity)
        
        if response:
            self.position = direction
            self.position_status = True
            self.enter_price = price

    def evaluate_exit_conditions(self, current_price):
        # Hardcoded TP/SL for now
        # TODO: Implement dynamic ATR-based stops
        tp_dist = 40
        sl_dist = 70
        
        should_close = False
        side = ""
        
        if self.position == "long":
            if current_price >= self.enter_price + tp_dist:
                logging.info("Take Profit hit")
                should_close = True
                side = "SELL"
            elif current_price <= self.enter_price - sl_dist:
                logging.info("Stop Loss hit")
                should_close = True
                side = "SELL"
                
        elif self.position == "short":
            if current_price <= self.enter_price - tp_dist:
                logging.info("Take Profit hit")
                should_close = True
                side = "BUY"
            elif current_price >= self.enter_price + sl_dist:
                logging.info("Stop Loss hit")
                should_close = True
                side = "BUY"
                
        if should_close:
            self.client.place_order(config.SYMBOL, side, True, self.quantity)
            self.position = None
            self.position_status = False
            self.enter_price = None
            time.sleep(10) # Cooldown to prevent double-entry on same candle
