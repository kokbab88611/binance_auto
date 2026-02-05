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
        self.client = BinanceClient()
        self.strategy = TechnicalStrategy()
        self.ml_model = PredictionModel()
        
        # State variables
        self.position_status = False 
        self.position = None 
        self.enter_price = None
        self.quantity = None
        self.ai_result = None
        self.balance = 0.0
        
        # Initialize Data
        self.get_initial_data()
        self.balance = self.client.get_balance()

    def get_initial_data(self):
        """Fetch historical klines to populate the dataframe"""
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
            
            # Drop unnecessary columns
            self.main_df = self.main_df.drop(self.main_df.columns[[6,7,8,9,10,11]], axis=1)
            
            # Convert types
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.main_df[cols] = self.main_df[cols].apply(pd.to_numeric, errors='coerce')
            
            # Remove last incomplete candle
            self.main_df = self.main_df.iloc[:-1]
            
            # Add category
            self.main_df['Category'] = np.where(
                self.main_df['Close'] > self.main_df['Open'], 'bullish', 'bearish'
            )
            logging.info("Initial data loaded successfully.")
            
        except Exception as e:
            logging.error(f"Error fetching initial data: {e}")

    def start_socket(self):
        """Start WebSocket connection"""
        ws = wb.WebSocketApp(
            url=config.WS_URL, 
            on_message=self.on_message, 
            on_close=self.on_close, 
            on_error=self.on_error
        )
        ws.run_forever()

    def on_close(self, ws):
        logging.info("WebSocket connection closed")

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        data = json.loads(message)
        kline = data['k']
        
        # Parse data
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
        
        # Update balance
        self.balance = self.client.get_balance()
        
        # Live processing
        if not self.position_status:
            # Predict and check entry
            self.ai_result = self.ml_model.predict_trend(self.main_df)
            self.check_entry(current_price)
        else:
            # Check exit
            self.check_exit(current_price)

        # Update DataFrame when candle closes
        if is_closed:
            self.update_dataframe(new_row)

    def update_dataframe(self, new_row):
        """Append new candle and maintain fixed size"""
        self.main_df.loc[len(self.main_df)] = new_row
        if len(self.main_df) > 80:
            self.main_df = self.main_df.iloc[1:].reset_index(drop=True)

    def check_entry(self, current_price):
        """Determine if we should enter a position"""
        # Calculate Indicators
        ema_14, ema_8 = self.strategy.calculate_ema(self.main_df)
        d, k = self.strategy.calculate_stoch_rsi(self.main_df)
        
        # Safe checks
        if len(ema_14) < 3 or len(k) < 2: return
        
        # Indicator Logic
        prev_d, curr_d = d[-2], d[-1]
        prev_k, curr_k = k[-2], k[-1]
        
        kd_prev_diff = prev_k - prev_d
        kd_curr_diff = curr_k - curr_d
        
        # Gradient checks
        ema_14_grad = ema_14[-1] - ema_14[-2]
        ema_8_grad = ema_8[-1] - ema_8[-2]
        
        highs = self.main_df['High'].tolist()
        lows = self.main_df['Low'].tolist()
        opens = self.main_df['Open'].tolist()
        curr_open = opens[-1]
        
        is_safe = self.strategy.check_danger(highs, lows)
        
        # Entry Logic
        if (kd_prev_diff > 0 and kd_curr_diff > 0 and 
            ema_14_grad > 0 and ema_8_grad > 0 and 
            current_price > curr_open and is_safe and 
            self.ai_result == "bullish"):
            
            self.open_position("long", current_price)
            
        elif (kd_prev_diff < 0 and kd_curr_diff < 0 and 
              ema_14_grad < 0 and ema_8_grad < 0 and 
              current_price < curr_open and is_safe and 
              self.ai_result == "bearish"):
              
            self.open_position("short", current_price)

    def open_position(self, direction, price):
        """Execute entry order"""
        self.quantity = round(((self.balance * config.LEVERAGE) / price) * config.QUANTITY_MULTIPLIER, 3)
        side = "BUY" if direction == "long" else "SELL"
        
        logging.info(f"Opening {direction} position at {price}")
        response = self.client.place_order(config.SYMBOL, side, False, self.quantity)
        
        if response:
            self.position = direction
            self.position_status = True
            self.enter_price = price

    def check_exit(self, current_price):
        """Check stop loss and take profit"""
        gain = 40
        loss = 70
        
        should_close = False
        side = ""
        
        if self.position == "long":
            if current_price >= self.enter_price + gain or current_price <= self.enter_price - loss:
                should_close = True
                side = "SELL"
        elif self.position == "short":
            if current_price <= self.enter_price - gain or current_price >= self.enter_price + loss:
                should_close = True
                side = "BUY"
                
        if should_close:
            logging.info(f"Closing {self.position} position at {current_price}")
            self.client.place_order(config.SYMBOL, side, True, self.quantity)
            self.position = None
            self.position_status = False
            self.enter_price = None
            time.sleep(10) # Cooldown
