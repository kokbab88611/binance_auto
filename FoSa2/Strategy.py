import ta
import pandas as pd
from indicators import Indicator
import requests
import websocket as wb
import json
import numpy as np
from ressup import SupportResistanceLevels as SRL

class Strategy:
    @staticmethod
    def check_trade_signal(df_5m, df_1h):
        # Calculate indicators
        vwap = Indicator.vwap(df_5m).iat[-1]
        bb_upper, bb_lower = Indicator.bollinger_bands(df_5m)

        # Calculate Stochastic RSI and RSI for 1h data
        latest_stoch_d, latest_stoch_k = Indicator.stochastic_rsi(df_1h)
        rsi = Indicator.rsi(df_1h).iat[-1]
        rsi_prev = Indicator.rsi(df_1h).iat[-2]

        # Determine if VWAP is within Bollinger Bands
        is_vwap_within_bb = bb_lower <= vwap <= bb_upper

        # Check Stochastic RSI condition
        is_stoch_rsi_long = latest_stoch_k > latest_stoch_d
        is_stoch_rsi_short = latest_stoch_k < latest_stoch_d
        
        ema_9, ema_15 = Indicator.EMA(df_5m, 9), Indicator.EMA(df_5m, 15)
        ema_cross_long = ema_9.iat[-1] > ema_15.iat[-1] and ema_9.iat[-2] <= ema_15.iat[-2]
        ema_cross_short = ema_9.iat[-1] < ema_15.iat[-1] and ema_9.iat[-2] >= ema_15.iat[-2]

        # Check RSI trend
        is_rsi_up = rsi > rsi_prev
        is_rsi_down = rsi < rsi_prev

        # Default scenarios if none provided
        long_scenarios = [
            is_vwap_within_bb and is_stoch_rsi_long and is_rsi_up and ema_cross_long
        ]
    
        short_scenarios = [
            is_vwap_within_bb and is_stoch_rsi_short and is_rsi_down and ema_cross_short
        ]

        # Evaluate scenarios
        long_condition = any(long_scenarios)
        short_condition = any(short_scenarios)

        # Trade signal
        if long_condition:
            print("Enter Long Position")
        elif short_condition:
            print("Enter Short Position")
        else:
            print("No Trade Signal")

    @staticmethod
    def box_trend_strategy(df_5m, df_15m, df_30m):
        resistance, support = SRL.identify_levels(df_15m)
        resistance_levels_filtered = SRL.remove_anomalies(resistance, resistance.iat[-1])
        support_levels_filtered = SRL.remove_anomalies(support, support.iat[-1])
        
        resistance_mean = resistance_levels_filtered.mean()
        support_mean = support_levels_filtered.mean()
        
        current_close_price = df_5m['close'].iat[-1]

        recent_30_min = df_30m[:-1].tail(4)

        # Get the minimum of the 'low' column and the maximum of the 'high' column
        recent_low_30 = recent_30_min['low'].min()
        recent_high_30 = recent_30_min['high'].max()

        # Calculate the 9 EMA and 15 EMA
        ema_9 = Indicator.EMA(df_5m['close'], window=9)
        ema_15 = Indicator.EMA(df_5m['close'], window=15)

        # Calculate ATR for 5m
        atr_5m = Indicator.atr(df_5m).iat[-1]

        # 박스 돌파권
        if (current_close_price > resistance_mean + atr_5m) and (current_close_price > recent_high_30):
            #롱 돌파
            Strategy.box_breakout('long', df_5m)
        elif (current_close_price < support_mean - atr_5m) and (current_close_price < recent_low_30):
            #숏 돌파
            Strategy.box_breakout('short', df_5m)

        # 박스권 롱
        if support_mean <= current_close_price <= resistance_mean:
            if ema_9.iat[-1] > ema_15.iat[-1] and ema_9.iat[-2] <= ema_15.iat[-2]:
                print(f"Long entry at {current_close_price}. EMA 9 crossed above EMA 15.")
                # Logic for placing a long order

        # 박스권 숏
        if resistance_mean >= current_close_price >= support_mean:
            if ema_9.iat[-1] < ema_15.iat[-1] and ema_9.iat[-2] >= ema_15.iat[-2]:
                print(f"Short entry at {current_close_price}. EMA 9 crossed below EMA 15.")
                # Logic for placing a short order

    @staticmethod
    def box_breakout(trend, df_5m):
        # Calculate the 9 EMA, 15 EMA, and 200 EMA
        ema_9 = Indicator.EMA(df_5m['close'], window=9)
        ema_15 = Indicator.EMA(df_5m['close'], window=15)
        ema_200 = Indicator.EMA(df_5m['close'], window=200)

        # Determine the current direction of the 200 EMA
        ema_200_direction = ema_200.iat[-1] - ema_200.iat[-2]  # Check the slope of the 200 EMA

        # Confirm the trend direction based on 200 EMA
        if trend == 'long' and ema_200_direction > 0:
            print(f"Confirmed long breakout with 200 EMA trending upwards at {df_5m['close'].iat[-1]}.")
            # Logic for placing a long order
        elif trend == 'short' and ema_200_direction < 0:
            print(f"Confirmed short breakout with 200 EMA trending downwards at {df_5m['close'].iat[-1]}.")
            # Logic for placing a short order
        else:
            print(f"No valid breakout signal. 200 EMA is not aligned with the trend direction at {df_5m['close'].iat[-1]}.")

