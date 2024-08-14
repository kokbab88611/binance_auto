import ta
import pandas as pd
from indicators import Indicator
import requests
import websocket as wb
import json
import numpy as np
import time
from ressup import SupportResistanceLevels as SRL

class Strategy:
    @staticmethod
    def check_trade_signal(df_5m, df_15m, df_1h, binancetrade):
        current_price = df_5m['close'].iat[-1]
        atr_15m = Indicator.atr(df_15m).iat[-1]
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
        ema_cross_long = ema_9.iat[-1] > ema_15.iat[-1] and ema_9.iat[-2] > ema_15.iat[-2]
        ema_cross_short = ema_9.iat[-1] < ema_15.iat[-1] and ema_9.iat[-2] < ema_15.iat[-2]

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

        quantity, balance = binancetrade.calculate_quantity(current_price)

        # Trade signal
        if long_condition:
            binancetrade.market_open_position(side="BUY", position_side="LONG", calced_quantity=quantity)
            binancetrade.order_sl_tp(balance, current_price, "long" , atr_15m, quantity, atr_multiplier_tp=1.6, atr_multiplier_sl=1.35)
            print("Enter Trend Long Position")
            return True
        elif short_condition:
            binancetrade.market_open_position(side="SELL", position_side="SHORT", calced_quantity=quantity)
            binancetrade.order_sl_tp(balance, current_price, "short" , atr_15m, quantity, atr_multiplier_tp=1.6, atr_multiplier_sl=1.35)
            print("Enter Trend Short Position")
            return True
        else:
            # print("No Trade Signal")
            return False

    @staticmethod
    def box_trend_strategy(df_5m, df_15m, df_30m, binancetrade):
        resistance, support = SRL.identify_levels(df_15m)
        resistance_levels_filtered = SRL.remove_anomalies(resistance, resistance.iat[-1])
        support_levels_filtered = SRL.remove_anomalies(support, support.iat[-1])
        resistance_mean = resistance_levels_filtered.mean()
        support_mean = support_levels_filtered.mean()
        
        current_price = df_5m['close'].iat[-1]
        recent_30_min = df_30m[:-1].tail(4)

        # Get the minimum of the 'low' column and the maximum of the 'high' column
        recent_low_30 = recent_30_min['low'].min()
        recent_high_30 = recent_30_min['high'].max()
        # Calculate the 9 EMA and 15 EMA
        ema_9 = Indicator.EMA(df_15m, length=9)
        ema_15 = Indicator.EMA(df_15m, length=15)
        # Calculate ATR for 5m
        atr_5m = Indicator.atr(df_5m).iat[-1]
        atr_15m = Indicator.atr(df_15m).iat[-1]
        
        # 박스 돌파권
        if (current_price > resistance_mean + atr_5m) and (current_price > recent_high_30):
            #롱 돌파
            position_opened = Strategy.box_breakout('long', df_5m, df_15m, atr_15m, current_price, binancetrade)
            return position_opened
        elif (current_price < support_mean - atr_5m) and (current_price < recent_low_30):
            #숏 돌파
            position_opened = Strategy.box_breakout('short', df_5m, df_15m, atr_15m, current_price, binancetrade)
            return position_opened

        # 박스권 롱
        if support_mean <= current_price <= resistance_mean:
            if ema_9.iat[-1] > ema_15.iat[-1] and ema_9.iat[-2] <= ema_15.iat[-2]:
                print("Enter BOX Long Position")
                quantity, balance = binancetrade.calculate_quantity(current_price)
                binancetrade.market_open_position(side="BUY", position_side="LONG", calced_quantity=quantity)
                binancetrade.order_sl_tp(balance, current_price, "long" ,atr_15m, quantity, atr_multiplier_tp=1.3, atr_multiplier_sl=1.1)
                return True

        # 박스권 숏
        if resistance_mean >= current_price >= support_mean:
            if ema_9.iat[-1] < ema_15.iat[-1] and ema_9.iat[-2] >= ema_15.iat[-2]:
                print("Enter BOX short Position")
                quantity, balance = binancetrade.calculate_quantity(current_price)
                binancetrade.market_open_position(side="SELL", position_side="SHORT", calced_quantity=quantity)
                binancetrade.order_sl_tp(balance, current_price, "short" ,atr_15m, quantity, atr_multiplier_tp=1.3, atr_multiplier_sl=1.1)
                return True
        return False

    @staticmethod
    def box_breakout(trend, df_5m, df_15m, atr_15m, current_price, binancetrade):
        # Calculate the 9 EMA and 15 EMA
        ema_9 = Indicator.EMA(df_15m, length=9)
        ema_15 = Indicator.EMA(df_15m, length=15)
        balance = binancetrade.fetch_balance()
        # Determine if 9 EMA is crossing above or below 15 EMA
        ema_cross_long = ema_9.iat[-1] > ema_15.iat[-1] and ema_9.iat[-2] <= ema_15.iat[-2]
        ema_cross_short = ema_9.iat[-1] < ema_15.iat[-1] and ema_9.iat[-2] >= ema_15.iat[-2]

        # Confirm the trend direction based on EMA cross
        if trend == 'long' and ema_cross_long:
            print("Enter BOX BREAK LONG Position")
            quantity, balance = binancetrade.calculate_quantity(current_price)
            binancetrade.market_open_position(side="BUY", position_side="LONG", calced_quantity=quantity)
            binancetrade.order_sl_tp(balance, current_price, "long", atr_15m, quantity, atr_multiplier_tp=1.3, atr_multiplier_sl=1.1)
            return True
        elif trend == 'short' and ema_cross_short:
            print("Enter BOX BREAK SHORT Position")
            quantity, balance = binancetrade.calculate_quantity(current_price)           
            binancetrade.market_open_position(side="SELL", position_side="SHORT", calced_quantity=quantity)
            binancetrade.order_sl_tp(balance, current_price, "short", atr_15m, quantity, atr_multiplier_tp=1.3, atr_multiplier_sl=1.1)
            return True
        else:
            return False

