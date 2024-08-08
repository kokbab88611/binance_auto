import ta
import pandas as pd
from indicators import Indicator
import requests
import websocket as wb
import json
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
    def box_trend_strategy(df_15m, df_5m):
        resistance, support = SRL.identify_levels(df_15m)
        resistance_levels_filtered = SRL.remove_anomalies(resistance, resistance.iat[-1])
        support_levels_filtered = SRL.remove_anomalies(support, support.iat[-1])
        
        resistance_mean = resistance_levels_filtered.mean()
        support_mean = support_levels_filtered.mean()
        pass