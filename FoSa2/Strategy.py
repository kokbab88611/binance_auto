import ta
import pandas as pd
import indicators
import requests
import websocket as wb
import json

class Strategy:
    def __init__(self, symbol):
        self.symbol = symbol

    def check_trade_signal(self, df_5m):
        # Calculate indicators
        vwap = indicators.vwap(df_5m).iat[-1]
        bb_upper, bb_lower = indicators.bollinger_bands(df_5m)

        # Fetch 1h data for Stochastic RSI
        df_1h = self.fetch_data('1h')
        latest_stoch_d, latest_stoch_k = indicators.stochastic_rsi(df_1h)
        rsi = indicators.rsi(df_1h).iat[-1]
        rsi_prev = indicators.rsi(df_1h).iat[-2]

        # Determine if VWAP is within Bollinger Bands
        is_vwap_within_bb = bb_lower <= vwap <= bb_upper

        # Check Stochastic RSI condition
        is_stoch_rsi_long = latest_stoch_k > latest_stoch_d
        is_stoch_rsi_short = latest_stoch_k < latest_stoch_d

        # Check RSI trend
        is_rsi_up = rsi > rsi_prev
        is_rsi_down = rsi < rsi_prev

        # Combine conditions
        long_condition = is_vwap_within_bb and is_stoch_rsi_long and is_rsi_up
        short_condition = is_vwap_within_bb and is_stoch_rsi_short and is_rsi_down

        # Trade signal
        if long_condition:
            print("Enter Long Position")
        elif short_condition:
            print("Enter Short Position")
        else:
            print("No Trade Signal")