import pandas as pd
import numpy as np
from indicators import Indicator
from collections import deque
import requests

class PatternDetection:
    @staticmethod
    def live_detect_box_pattern(ema_medium, atr, is_closed, atr_multiplier=0.1, box_status = None):
        atr_threshold = atr.mean() * atr_multiplier

        if len(ema_medium) < 2:
            return
            
        ema_diff = abs(ema_medium.iloc[-1] - ema_medium.iloc[-2])
        
        if is_closed:
            if ema_diff < atr_threshold:
                if box_status and (box_status[-1] == 0.5 or box_status[-1] == 1):
                    box_status.append(1)
                else:
                    box_status.append(0.5)
            else:
                if box_status and (box_status[-1] == 0.5):
                    box_status[-1] = 0
                box_status.append(0)
        else:
            if ema_diff < atr_threshold:
                if box_status and (box_status[-1] == 0.5 or box_status[-1] == 1):
                    box_status[-1] = 1
                else:
                    box_status[-1] = 0.5
            else:
                if box_status and box_status[-1] == 0.5:
                    box_status[-1] = 0
        return box_status

    @staticmethod
    def box_pattern_init(main_df, atr_multiplier = 0.05):
        ema_medium = Indicator.EMA(main_df)[1] #1 index에 medium series가 있음
        atr = Indicator.atr(main_df)
        
        # Get the recent 5 ATR mean values
        atr_means = atr.rolling(window=5).mean().dropna().iloc[-5:]
        atr_threshold = atr_means.mean() * atr_multiplier
        # print(f"ATR Threshold: {atr_threshold}")
        
        # Only check the last 3 EMA differences
        ema_diffs = abs(ema_medium.diff().dropna().iloc[-3:])
        # print(f"Last 3 EMA Differences: {ema_diffs.values}, ATR Threshold: {atr_threshold}")
        
        box_status = deque([1 if diff < atr_threshold else 0 for diff in ema_diffs], maxlen=5)
        print(f"Box Status: {box_status}")
        return box_status

    @staticmethod
    def print_previous_box_trends(main_df, num_candles=10):
        print(f"Box trend status for the previous {num_candles} candles:")
        for i in range(-num_candles, 0):
            trend_status = main_df['BoxPattern'].iloc[i]
            time = main_df['openTime'].iloc[i]
            print(f"Time: {time}, Box Trend: {trend_status}")

def get_prev_data() -> pd.DataFrame:
    url = 'https://fapi.binance.com/fapi/v1/klines'
    params = {'symbol': 'BTCUSDT', 'interval': '15m', 'limit': 500}
    response = requests.get(url, params=params).json()
    columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
    df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df[columns]

# Example usage
df = get_prev_data()

# Detect box patterns
box_status = PatternDetection.box_pattern_init(df)
print(box_status)
