import ta
import numpy as np
import pandas as pd

class TechnicalStrategy:
    @staticmethod
    def calculate_ema(df, close_col='Close'):
        close_series = df[close_col]
        
        # Calculate EMAs
        ema_fourteen = ta.trend.EMAIndicator(close_series, window=14)
        ema_eight = ta.trend.EMAIndicator(close_series, window=8)
        
        # Get indicators
        ema_14_indicator = ema_fourteen.ema_indicator()
        ema_8_indicator = ema_eight.ema_indicator()
        
        # Return last 5 values as list
        return ema_14_indicator.tail(5).tolist(), ema_8_indicator.tail(5).tolist()

    @staticmethod
    def check_bollinger_bands(df, high_col='High', low_col='Low'):
        high_series = pd.to_numeric(df[high_col], errors='coerce')
        low_series = pd.to_numeric(df[low_col], errors='coerce')
        
        bhi = ta.volatility.bollinger_hband_indicator(high_series, window=20)
        bli = ta.volatility.bollinger_lband_indicator(low_series, window=20)
        
        bhi_values = np.array(bhi.tail(3).tolist()).astype('int')
        bli_values = np.array(bli.tail(3).tolist()).astype('int')
        
        if 1 in bhi_values:
            return "nl" # no long
        elif 1 in bli_values:
            return "ns" # no short
        else:
            return "safe"

    @staticmethod
    def calculate_stoch_rsi(df, close_col='Close'):
        close_series = pd.to_numeric(df[close_col], errors='coerce')
        rsi = ta.momentum.StochRSIIndicator(close_series, window=14)
        d = rsi.stochrsi_d()
        k = rsi.stochrsi_k()
        
        return d.tail(2).tolist(), k.tail(2).tolist()

    @staticmethod
    def check_danger(high_list, low_list):
        if len(high_list) < 2 or len(low_list) < 2:
            return True
            
        prev_high = float(high_list[-2])
        prev_low = float(low_list[-2])
        
        if prev_low == 0: return True # Avoid division by zero
        
        prev_percentage_change = (prev_high - prev_low) / prev_low
        
        # If change is greater than or equal to 1%, consider it dangerous (volatility check)
        if abs(prev_percentage_change) >= 0.01:
            return False
        else:
            return True
