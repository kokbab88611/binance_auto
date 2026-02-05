import ta
import numpy as np
import pandas as pd

class TechnicalStrategy:
    """
    Handles technical analysis indicators. 
    Using standard EMA 8/14 crossover strategy combined with StochRSI for momentum.
    """
    
    @staticmethod
    def get_ema_signals(df, close_col='Close'):
        # using EMA 8 and 14 for fast reaction to crypto volatility
        # standard 9/21 might be too slow for 3m timeframe
        close = df[close_col]
        
        ema_fast = ta.trend.EMAIndicator(close, window=8).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(close, window=14).ema_indicator()
        
        # We only need the tail to check for recent crossovers
        return ema_slow.tail(5).tolist(), ema_fast.tail(5).tolist()

    @staticmethod
    def check_volatility_bands(df, high_col='High', low_col='Low'):
        # Bollinger Bands (20, 2)
        # If price touches bands, we might be overextended
        highs = pd.to_numeric(df[high_col], errors='coerce')
        lows = pd.to_numeric(df[low_col], errors='coerce')
        
        bhi = ta.volatility.bollinger_hband_indicator(highs, window=20)
        bli = ta.volatility.bollinger_lband_indicator(lows, window=20)
        
        # Check last 3 candles
        recent_bhi = np.array(bhi.tail(3).tolist()).astype('int')
        recent_bli = np.array(bli.tail(3).tolist()).astype('int')
        
        # Signal filtering
        if 1 in recent_bhi:
            return "no_long"  # Touched upper band, danger of reversal
        elif 1 in recent_bli:
            return "no_short" # Touched lower band, danger of bounce
        return "safe"

    @staticmethod
    def get_momentum(df, close_col='Close'):
        # StochRSI (14) for identifying overbought/oversold conditions
        close = pd.to_numeric(df[close_col], errors='coerce')
        rsi = ta.momentum.StochRSIIndicator(close, window=14)
        
        # D is signal line, K is fast line
        return rsi.stochrsi_d().tail(2).tolist(), rsi.stochrsi_k().tail(2).tolist()

    @staticmethod
    def is_market_safe(high_list, low_list):
        # Pump protection: avoid entering if previous candle was massive (>1%)
        if len(high_list) < 2: return True
            
        prev_high = float(high_list[-2])
        prev_low = float(low_list[-2])
        
        if prev_low == 0: return True 
        
        volatility = (prev_high - prev_low) / prev_low
        
        # 1% move in 3m is huge for BTC, likely chop/manipulation follows
        if abs(volatility) >= 0.01:
            return False
            
        return True