import ta
import numpy as np
import pandas as pd

class TechnicalStrategy:
    @staticmethod
    def EMA(df):
        ema_fourteen = ta.trend.EMAIndicator(df['Close'], window=14)
        ema_eight = ta.trend.EMAIndicator(df['Close'], window=8)
        ema_fourteen_indicator = ema_fourteen.ema_indicator()
        ema_eight_indicator = ema_eight.ema_indicator()
        ema_fourteen_list = (ema_fourteen_indicator.tail(5)).tolist()
        ema_eight_list = (ema_eight_indicator.tail(5)).tolist()
        return ema_fourteen_list, ema_eight_list

    @staticmethod
    def peak_check(df):
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        bhi = ta.volatility.bollinger_hband_indicator(df['High'], window=20)
        bli = ta.volatility.bollinger_lband_indicator(df['Low'], window=20)
        bhi = np.array(bhi.tail(3).tolist())
        bli = np.array(bli.tail(3).tolist())
        bhi = bhi.astype('int')
        bli = bli.astype('int')
        if 1 in bhi:
            return "nl"
        elif 1 in bli:
            return "ns"
        else:
            return "safe"

    @staticmethod
    def stochRSI(df):
        df_close = pd.to_numeric(df['Close'], errors='coerce')
        rsi = ta.momentum.StochRSIIndicator(df_close, window=14)
        d = rsi.stochrsi_d()
        k = rsi.stochrsi_k()
        d_two = d.tail(2).tolist()
        k_two = k.tail(2).tolist()
        return d_two, k_two

    @staticmethod
    def danger_check(high, low):
        prev_high, prev_low = float(high[-2]), float(low[-2]) 
        prev_percentage_change = (prev_high-prev_low)/prev_low
        if abs(prev_percentage_change) >= 0.01:
            return False
        else:
            return True
