import ta
import pandas as pd

class Indicator:

    def EMA(self):
        ema_short = ta.trend.EMAIndicator(self.main_df['close'], window=9).ema_indicator()
        ema_medium = ta.trend.EMAIndicator(self.main_df['close'], window=21).ema_indicator()
        ema_long = ta.trend.EMAIndicator(self.main_df['close'], window=50).ema_indicator()
        return ema_short, ema_medium, ema_long

    def RSI(self):
        return ta.momentum.RSIIndicator(self.main_df['close'], window=14).rsi()

    def ATR(self):
        return ta.volatility.AverageTrueRange(self.main_df['high'], self.main_df['low'], self.main_df['close']).average_true_range()

    def bollinger_bands(self):
        bb = ta.volatility.BollingerBands(close=self.main_df['close'], window=20, window_dev=2)
        bb_middle = bb.bollinger_mavg()
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        return bb_middle, bb_upper, bb_lower

    def stochastic_oscillator(self):
        stoch = ta.momentum.StochasticOscillator(self.main_df['high'], self.main_df['low'], self.main_df['close'])
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()
        return stoch_k, stoch_d

    def ichimoku(self):
        ichimoku = ta.trend.IchimokuIndicator(high=self.main_df['high'], low=self.main_df['low'], window1=9, window2=26, window3=52)
        ichimoku_base = ichimoku.ichimoku_base_line()
        ichimoku_conversion = ichimoku.ichimoku_conversion_line()
        ichimoku_span_a = ichimoku.ichimoku_a()
        ichimoku_span_b = ichimoku.ichimoku_b()
        return ichimoku_base, ichimoku_conversion, ichimoku_span_a, ichimoku_span_b

    def check_uptrend(self, ema_short, ema_medium, ema_long):
        uptrend = (ema_short.iloc[-1] > ema_medium.iloc[-1] > ema_long.iloc[-1]) and (ema_short.iloc[-2] > ema_medium.iloc[-2] > ema_long.iloc[-2])
        return uptrend

    def check_rsi_trend(self, rsi):
        rsi_uptrend = rsi.iloc[-1] > rsi.iloc[-2] and rsi.iloc[-2] > rsi.iloc[-3]
        rsi_downtrend = rsi.iloc[-1] < rsi.iloc[-2] and rsi.iloc[-2] < rsi.iloc[-3]
        return rsi_uptrend, rsi_downtrend