import ta
import pandas as pd
import requests
import matplotlib.pyplot as plt

class Indicator:
    def EMA(main_df):
        ema_short = ta.trend.EMAIndicator(main_df['close'], window=9).ema_indicator()
        ema_medium = ta.trend.EMAIndicator(main_df['close'], window=21).ema_indicator()
        ema_long = ta.trend.EMAIndicator(main_df['close'], window=100).ema_indicator()
        return ema_short, ema_medium, ema_long

    def vwap(main_df):
        return ta.volume.VolumeWeightedAveragePrice(main_df['high'], main_df['low'], main_df['close'], main_df['volume'], window = 100).volume_weighted_average_price()

    def stochastic_rsi(main_df):
        """
        Calculate the latest Stochastic RSI %D and %K values.

        Args:
            main_df (pd.DataFrame): DataFrame containing 'close' prices.

        Returns:
            tuple: (latest_stoch_d, latest_stoch_k)
                - latest_stoch_d (float): Latest %D float value.
                - latest_stoch_k (float): Latest %K float value.

        Note:
            If %K > %D, it indicates an uptrend.
        """
        stoch_rsi = ta.momentum.StochRSIIndicator(close=main_df['close'], window=14)
        stoch_d = stoch_rsi.stochrsi_d()
        stoch_k = stoch_rsi.stochrsi_k()
        latest_stoch_d = stoch_d.iat[-1]
        latest_stoch_k = stoch_k.iat[-1]
        print(type(latest_stoch_d))
        return latest_stoch_d, latest_stoch_k

    def rsi(main_df):
        return ta.momentum.RSIIndicator(close=main_df['close'], window=14).rsi()

    def atr(main_df):
        return ta.volatility.AverageTrueRange(main_df['high'], main_df['low'], main_df['close']).average_true_range()

    def bollinger_bands(main_df):
        bb = ta.volatility.BollingerBands(close=main_df['close'], window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iat[-1]
        bb_lower = bb.bollinger_lband().iat[-1]
        return bb_upper, bb_lower

    
    # def ichimoku(main_df):
    #     ichimoku = ta.trend.IchimokuIndicator(high=main_df['high'], low=main_df['low'], window1=9, window2=26, window3=52)
    #     ichimoku_base = ichimoku.ichimoku_base_line()
    #     ichimoku_conversion = ichimoku.ichimoku_conversion_line()
    #     ichimoku_span_a = ichimoku.ichimoku_a()
    #     ichimoku_span_b = ichimoku.ichimoku_b()
    #     return ichimoku_base, ichimoku_conversion, ichimoku_span_a, ichimoku_span_b

    # def check_uptrend(ema_short, ema_medium, ema_long, main_df):
    #     uptrend = (ema_short.iloc[-1] > ema_medium.iloc[-1] > ema_long.iloc[-1]) and (ema_short.iloc[-2] > ema_medium.iloc[-2] > ema_long.iloc[-2])
    #     return uptrend

    # def check_rsi_trend(rsi, main_df):
    #     rsi_uptrend = rsi.iloc[-1] > rsi.iloc[-2] and rsi.iloc[-2] > rsi.iloc[-3]
    #     rsi_downtrend = rsi.iloc[-1] < rsi.iloc[-2] and rsi.iloc[-2] < rsi.iloc[-3]
    #     return rsi_uptrend, rsi_downtrend
"""
test
"""
# def get_prev_data() -> pd.DataFrame:
#     url = 'https://fapi.binance.com/fapi/v1/klines'
#     params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 200}
#     response = requests.get(url, params=params).json()
#     columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
#     df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
#     df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
#     df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
#     return df[columns]

# def identify_support_resistance(df, window=20):
#     df['rolling_min'] = df['low'].rolling(window=window, min_periods=1).min()
#     df['rolling_max'] = df['high'].rolling(window=window, min_periods=1).max()
#     support_levels = df['rolling_min'].drop_duplicates().reset_index(drop=True)
#     resistance_levels = df['rolling_max'].drop_duplicates().reset_index(drop=True)
#     return support_levels, resistance_levels

# # Fetch data and identify support and resistance levels
# df = get_prev_data()
# supports, resistances = identify_support_resistance(df)

# # Calculate VWAP using the Indicator class
# vwap_values = Indicator.vwap(df)
# print(vwap_values)

# # Plotting the data and VWAP
# plt.figure(figsize=(14, 7))
# plt.plot(df['openTime'], df['close'], label='Close Price')
# plt.plot(df['openTime'], vwap_values, label='VWAP', color='orange')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title('BTCUSDT Price and VWAP')
# plt.legend()
# plt.show()