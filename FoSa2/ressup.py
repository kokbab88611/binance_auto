import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SupportResistanceLevels:
    # @staticmethod
    # def get_prev_data(symbol='BTCUSDT', interval='5m', limit=100) -> pd.DataFrame:
    #     url = 'https://fapi.binance.com/fapi/v1/klines'
    #     params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    #     response = requests.get(url, params=params).json()
    #     columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
    #     df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
    #     df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
    #     df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    #     return df[columns]

    @staticmethod
    def identify_levels(df, window=5):
        # Identify resistance levels
        rolling_max = df['high'].rolling(window=window, min_periods=1).max()
        resistance_levels = rolling_max[rolling_max.shift(1) < rolling_max]
        recent_resistances = resistance_levels.dropna().tail(3)  # Get the last 3 resistance levels

        # Identify support levels
        rolling_min = df['low'].rolling(window=window, min_periods=1).min()
        support_levels = rolling_min[rolling_min.shift(1) > rolling_min]
        recent_supports = support_levels.dropna().tail(3)  # Get the last 3 support levels

        return recent_resistances, recent_supports

    @staticmethod
    def remove_anomalies(levels, reference_price, threshold=0.005):
        filtered_levels = levels[np.abs((levels - reference_price) / reference_price) <= threshold]
        return filtered_levels

    @staticmethod
    def get_levels(df):
        resistance_levels, support_levels = SupportResistanceLevels.identify_levels(df)

        # Determine the most recent resistance and support
        most_recent_resistance = resistance_levels.iloc[-1] if not resistance_levels.empty else None
        most_recent_support = support_levels.iloc[-1] if not support_levels.empty else None

        # Remove anomalies based on recent resistance and support
        if most_recent_resistance is not None:
            resistance_levels_filtered = SupportResistanceLevels.remove_anomalies(resistance_levels, most_recent_resistance)
        else:
            resistance_levels_filtered = resistance_levels

        if most_recent_support is not None:
            support_levels_filtered = SupportResistanceLevels.remove_anomalies(support_levels, most_recent_support)
        else:
            support_levels_filtered = support_levels

        # Calculate averages
        average_resistance = SupportResistanceLevels.calculate_average_level(resistance_levels_filtered)
        average_support = SupportResistanceLevels.calculate_average_level(support_levels_filtered)

        return resistance_levels_filtered, support_levels_filtered, average_resistance, average_support

#     @staticmethod
#     def plot_levels(df, resistance_levels, support_levels, average_resistance, average_support):
#         plt.figure(figsize=(14, 7))
#         plt.plot(df['openTime'], df['close'], label='Close Price', color='blue')

#         for index, level in resistance_levels.items():
#             plt.axhline(y=level, color='red', linestyle='--', label=f'Resistance {level:.2f}')
#             plt.plot(df.loc[index, 'openTime'], level, 'ro')  # Mark the resistance point

#         for index, level in support_levels.items():
#             plt.axhline(y=level, color='green', linestyle='--', label=f'Support {level:.2f}')
#             plt.plot(df.loc[index, 'openTime'], level, 'go')  # Mark the support point

#         plt.axhline(y=average_resistance, color='red', linestyle='-', label=f'Average Resistance {average_resistance:.2f}')
#         plt.axhline(y=average_support, color='green', linestyle='-', label=f'Average Support {average_support:.2f}')

#         plt.title('BTC/USDT Price with Recent 3 Support and Resistance Levels and Averages')
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.legend()
#         plt.show()

# # Example usage
# df = SupportResistanceLevels.get_prev_data()
# resistance_levels_filtered, support_levels_filtered, average_resistance, average_support = SupportResistanceLevels.get_levels(df)

# print("Filtered Resistance Levels:")
# print(resistance_levels_filtered)
# print(f"Average Resistance Level: {average_resistance:.2f}")

# print("Filtered Support Levels:")
# print(support_levels_filtered)
# print(f"Average Support Level: {average_support:.2f}")

# # Example usage for visual display (commented out, not used in main execution)
# # SupportResistanceLevels.plot_levels(df, resistance_levels_filtered, support_levels_filtered, average_resistance, average_support)
