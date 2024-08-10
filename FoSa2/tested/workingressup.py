import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fetch Historical Data
def get_prev_data() -> pd.DataFrame:
    url = 'https://fapi.binance.com/fapi/v1/klines'
    params = {'symbol': 'BTCUSDT', 'interval': '15m', 'limit': 15}
    response = requests.get(url, params=params).json()
    columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
    df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df[columns]

# Identify Resistance Levels and their locations
def identify_resistance_levels(df, window=5):
    rolling_max = df['high'].rolling(window=window, min_periods=1).max()
    resistance_levels = rolling_max[rolling_max.shift(1) < rolling_max]
    recent_resistances = resistance_levels.dropna().tail(3)  # Get the last 3 resistance levels
    return recent_resistances

# Identify Support Levels and their locations
def identify_support_levels(df, window=5):
    rolling_min = df['low'].rolling(window=window, min_periods=1).min()
    support_levels = rolling_min[rolling_min.shift(1) > rolling_min]
    recent_supports = support_levels.dropna().tail(3)  # Get the last 3 support levels
    return recent_supports

# Remove anomalies if they are more than 0.5% away from the reference price
def remove_anomalies(levels, reference_price, threshold=0.005):
    filtered_levels = levels[np.abs((levels - reference_price) / reference_price) <= threshold]
    return filtered_levels

# Calculate the average of levels
def calculate_average_level(levels):
    return levels.mean()

# Plot Support and Resistance Levels, points and averages
def plot_levels(df, resistance_levels, support_levels, average_resistance, average_support):
    plt.figure(figsize=(14, 7))
    plt.plot(df['openTime'], df['close'], label='Close Price', color='blue')
    
    for index, level in resistance_levels.items():
        plt.axhline(y=level, color='red', linestyle='--', label=f'Resistance {level:.2f}')
        plt.plot(df.loc[index, 'openTime'], level, 'ro')  # Mark the resistance point

    for index, level in support_levels.items():
        plt.axhline(y=level, color='green', linestyle='--', label=f'Support {level:.2f}')
        plt.plot(df.loc[index, 'openTime'], level, 'go')  # Mark the support point
    
    plt.axhline(y=average_resistance, color='red', linestyle='-', label=f'Average Resistance {average_resistance:.2f}')
    plt.axhline(y=average_support, color='green', linestyle='-', label=f'Average Support {average_support:.2f}')
    
    plt.title('BTC/USDT Price with Recent 3 Support and Resistance Levels and Averages')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main Execution
df = get_prev_data()

resistance_levels = identify_resistance_levels(df)
support_levels = identify_support_levels(df)

# Determine the most recent resistance and support
most_recent_resistance = resistance_levels.iloc[-1] if not resistance_levels.empty else None
most_recent_support = support_levels.iloc[-1] if not support_levels.empty else None

# Remove anomalies based on recent resistance and support
if most_recent_resistance is not None:
    resistance_levels_filtered = remove_anomalies(resistance_levels, most_recent_resistance)
else:
    resistance_levels_filtered = resistance_levels

if most_recent_support is not None:
    support_levels_filtered = remove_anomalies(support_levels, most_recent_support)
else:
    support_levels_filtered = support_levels

# Calculate averages
average_resistance = calculate_average_level(resistance_levels_filtered)
average_support = calculate_average_level(support_levels_filtered)

# Plot levels
plot_levels(df, resistance_levels_filtered, support_levels_filtered, average_resistance, average_support)
