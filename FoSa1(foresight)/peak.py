import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress
import ta
import requests

# Function to get previous data
def get_prev_data() -> pd.DataFrame:
    url = 'https://fapi.binance.com/fapi/v1/klines'
    params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 50}
    response = requests.get(url, params=params).json()
    columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
    df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df[columns]

# Load data
df = get_prev_data()

# Detect high peaks
high_peaks, _ = find_peaks(df['high'], distance=5)

# Detect low peaks (invert the price series)
low_peaks, _ = find_peaks(-df['low'], distance=5)

# Linear regression for high peaks
high_x = np.array(range(len(df)))[high_peaks]
high_y = df['high'].iloc[high_peaks].values
high_slope, high_intercept, _, _, _ = linregress(high_x, high_y)
high_line = high_slope * np.array(range(len(df))) + high_intercept

# Linear regression for low peaks
low_x = np.array(range(len(df)))[low_peaks]
low_y = df['low'].iloc[low_peaks].values
low_slope, low_intercept, _, _, _ = linregress(low_x, low_y)
low_line = low_slope * np.array(range(len(df))) + low_intercept

# Plot the data along with high and low peaks and trend lines
plt.figure(figsize=(14, 7))
plt.plot(df['openTime'], df['close'], label='Close Price')
plt.plot(df['openTime'], df['high'], label='High Price', linestyle='--', alpha=0.3)
plt.plot(df['openTime'], df['low'], label='Low Price', linestyle='--', alpha=0.3)
plt.plot(df['openTime'].iloc[high_peaks], df['high'].iloc[high_peaks], 'r^', label='High Peaks')
plt.plot(df['openTime'].iloc[low_peaks], df['low'].iloc[low_peaks], 'gv', label='Low Peaks')
plt.plot(df['openTime'], high_line, 'r-', label='High Peaks Trend Line')
plt.plot(df['openTime'], low_line, 'g-', label='Low Peaks Trend Line')
plt.title('Close Prices with Detected High and Low Peaks and Trend Lines')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
