import requests
import pandas as pd
import matplotlib.pyplot as plt

# Fetch Historical Data
def get_prev_data() -> pd.DataFrame:
    url = 'https://fapi.binance.com/fapi/v1/klines'
    params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 200}
    response = requests.get(url, params=params).json()
    columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
    df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df[columns]

# Calculate Pivot Points
def calculate_pivot_points(data):
    high = data['high'].iloc[-2]
    low = data['low'].iloc[-2]
    close = data['close'].iloc[-2]
    
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return pivot, r1, s1, r2, s2, r3, s3

# Visualize Pivot Points
def visualize_pivot_points(data, pivot, r1, s1, r2, s2, r3, s3):
    plt.figure(figsize=(14, 7))
    plt.plot(data['openTime'], data['close'], label='Close Price', color='blue')
    
    plt.axhline(y=pivot, color='black', linestyle='-', label='Pivot Point (P)')
    plt.axhline(y=r1, color='red', linestyle='--', label='First Resistance (R1)')
    plt.axhline(y=s1, color='green', linestyle='--', label='First Support (S1)')
    plt.axhline(y=r2, color='red', linestyle=':', label='Second Resistance (R2)')
    plt.axhline(y=s2, color='green', linestyle=':', label='Second Support (S2)')
    plt.axhline(y=r3, color='red', linestyle='-.', label='Third Resistance (R3)')
    plt.axhline(y=s3, color='green', linestyle='-.', label='Third Support (S3)')
    
    plt.title('BTC/USDT Price with Pivot Points')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Putting it all together
def analyze_and_visualize():
    data = get_prev_data()
    pivot, r1, s1, r2, s2, r3, s3 = calculate_pivot_points(data)
    visualize_pivot_points(data, pivot, r1, s1, r2, s2, r3, s3)

# Run the function
analyze_and_visualize()
