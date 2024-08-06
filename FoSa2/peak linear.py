import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Fetch Historical Data
def get_prev_data() -> pd.DataFrame:
    url = 'https://fapi.binance.com/fapi/v1/klines'
    params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 500}
    response = requests.get(url, params=params).json()
    columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
    df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df[columns]

# Step 2: Identify Local Minima and Maxima
def identify_local_extremes(data, window=5):
    data['local_min'] = data['low'][(data['low'].shift(1) > data['low']) & (data['low'].shift(-1) > data['low'])]
    data['local_max'] = data['high'][(data['high'].shift(1) < data['high']) & (data['high'].shift(-1) < data['high'])]
    return data

# Step 3: Fit Linear Regression Lines
def fit_regression_line(x, y):
    model = LinearRegression()
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    model.fit(x, y)
    return model

# Step 4: Draw Trend Lines with Breaks
def draw_trend_lines(data, threshold=0.02):
    plt.figure(figsize=(14, 7))
    plt.plot(data['openTime'], data['close'], label='Close Price', color='blue')
    
    support_points = data.dropna(subset=['local_min'])
    resistance_points = data.dropna(subset=['local_max'])

    # Convert 'openTime' to numerical values (Unix timestamps)
    support_points['time_numeric'] = support_points['openTime'].apply(lambda x: x.timestamp())
    resistance_points['time_numeric'] = resistance_points['openTime'].apply(lambda x: x.timestamp())

    # Plot support points
    plt.scatter(support_points['openTime'], support_points['local_min'], label='Support (Local Minima)', color='green', marker='v', s=100)
    # Plot resistance points
    plt.scatter(resistance_points['openTime'], resistance_points['local_max'], label='Resistance (Local Maxima)', color='red', marker='^', s=100)

    # Draw support trend lines
    last_point = support_points.iloc[0]
    segment_x, segment_y = [last_point['time_numeric']], [last_point['local_min']]
    for i in range(1, len(support_points)):
        current_point = support_points.iloc[i]
        if abs(current_point['local_min'] - last_point['local_min']) / last_point['local_min'] > threshold:
            model = fit_regression_line(segment_x, segment_y)
            plt.plot(pd.to_datetime(segment_x, unit='s'), model.predict(np.array(segment_x).reshape(-1, 1)), color='green', linestyle='--')
            segment_x, segment_y = [], []
        segment_x.append(current_point['time_numeric'])
        segment_y.append(current_point['local_min'])
        last_point = current_point
    if segment_x:
        model = fit_regression_line(segment_x, segment_y)
        plt.plot(pd.to_datetime(segment_x, unit='s'), model.predict(np.array(segment_x).reshape(-1, 1)), color='green', linestyle='--')

    # Draw resistance trend lines
    last_point = resistance_points.iloc[0]
    segment_x, segment_y = [last_point['time_numeric']], [last_point['local_max']]
    for i in range(1, len(resistance_points)):
        current_point = resistance_points.iloc[i]
        if abs(current_point['local_max'] - last_point['local_max']) / last_point['local_max'] > threshold:
            model = fit_regression_line(segment_x, segment_y)
            plt.plot(pd.to_datetime(segment_x, unit='s'), model.predict(np.array(segment_x).reshape(-1, 1)), color='red', linestyle='--')
            segment_x, segment_y = [], []
        segment_x.append(current_point['time_numeric'])
        segment_y.append(current_point['local_max'])
        last_point = current_point
    if segment_x:
        model = fit_regression_line(segment_x, segment_y)
        plt.plot(pd.to_datetime(segment_x, unit='s'), model.predict(np.array(segment_x).reshape(-1, 1)), color='red', linestyle='--')

    plt.title('BTC/USDT Price with Support and Resistance Trend Lines')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Putting it all together
def analyze_and_visualize():
    data = get_prev_data()
    data = identify_local_extremes(data)
    draw_trend_lines(data)

# Run the function
analyze_and_visualize()
