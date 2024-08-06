from threading import Thread
import pandas as pd
import requests
import websocket as wb
import json
import plotly.graph_objects as go
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
class Kaufmantest:
    def __init__(self, symbol, interval) -> None:
        # 차트 분봉 
        print(f'{interval} 받음')
        self.symbol = symbol
        self.interval = interval
        self.main_df = self.get_prev_data()

    def get_prev_data(self) -> pd.DataFrame:
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {'symbol': self.symbol, 'interval': self.interval, 'limit': 100}
        response = requests.get(url, params=params).json()
        columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
        return df[columns].astype(float)

    def add_frame(self, df2):
        self.main_df = pd.concat([self.main_df, pd.DataFrame([df2])], ignore_index=True)
        return self.main_df

    def live_edit(self, df2):
        self.main_df.iloc[-1] = list(df2.values())
        if len(self.main_df) == 150:
            self.main_df = self.main_df.iloc[int(75):].reset_index(drop=True)

    def update_trade_status(self):
        self.previous_active = self.current_active
        self.current_active = self.trade.check_open_orders()

    def update_trade_status(self):
        time_series = self.main_df["close"]
        indices = find_peaks(time_series, distance=5)[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=time_series,
            mode='lines+markers',
            name='Original Plot'
        ))

        fig.add_trace(go.Scatter(
            x=indices,
            y=[time_series[j] for j in indices],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='cross'
            ),
            name='Detected Peaks'
        ))

        fig.show()

    def kaufman_efficiency_ratio(self, df, period=10):
        """
        Calculate the Kaufman Efficiency Ratio (ER) for a DataFrame with columns
        ['openTime', 'open', 'high', 'low', 'close', 'volume'].
        
        :param df: A pandas DataFrame with the required columns.
        :param period: The period over which to calculate the ER (default is 10).
        :return: A pandas Series with the ER values.
        """
        # Extract the 'close' prices
        prices = df['close']
        
        # Calculate the net change (not absolute)
        direction = abs(prices - prices.shift(period))
        
        # Calculate the Volatility
        volatility = prices.diff().abs().rolling(window=period).sum()
        
        # Compute the Efficiency Ratio
        er = direction / volatility
        
        print(er)

        # Plot the ER values
        plt.figure(figsize=(10, 5))
        plt.plot(df['openTime'], er, marker='o', markersize=8, linestyle='-', label='Kaufman Efficiency Ratio')
        plt.xlabel('Date')
        plt.ylabel('Efficiency Ratio')
        plt.title('Kaufman Efficiency Ratio Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    bot = Kaufmantest("btcusdt", "15m")
    print(bot.main_df)
    ratio = bot.kaufman_efficiency_ratio(bot.main_df, period=8)
    print(ratio)
