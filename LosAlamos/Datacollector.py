import numpy as np
import pandas as pd
import ta  # Technical Analysis library for financial indicators
import requests
import json
from threading import Thread
import websocket as wb

class DataCollector:
    def __init__(self):
        self.symbol = "btcusdt"
        self.interval = "3m"
        self.volstream = "wss://fstream.binance.com/ws/btcusdt@aggTrade"
        self.websocket_url = f"wss://fstream.binance.com/ws/{self.symbol}@kline_{self.interval}"
        self.sell_volume = 0
        self.buy_volume = 0
        self.main_df = self.get_prev_data()
        self.results_file = "trade_results.log"

    def on_message_vol(self, ws, message):
        data = json.loads(message)
        market_maker = data['m']
        quantity = float(data['q'])
        if market_maker:
            self.sell_volume += quantity
        else:
            self.buy_volume += quantity

    def on_message_kline(self, ws, message):
        data = json.loads(message)
        openTime = data['k']['t']
        Open = float(data['k']['o'])
        High = float(data['k']['h'])
        Low = float(data['k']['l'])
        Close = float(data['k']['c'])
        Volume = float(data['k']['v'])
        isClosed = data['k']['x']
        df2 = {'openTime': openTime, 'Open': Open, 'High': High, 'Low': Low, 'Close': Close, 'Volume': Volume}
        self.live_edit(df2)
        if isClosed:
            self.main_df = self.add_frame(df2)
            self.buy_volume = 0
            self.sell_volume = 0

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def websocket_thread_vol(self):
        ws = wb.WebSocketApp(url=self.volstream, on_message=self.on_message_vol, on_close=self.on_close, on_error=self.on_error)
        ws.run_forever()

    def websocket_thread_kline(self):
        ws = wb.WebSocketApp(url=self.websocket_url, on_message=self.on_message_kline, on_close=self.on_close, on_error=self.on_error)
        ws.run_forever()

    def get_prev_data(self) -> pd.DataFrame:
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {'symbol': self.symbol, 'interval': self.interval, 'limit': 500}
        response = requests.get(url, params=params).json()
        df = pd.DataFrame(response, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
        df = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
        df = df.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'float'})
        return df

    def add_frame(self, df2):
        self.main_df.loc[len(self.main_df)] = df2
        return self.main_df

    def live_edit(self, df2):
        df2 = list(df2.values())
        self.main_df.iloc[-1] = df2
        if len(self.main_df) == 55:
            self.main_df = self.main_df.drop(self.main_df.index[:15]).reset_index(drop=True)

    def EMA(self):
        ema_short = ta.trend.EMAIndicator(self.main_df['Close'], window=9).ema_indicator()
        ema_medium = ta.trend.EMAIndicator(self.main_df['Close'], window=21).ema_indicator()
        ema_long = ta.trend.EMAIndicator(self.main_df['Close'], window=50).ema_indicator()
        return ema_short, ema_medium, ema_long

    def RSI(self):
        return ta.momentum.RSIIndicator(self.main_df['Close'], window=14).rsi()

    def VWAP(self):
        return ta.volume.VolumeWeightedAveragePrice(self.main_df['High'], self.main_df['Low'], self.main_df['Close'], self.main_df['Volume']).volume_weighted_average_price()

    def ATR(self):
        return ta.volatility.AverageTrueRange(self.main_df['High'], self.main_df['Low'], self.main_df['Close']).average_true_range()

    def volume_profile(self):
        vp = self.main_df.groupby('Close')['Volume'].sum()
        poc = vp.idxmax()
        return vp, poc

    def bollinger_bands(self):
        bb = ta.volatility.BollingerBands(close=self.main_df['Close'], window=20, window_dev=2)
        bb_middle = bb.bollinger_mavg()
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        return bb_middle, bb_upper, bb_lower

    def stochastic_oscillator(self):
        stoch = ta.momentum.StochasticOscillator(self.main_df['High'], self.main_df['Low'], self.main_df['Close'])
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()
        return stoch_k, stoch_d

    def MACD(self):
        macd = ta.trend.MACD(self.main_df['Close'])
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        return macd_line, signal_line

    def ADX(self):
        adx = ta.trend.ADXIndicator(high=self.main_df['High'], low=self.main_df['Low'], close=self.main_df['Close'], window=14)
        return adx.adx()

    def ichimoku(self):
        ichimoku = ta.trend.IchimokuIndicator(high=self.main_df['High'], low=self.main_df['Low'], window1=9, window2=26, window3=52)
        ichimoku_base = ichimoku.ichimoku_base_line()
        ichimoku_conversion = ichimoku.ichimoku_conversion_line()
        ichimoku_span_a = ichimoku.ichimoku_a()
        ichimoku_span_b = ichimoku.ichimoku_b()
        return ichimoku_base, ichimoku_conversion, ichimoku_span_a, ichimoku_span_b

    def prepare_dataset(self):
        # Compute technical indicators
        ema_short, ema_medium, ema_long = self.EMA()
        rsi = self.RSI()
        vwap = self.VWAP()
        atr = self.ATR()
        bb_middle, bb_upper, bb_lower = self.bollinger_bands()
        stoch_k, stoch_d = self.stochastic_oscillator()
        macd_line, signal_line = self.MACD()
        adx = self.ADX()
        ichimoku_base, ichimoku_conversion, ichimoku_span_a, ichimoku_span_b = self.ichimoku()

        # Create a DataFrame with the indicators as features
        features = pd.DataFrame({
            'ema_short': ema_short,
            'ema_medium': ema_medium,
            'ema_long': ema_long,
            'rsi': rsi,
            'vwap': vwap,
            'atr': atr,
            'bb_middle': bb_middle,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'adx': adx,
            'ichimoku_base': ichimoku_base,
            'ichimoku_conversion': ichimoku_conversion,
            'ichimoku_span_a': ichimoku_span_a,
            'ichimoku_span_b': ichimoku_span_b,
        }).dropna()

        # Add the target variable (e.g., future price change)
        features['target'] = self.main_df['Close'].shift(-1) - self.main_df['Close']
        features['target'] = features['target'].apply(lambda x: 1 if x > 0 else (2 if x < 0 else 0))  # 1 = Buy, 2 = Sell, 0 = Hold

        return features.dropna()
