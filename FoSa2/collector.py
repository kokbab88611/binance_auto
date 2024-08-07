from threading import Thread
import pandas as pd
import requests
import websocket as wb
import json

class CollectData:
    def __init__(self, symbol, interval, callback=None) -> None:
        print(f'{interval} 받음')
        self.symbol = symbol
        self.interval = interval
        self.isClosed = None
        self.websocket_url = f"wss://fstream.binance.com/ws/{self.symbol}@kline_{self.interval}"
        self.main_df = CollectData.get_prev_data(symbol, interval)
        self.callback = callback
        websocket_thread_kline = Thread(target=self.websocket_thread_kline)
        websocket_thread_kline.start()

    def on_message_kline(self, ws, message):
        data = json.loads(message)
        kline_data = data['k']
        self.isClosed = kline_data['x']
        df2 = {
            'openTime': pd.to_datetime(kline_data['t'], unit='ms'),
            'open': float(kline_data['o']),
            'high': float(kline_data['h']),
            'low': float(kline_data['l']),
            'close': float(kline_data['c']),
            'volume': float(kline_data['v'])
        }
        self.live_edit(df2)
        print(self.main_df)
        if self.isClosed:
            self.main_df = self.add_frame(df2)
            self.isClosed = None

        # Call the callback function if provided
        if self.callback:
            self.callback()

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed with code:", close_status_code, "and message:", close_msg)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def websocket_thread_kline(self):
        ws = wb.WebSocketApp(
            url=self.websocket_url,
            on_message=self.on_message_kline,
            on_close=self.on_close,
            on_error=self.on_error
        )
        ws.run_forever()

    @staticmethod
    def get_prev_data(symbol, interval) -> pd.DataFrame:
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {'symbol': symbol, 'interval': interval, 'limit': 150}
        response = requests.get(url, params=params).json()
        columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
        df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df[columns]

    def add_frame(self, df2):
        df2 = pd.DataFrame([df2])
        df2['openTime'] = pd.to_datetime(df2['openTime'], unit='ms')  # Ensure correct dtype
        self.main_df = pd.concat([self.main_df, df2], ignore_index=True)
        return self.main_df

    def live_edit(self, df2):
        self.main_df.iloc[-1] = list(df2.values())
        if len(self.main_df) == 155:
            self.main_df = self.main_df.iloc[int(5):].reset_index(drop=True)

if __name__ == "__main__":
    bot = CollectData("btcusdt", "5m")
