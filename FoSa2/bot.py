
from threading import Thread, Event
from datetime import datetime
import numpy as np
from collector import CollectData
import pandas as pd
import websocket as wb
import trend
from strategy import Strategy  # Ensure you have Strategy class with required methods
from binancetrade import BinanceTrade
class Bot:
    def __init__(self) -> None:
        self.symbol = "btcusdt"

        self.min_data_5 = CollectData(self.symbol, "5m", self.on_new_data)
        self.min_data_15 = CollectData(self.symbol, "15m")
        self.min_data_30 = CollectData(self.symbol, "30m")
        self.hour_data_1 = CollectData(self.symbol, "1h")

        self.box_status = self.box_initialisation("15m")
        self.stop_event = Event()
        self.box_check_thread = Thread(target=self.run_box_check)
        self.box_check_thread.start()

        self.binance_trade = BinanceTrade()

    def box_initialisation(self, interval):
        fifteen_prev = CollectData.get_prev_data(self.symbol, interval)
        box_status = trend.PatternDetection.box_pattern_init(fifteen_prev)
        print(f"Initial Box Status: {box_status}")
        return box_status

    def check_trend(self):
        if not self.is_data_initialized():
            return
        
        fifteen_min_data = self.min_data_15.main_df
        is_closed = self.min_data_15.isClosed
        self.box_status = trend.PatternDetection.live_detect_box_pattern(
            fifteen_min_data, is_closed, atr_multiplier=0.1, box_status=self.box_status
        )
        print(f"Updated Box Status: {self.box_status}")

    def execute_strategy(self):
        if not self.is_data_initialized():
            return
        
        df_5m = self.min_data_5.main_df
        df_15m = self.min_data_15.main_df
        df_30m = self.min_data_30.main_df
        df_1h = self.hour_data_1.main_df

        if self.box_status[-1] == 0:  # Not in a box trend
            Strategy.check_trade_signal(df_5m, df_1h, self.binance_trade)
        else:  # In a box trend
            Strategy.box_trend_strategy(df_5m, df_15m, df_30m, self.binance_trade)

    def on_new_data(self):
        # Execute strategy every time new data is updated
        self.execute_strategy()

    def run_box_check(self):
        while not self.stop_event.is_set():
            self.check_trend()
            self.stop_event.wait(60)  # Wait for 1 minute before checking again

    def is_data_initialized(self):
        # Check if all the data frames are initialized and have data
        return (
            self.min_data_5.main_df is not None and not self.min_data_5.main_df.empty and
            self.min_data_15.main_df is not None and not self.min_data_15.main_df.empty and
            self.hour_data_1.main_df is not None and not self.hour_data_1.main_df.empty
        )

    def stop(self):
        self.stop_event.set()
        self.box_check_thread.join()

if __name__ == "__main__":
    bot = Bot()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        bot.stop()
