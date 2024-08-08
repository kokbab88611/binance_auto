
from threading import Thread, Event
from datetime import datetime
import numpy as np
from collector import CollectData
import pandas as pd
import websocket as wb
import trend
from strategy import Strategy  # Ensure you have Strategy class with required methods

class Bot:
    def __init__(self) -> None:
        self.symbol = "btcusdt"
        self.five_min_data = CollectData(self.symbol, "5m", self.on_new_data)
        self.fifteen_min_data = CollectData(self.symbol, "15m")
        self.one_hour_data = CollectData(self.symbol, "1h")
        self.box_status = self.box_initialisation("15m")
        self.stop_event = Event()
        self.box_check_thread = Thread(target=self.run_box_check)
        self.box_check_thread.start()

    def box_initialisation(self, interval):
        fifteen_prev = CollectData.get_prev_data(self.symbol, interval)
        box_status = trend.PatternDetection.box_pattern_init(fifteen_prev)
        print(f"Initial Box Status: {box_status}")
        return box_status

    def check_trend(self):
        if not self.is_data_initialized():
            return
        
        fifteen_min_data = self.fifteen_min_data.main_df
        is_closed = self.fifteen_min_data.isClosed
        self.box_status = trend.PatternDetection.live_detect_box_pattern(
            fifteen_min_data, is_closed, atr_multiplier=0.1, box_status=self.box_status
        )
        print(f"Updated Box Status: {self.box_status}")

    def execute_strategy(self):
        if not self.is_data_initialized():
            return
        
        df_5m = self.five_min_data.main_df
        df_1h = self.one_hour_data.main_df
        df_15m = self.fifteen_min_data.main_df

        if self.box_status[-1] == 0:  # Not in a box trend
            Strategy.check_trade_signal(df_5m, df_1h)
        else:  # In a box trend
            Strategy.box_trend_strategy(df_15m)

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
            self.five_min_data.main_df is not None and not self.five_min_data.main_df.empty and
            self.fifteen_min_data.main_df is not None and not self.fifteen_min_data.main_df.empty and
            self.one_hour_data.main_df is not None and not self.one_hour_data.main_df.empty
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
