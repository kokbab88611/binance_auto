
from threading import Thread, Event
from datetime import datetime
import numpy as np
from collector import CollectData
import pandas as pd
import websocket as wb
import time
from trend import PatternDetection
from strategy import Strategy  # Ensure you have Strategy class with required methods
from binancetrade import BinanceTrade
class Bot:
    def __init__(self) -> None:
        self.symbol = "btcusdt"
        self.box_status = self.box_initialisation("15m")
        self.min_data_30 = CollectData(self.symbol, "30m")
        self.hour_data_1 = CollectData(self.symbol, "1h")
        self.min_data_15 = CollectData(self.symbol, "15m", box_update=self.on_new_candle_closed)
        self.min_data_5 = CollectData(self.symbol, "5m", callback=self.on_new_data)

        self.binance_trade = BinanceTrade(self.symbol)
        self.binance_trade.set_leverage(15)
        self.print_status_event = Event()
        self.print_status_thread = Thread(target=self.print_box_status_periodically)
        self.print_status_thread.start()

        self.position_open_status = False
        self.cooldown = False

    def on_new_candle_closed(self, main_df):
        # Update box status when a new candle closes
        self.box_status = PatternDetection.box_is_closed(main_df, atr_multiplier=0.1, box_status=self.box_status)
        print(f"Box Status Updated: {self.box_status}")

    def box_initialisation(self, interval):
        fifteen_prev = CollectData.get_prev_data(self.symbol, interval)
        box_status = PatternDetection.box_pattern_init(fifteen_prev)
        return box_status

    def check_trend(self):
        if not self.is_data_initialized():
            return
        fifteen_min_data = self.min_data_15.main_df
        is_closed = self.min_data_15.isClosed
        self.box_status = PatternDetection.live_detect_box_pattern(
            fifteen_min_data, atr_multiplier=0.1, box_status=self.box_status
        )

    def execute_strategy(self):
        """
        매번 신규 데이터를 받을경우 전략 가동
        """
        self.position_open_status = self.binance_trade.check_open_orders()

        if not self.is_data_initialized():
            return
        df_5m = self.min_data_5.main_df
        df_15m = self.min_data_15.main_df
        df_30m = self.min_data_30.main_df
        df_1h = self.hour_data_1.main_df
        self.check_trend()
        if not self.position_open_status and not self.cooldown:
            if self.box_status[-1] == 0:  # Not in a box trend
                self.position_open_status = Strategy.check_trade_signal(df_5m, df_15m, df_1h, self.binance_trade)
            else:  # In a box trend
                self.position_open_status = Strategy.box_trend_strategy(df_5m, df_15m, df_30m, self.binance_trade)
        else:
            cooldown = self.binance_trade.close_one_order()
            if cooldown:
                self.start_cooldown()

    def start_cooldown(self):
        self.cooldown = True
        print("Cooldown started for 5 minutes.")
        thread = Thread(target=self.cooldown_timer)
        thread.start()

    def cooldown_timer(self):
        time.sleep(300)  # 300 seconds = 5 minutes
        self.cooldown = False
        print("Cooldown ended, trading can resume.")

    def on_new_data(self):
        # Execute strategy every time new data is updated
        self.execute_strategy()

    def is_data_initialized(self):
        # Check if all the data frames are initialized and have data
        return (
            self.min_data_5.main_df is not None and not self.min_data_5.main_df.empty and
            self.min_data_15.main_df is not None and not self.min_data_15.main_df.empty and
            self.hour_data_1.main_df is not None and not self.hour_data_1.main_df.empty
        )

    def print_box_status_periodically(self):
        # Print the box status every 5 minutes
        while not self.print_status_event.is_set():
            print(f"Box Status: {self.box_status}")
            self.print_status_event.wait(300)  
                

    def stop(self):
        # Close all WebSocket connections before stopping
        self.min_data_5.close_websocket()
        self.min_data_15.close_websocket()
        self.min_data_30.close_websocket()
        self.hour_data_1.close_websocket()
        self.box_check_thread.join()

if __name__ == "__main__":
    bot = Bot()

    while True:
        pass

