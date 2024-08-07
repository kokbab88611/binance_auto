from binance.um_futures import UMFutures
from binance.error import ClientError
from threading import Thread, Event
from datetime import datetime
import numpy as np
from collector import CollectData 
import ta
import pandas as pd
import requests
import websocket as wb
import json
import trend

class Bot:
    def __init__(self) -> None:
        self.symbol = "btcusdt"
        self.five_min_data = CollectData(self.symbol, "5m")
        self.fifteen_min_data = CollectData(self.symbol, "15m")
        self.one_hour_data = CollectData(self.symbol, "1h")
        self.box_status = self.box_initialisation("15m")
        self.stop_event = Event()
        self.box_check_thread = Thread(target=self.run)
        self.box_check_thread.start()

    def box_initialisation(self, interval):
        fifteen_prev = CollectData.get_prev_data(self.symbol, interval)
        box_status = trend.PatternDetection.box_pattern_init(fifteen_prev)
        print(f"Initial Box Status: {box_status}")
        return box_status

    def check_trend(self):
        fifteen_min_data = self.fifteen_min_data.main_df
        ema_medium = trend.Indicator.EMA(fifteen_min_data)[1]
        atr = trend.Indicator.atr(fifteen_min_data)
        is_closed = self.fifteen_min_data.isClosed
        self.box_status = trend.PatternDetection.live_detect_box_pattern(
            ema_medium, atr, is_closed, atr_multiplier=0.1, box_status=self.box_status
        )
        print(f"Updated Box Status: {self.box_status}")

    def run(self):
        while not self.stop_event.is_set():
            self.check_trend()
            self.stop_event.wait(60)  # Wait for 1 minute before checking again

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
