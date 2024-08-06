from binance.um_futures import UMFutures
from binance.error import ClientError
from threading import Timer
from threading import Thread
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
        self.box_pattern = self.box_initialisation("15m")

    def box_initialisation(self, interval):
        fifteen_prev = CollectData.get_prev_data(self.symbol, interval)
        self.box_pattern = trend.PatternDetection.box_pattern_init(fifteen_prev)
        print(self.box_pattern[0])

if __name__ == "__main__":
    bot = Bot()
    

