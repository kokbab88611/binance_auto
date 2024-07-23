from binance.um_futures import UMFutures
from binance.error import ClientError
from threading import Timer
from threading import Thread
from datetime import datetime
import numpy as np
import ta
import pandas as pd
import requests
import websocket as wb
import json
import os 

class Bot:
    def __init__(self) -> None:
        pass