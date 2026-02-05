import os

# Trading Settings
SYMBOL = "BTCUSDT"
INTERVAL = "3m"
LIMIT = "70"

# WebSocket URL
WS_URL = f"wss://stream.binance.us:9443/{SYMBOL.lower()}@kline_{INTERVAL}"

# API Endpoints
API_URL = 'https://fapi.binance.us/fapi/v1/klines'

# Trading Parameters
LEVERAGE = 25
RISK_REWARD_RATIO = 1.5
QUANTITY_MULTIPLIER = 0.95

# API Keys (Loaded from environment variables for security)
API_KEY = os.getenv('Bin_API_KEY')
SECRET_KEY = os.getenv('Bin_SECRET_KEY')
