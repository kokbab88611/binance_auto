# Binance Auto Trading Bot

Automated futures trading bot for Binance US that combines technical analysis with machine learning for trade signal generation.

⚠️ **Educational Project** - This bot is still in development. Do not use with real funds.

## Overview

This bot monitors BTC/USDT futures on 3-minute timeframes and executes trades based on:
- EMA crossover signals (8/14)
- StochRSI momentum indicators
- Logistic Regression trend prediction
- Volatility-based risk filters

## Features

- Real-time WebSocket price monitoring
- Automated entry/exit based on technical + ML signals
- 25x leverage position management
- Stop loss and take profit automation

## Project Structure

```
binance_auto/
├── config.py              # Trading parameters and API settings
├── main.py                # Entry point
├── requirements.txt       # Dependencies
└── src/
    ├── bot.py            # Main trading logic
    ├── client.py         # Binance API wrapper
    ├── strategy.py       # Technical indicators
    └── ml_models.py      # ML prediction model
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export Bin_API_KEY="your_api_key"
export Bin_SECRET_KEY="your_secret_key"
```

3. Run the bot:
```bash
python main.py
```

## How It Works

1. **Data Collection**: Historical klines fetched via REST API, live data via WebSocket
2. **Indicator Calculation**: EMA, StochRSI, Bollinger Bands computed on each update
3. **ML Prediction**: Logistic Regression trained on recent candles to predict bullish/bearish trend
4. **Signal Generation**: All conditions must align (momentum + trend + ML + volatility filter)
5. **Order Execution**: Market orders placed with 25x leverage, managed with fixed TP/SL

## Goal

Achieve consistent 2% profit per trade using 25x leverage on crypto futures.

## License

MIT
