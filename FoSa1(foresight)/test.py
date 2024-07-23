import numpy as np
import ta
import pandas as pd
def supertrend(df, period=10, multiplier=3):
    # Initialize the Average True Range (ATR) from the ta library
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()

    # Calculate the basic upper and lower bands of the Supertrend indicator
    hl2 = (df['high'] + df['low']) / 2
    df['upperband'] = hl2 - (multiplier * atr)
    df['lowerband'] = hl2 + (multiplier * atr)
    df['Supertrend'] = 0.0
    df['trend'] = 0

    # Initial trend conditions
    if df['close'][0] > df['upperband'][0]:
        df['Supertrend'][0] = df['upperband'][0]
        df['trend'][0] = 1
    else:
        df['Supertrend'][0] = df['lowerband'][0]
        df['trend'][0] = 0

    # Compute Supertrend 
    for i in range(1, len(df)):
        if df['close'][i] > df['upperband'][i-1]:
            df['Supertrend'][i] = df['upperband'][i]
            df['trend'][i] = 1
        elif df['close'][i] < df['lowerband'][i-1]:
            df['Supertrend'][i] = df['lowerband'][i]
            df['trend'][i] = -1
        else:
            df['Supertrend'][i] = df['Supertrend'][i-1] if df['close'][i] > df['Supertrend'][i-1] else df['lowerband'][i]
            df['trend'][i] = df['trend'][i-1]

        df['upperband'][i] = max(df['upperband'][i], df['Supertrend'][i]) if df['close'][i] > df['Supertrend'][i-1] else df['upperband'][i]
        df['lowerband'][i] = min(df['lowerband'][i], df['Supertrend'][i]) if df['close'][i] < df['Supertrend'][i-1] else df['lowerband'][i]

    df['buy_signal'] = ((df['trend'] == 1) & (df['trend'].shift(1) == -1))
    df['sell_signal'] = ((df['trend'] == -1) & (df['trend'].shift(1) == 1))

    last_signal = "None"
    if df.iloc[-1]['buy_signal']:
        last_signal = "Buy"
    elif df.iloc[-1]['sell_signal']:
        last_signal = "Sell"

    return last_signal, df.iloc[-1]['trend']

df = pd.DataFrame(data = {
    'high': [130, 132, 135, 133, 136, 137, 138, 140, 142, 141, 143, 145, 146, 147, 148, 150, 151, 152, 153, 155],
    'low': [128, 130, 132, 131, 133, 134, 135, 137, 139, 138, 140, 142, 143, 144, 145, 147, 148, 149, 150, 152],
    'close': [129, 131, 133, 132, 135, 136, 137, 139, 140, 140, 142, 144, 145, 146, 147, 149, 150, 151, 152, 154]
})
signal, current_trend = supertrend(df)
print("Latest Signal:", signal)
print("Current Trend:", current_trend)