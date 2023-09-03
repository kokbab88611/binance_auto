from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
import requests
import ta
import numpy as np
import pandas as pd

url = 'https://fapi.binance.com/fapi/v1/klines'
params = {
    'symbol': 'BTCUSDT',
    'interval': '3m',
    'limit': "100"
}

response = requests.get(url, params=params)
response = response.json()
df = pd.DataFrame(response, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime',
                                     'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
df_main = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
df_main['High'] = pd.to_numeric(df_main['High'], errors='coerce')
df_main['Low'] = pd.to_numeric(df_main['Low'], errors='coerce')
df_main['Close'] = pd.to_numeric(df_main['Close'], errors='coerce')
df_main['Volume'] = pd.to_numeric(df_main['Volume'], errors='coerce')
df_main['Open'] = pd.to_numeric(df_main['Open'], errors='coerce')
df_main['Category'] = np.where(df_main['Close'] > df_main['Open'], 'bullish', 'bearish')

def stochRSI():
    rsi = ta.momentum.StochRSIIndicator(df_main['Close'], window = 14)
    d = rsi.stochrsi_d()
    k = rsi.stochrsi_k()
    df_main['d'] = d 
    df_main['k'] = k    

def EMA():
    ema_fourteen = ta.trend.EMAIndicator(df_main['Close'], window=14, fillna = True)
    ema_eight = ta.trend.EMAIndicator(df_main['Close'], window=8, fillna = True)
    ema_five = ta.trend.EMAIndicator(df_main['Close'], window=5, fillna = True)
    ema_fourteen_indicator = ema_fourteen.ema_indicator()
    ema_eight_indicator = ema_eight.ema_indicator()
    ema_five_indicator = ema_five.ema_indicator()
    df_main['ema14'] = ema_fourteen_indicator 
    df_main['ema8'] = ema_eight_indicator 
    df_main['ema5'] = ema_five_indicator 

def macd():
    ma = (ta.trend.macd_diff(df_main['Close'], window_slow=13, window_fast=6, window_sign=4, fillna = True))
    df_main['ma'] = ma

def money_flow_index():
    mfi = ta.volume.money_flow_index(df_main['High'], df_main['Low'], df_main['Close'], df_main['Volume'], fillna = True)
    df_main['mfi'] = mfi

EMA()
macd()
money_flow_index()
stochRSI()
df_main = df_main.drop(df_main.index[:49])
df_main = df_main.reset_index(drop=True)
print(df_main.to_string())
label_encoder = LabelEncoder()
df_main['Category'] = label_encoder.fit_transform(df_main['Category'])

# Split the Data
X_train_category, X_test_category, y_train_category, y_test_category = train_test_split(df_main[['Close', 'Volume', 'High', 'Low']], df_main['Category'],
                                                                      test_size=0.2, random_state=42)

# Create the Logistic Regression model for Category prediction
df_train_category = pd.concat([X_train_category, y_train_category], axis=1).dropna()
X_train_category = df_train_category.iloc[:, :-1]
y_train_category = df_train_category.iloc[:, -1]

# Create the Logistic Regression model for Category prediction
model_category = LogisticRegression(max_iter=1000)
model_category.fit(X_train_category, y_train_category)

# Evaluate the models
# accuracy_category = model_category.score(X_test_category, y_test_category)
# print("Accuracy (Category):", accuracy_category)

# Predict Future Candles
X_new_category = X_test_category.tail(1)  # Get the last 5 rows of category test data

# Predict Category for Next 5 Candles
predicted_category = model_category.predict(X_new_category)
# predicted_category_labels = label_encoder.inverse_transform(predicted_category)  # Convert numeric labels to original labels
print("Predicted Category for Next 5 Candles:", predicted_category[0])
