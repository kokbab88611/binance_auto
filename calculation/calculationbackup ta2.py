import requests
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

url = 'https://fapi.binance.com/fapi/v1/klines'
params = {
  'symbol': 'btcusdt',
  'interval': '3m',
  'limit': "500"

}
response = requests.get(url, params=params)
response = response.json()
df = pd.DataFrame(response, columns =['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime',
                                    'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore']) 
df = df.drop(df.columns[[6,7,8,9,10,11]], axis=1)
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
# print(df)
a = ta.trend.EMAIndicator(df['Close'], window=50)
b = a.ema_indicator()
# print(df['High'])

# bb = ta.volatility.BollingerBands(df['Close'], window = 20)
# bbh = bb.bollinger_hband()``
# bbl = bb.bollinger_lband()


bbh = ta.volatility.bollinger_hband(df['High'], window = 23)
bbl = ta.volatility.bollinger_lband(df['Low'], window = 23)

bh = ta.volatility.bollinger_hband(df['High'], window = 30)
bl = ta.volatility.bollinger_lband(df['Low'], window = 30)

nhi = ta.volatility.bollinger_hband_indicator(df['High'], window = 30)
nli = ta.volatility.bollinger_lband_indicator(df['Low'], window = 30)

bhi = ta.volatility.bollinger_hband_indicator(df['High'], window = 20)
bli = ta.volatility.bollinger_lband_indicator(df['Low'], window = 20)

bhi = np.array(bhi.tail(37).tolist())
bli = np.array(bli.tail(37).tolist())
bhi = bhi.astype('int')
bli = bli.astype('int')

nhi = np.array(nhi.tail(37).tolist())
nli = np.array(nli.tail(37).tolist())
nhi = nhi.astype('int')
nli = nli.astype('int')

sma = ta.trend.SMAIndicator(df['Close'], window = 13)
ss = sma.sma_indicator()

qq = ta.momentum.StochRSIIndicator(df['Close'], window = 14)
d = qq.stochrsi_d()
k = qq.stochrsi_k()

d_two = d.tail(2).tolist()
k_two = k.tail(2).tolist()

df_high = df['High']
df_low = df['Low']
df_close = df['Close']

money_flow = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
ma = (ta.trend.macd_diff(df['Close'], window_slow=13, window_fast=6, window_sign=4).tail(5)).tolist()
print(ma[-3] < ma[-2] and ma[-2] < ma[-1])
close_list = (df['Close'].tail(20)).to_list()

vwap = ta.volume.volume_weighted_average_price(df['High'],df['Low'],df['Close'],df['Volume']).tail(5)
vwap_high = vwap + 10
vwap_low = vwap - 10
print(f"vwap high {vwap_high}")
print(f"vwap low {vwap_low}")
# vwap_high_list, vwap_low_list = np.array(vwap_high), np.array(vwap_low)   
# vwap_short_check = np.subtract(close_list, vwap_high_list)[-20:]
# vwap_long_check = np.subtract(close_list, vwap_low_list)[-20:]
# vwap_short_check_bool = np.any(vwap_short_check > 0) and np.any(vwap_short_check < 0)
# vwap_long_check_bool = np.any(vwap_long_check > 0) and np.any(vwap_long_check < 0)

# print(vwap_high_list[-5:])
# print(vwap_low_list[-5:])

macd = ta.trend.macd(df_close)
atr = ta.volatility.AverageTrueRange(df_high, df_low, df_close)
atr_indicator = atr.average_true_range()

ema_fifty = ta.trend.EMAIndicator(df_close, window=50)
ema_fourteen = ta.trend.EMAIndicator(df_close, window=14)
ema_eight = ta.trend.EMAIndicator(df_close, window=8)
ema_fifty_indicator = ema_fifty.ema_indicator()
ema_fourteen_indicator = ema_fourteen.ema_indicator()
ema_eight_indicator = ema_eight.ema_indicator()
df['ema50'] = ema_fifty_indicator 
df['ema14'] = ema_fourteen_indicator 
df['ema8'] = ema_eight_indicator 
df['bbh'] = bbh.tail(400)
df['bbl'] = bbl.tail(400)
df["bh"] = bh.tail(400)
df["bl"] = bl.tail(400)
df['d'] = d
df['k'] = k
df['vwap'] = vwap
df['vwap+10'] = df['vwap'] + 10
df['vwap-10'] = df['vwap'] - 10
df['ss'] = ss.tail(400)

# print(bhi.tail(30).tolist())
# print(bhi.tail(30).tolist())
# print(d)
# print(k)
# print(d_two)
# print(type(df['vwap']))
df.plot(x="openTime", y=["Close", 'vwap','vwap+10','vwap-10'],
        kind="line", figsize=(25, 10))
# df.plot(x="openTime", y=["d", "k"],
#         kind="line", figsize=(10, 10))
mp.show()
