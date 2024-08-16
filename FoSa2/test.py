import pandas as pd
import requests
import time

from indicators import Indicator
from ressup import SupportResistanceLevels as SRL

class Strategy:
    @staticmethod
    def check_trade_signal(df_5m, df_15m, df_1h):
        current_price = df_5m['close'].iat[-1]
        atr_15m = Indicator.atr(df_15m).iat[-1]

        # Calculate indicators
        vwap = Indicator.vwap(df_5m).iat[-1]
        bb_upper, bb_lower = Indicator.bollinger_bands(df_5m)

        # Calculate Stochastic RSI and RSI for 1h data
        latest_stoch_d, latest_stoch_k = Indicator.stochastic_rsi(df_1h)
        rsi = Indicator.rsi(df_1h).iat[-1]
        rsi_prev = Indicator.rsi(df_1h).iat[-2]

        # Determine if VWAP is within Bollinger Bands
        is_vwap_within_bb = bb_lower <= vwap <= bb_upper
        print(f"is_vwap_within_bb: {is_vwap_within_bb}")

        # Check Stochastic RSI condition
        is_stoch_rsi_long = latest_stoch_k > latest_stoch_d
        print(f"is_stoch_rsi_long: {is_stoch_rsi_long}")

        is_stoch_rsi_short = latest_stoch_k < latest_stoch_d
        print(f"is_stoch_rsi_short: {is_stoch_rsi_short}")
        
        ema_9, ema_15 = Indicator.EMA(df_5m, 9), Indicator.EMA(df_5m, 15)
        ema_cross_long = ema_9.iat[-1] > ema_15.iat[-1] and ema_9.iat[-2] > ema_15.iat[-2]
        print(f"ema_cross_long: {ema_cross_long}")

        ema_cross_short = ema_9.iat[-1] < ema_15.iat[-1] and ema_9.iat[-2] < ema_15.iat[-2]
        print(f"ema_cross_short: {ema_cross_short}")

        # Check RSI trend
        is_rsi_up = rsi > rsi_prev
        print(f"is_rsi_up: {is_rsi_up}")

        is_rsi_down = rsi < rsi_prev
        print(f"is_rsi_down: {is_rsi_down}")

        # Default scenarios if none provided
        long_scenarios = [
            is_vwap_within_bb and is_stoch_rsi_long and is_rsi_up and ema_cross_long
        ]
    
        short_scenarios = [
            is_vwap_within_bb and is_stoch_rsi_short and is_rsi_down and ema_cross_short
        ]

        # Evaluate scenarios 
        long_condition = any(long_scenarios)
        short_condition = any(short_scenarios)
        print(f"long condition {long_condition}")
        print(f"short condition {short_condition}")

    @staticmethod
    def box_trend_strategy(df_5m, df_15m, df_30m):
        resistance, support = SRL.identify_levels(df_15m)
        resistance_levels_filtered = SRL.remove_anomalies(resistance, resistance.iat[-1])
        support_levels_filtered = SRL.remove_anomalies(support, support.iat[-1])
        resistance_mean = resistance_levels_filtered.mean()
        support_mean = support_levels_filtered.mean()
        
        current_price = df_5m['close'].iat[-1]
        recent_30_min = df_30m[:-1].tail(4)

        recent_low_30 = recent_30_min['low'].min()
        recent_high_30 = recent_30_min['high'].max()

        ema_9 = Indicator.EMA(df_15m, length=9)
        ema_15 = Indicator.EMA(df_15m, length=15)
        atr_5m = Indicator.atr(df_5m).iat[-1]
        atr_15m = Indicator.atr(df_15m).iat[-1]
        
        print(f"support mean: {support_mean}")
        print(f"support mean: {resistance_mean}")
        print(f"BOX RESISTANCE ema_9 cross over 15: {ema_9.iat[-1] > ema_15.iat[-1] and ema_9.iat[-2] <= ema_15.iat[-2]}")

        if support_mean <= current_price <= resistance_mean:
            if ema_9.iat[-1] > ema_15.iat[-1] and ema_9.iat[-2] <= ema_15.iat[-2]:
                print("Enter BOX Long Position")


        if resistance_mean >= current_price >= support_mean:
            if ema_9.iat[-1] < ema_15.iat[-1] and ema_9.iat[-2] >= ema_15.iat[-2]:
                print("Enter BOX short Position")
        return False

    @staticmethod
    def box_breakout(trend, df_5m, df_15m, atr_15m, current_price, binancetrade):
        ema_9 = Indicator.EMA(df_15m, length=9)
        ema_15 = Indicator.EMA(df_15m, length=15)
        balance = binancetrade.fetch_balance()
        ema_cross_long = ema_9.iat[-1] > ema_15.iat[-1] and ema_9.iat[-2] <= ema_15.iat[-2]
        ema_cross_short = ema_9.iat[-1] < ema_15.iat[-1] and ema_9.iat[-2] >= ema_15.iat[-2]
        print(f"BOX BREAK ema_9 cross over 15: {ema_9.iat[-1] > ema_15.iat[-1] and ema_9.iat[-2] <= ema_15.iat[-2]}")
        if trend == 'long' and ema_cross_long:
            print("Enter BOX BREAK LONG Position")
            quantity, balance = binancetrade.calculate_quantity(current_price)
            binancetrade.market_open_position(side="BUY", position_side="LONG", calced_quantity=quantity)
            time.sleep(1)
            binancetrade.order_sl_tp(balance, current_price, "long", atr_15m, quantity, atr_multiplier_tp=1.3, atr_multiplier_sl=1.1)
            return True
        elif trend == 'short' and ema_cross_short:
            print("Enter BOX BREAK SHORT Position")
            quantity, balance = binancetrade.calculate_quantity(current_price)           
            binancetrade.market_open_position(side="SELL", position_side="SHORT", calced_quantity=quantity)
            time.sleep(1)
            binancetrade.order_sl_tp(balance, current_price, "short", atr_15m, quantity, atr_multiplier_tp=1.3, atr_multiplier_sl=1.1)
            return True
        else:
            return False

    @staticmethod
    def get_prev_data(symbol, interval) -> pd.DataFrame:
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {'symbol': symbol, 'interval': interval, 'limit': 150}
        response = requests.get(url, params=params).json()
        columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(response, columns=columns + ['closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
        df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df[columns]

if __name__ == "__main__":
    # Fetch historical data
    symbol = "BTCUSDT"
    df_5m = Strategy.get_prev_data(symbol, "5m")
    df_15m = Strategy.get_prev_data(symbol, "15m")
    df_30m = Strategy.get_prev_data(symbol, "30m")
    df_1h = Strategy.get_prev_data(symbol, "1h")

    # Mock BinanceTrade instance (replace with actual implementation)

    # Test check_trade_signal
    print("Testing check_trade_signal...")
    trade_signal_result = Strategy.check_trade_signal(df_5m, df_15m, df_1h)
    # Test box_trend_strategy
    print("Testing box_trend_strategy...")
    box_strategy_result = Strategy.box_trend_strategy(df_5m, df_15m, df_30m)
    print(f"Box Strategy Result: {box_strategy_result}")
