import pandas as pd
from indicators import Indicator

import websocket as wb

class TpSlWeb:
    @staticmethod
    def flexible_tpsl(df_5m):
        atr = Indicator.atr(df_5m)
        pass

    def set_atr_based_sl_tp(self, entry_price, atr, position, balance = 0, quantity = 0):
        balance *= self.leverage
        simple_fee_usdt = balance * (0.063 / 100) #hardcode for bnb 
        fee_proifit_val = simple_fee_usdt/quantity
        one_percent_calc = entry_price * (0.01/self.leverage)
        long_minimum_tp = entry_price + fee_proifit_val + one_percent_calc
        short_minimum_tp = entry_price - fee_proifit_val - one_percent_calc
        if atr > 2:
            atr = 2
        # Total required return to ensure minimum profit after fees
        if position == "long":
            stop_loss_price = entry_price - (atr * 0.9)
            atr_based_tp = entry_price + (atr * 1)
            take_profit_price = max(atr_based_tp, long_minimum_tp)
        # Adjust take-profit to ensure at least 1% profit after fees
        elif position == "short":
            stop_loss_price = entry_price + (atr * 0.9)
            atr_based_tp = entry_price - (atr * 1)
            take_profit_price = min(atr_based_tp, short_minimum_tp)
        return str(round(take_profit_price,2)), str(round(stop_loss_price,2))

    @staticmethod
    def box_tpsl(df_5m):
        