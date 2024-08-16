import os 
from binance.um_futures import UMFutures
from binance.error import ClientError
import time
from datetime import datetime
from alarm import TelegramFosa
import asyncio

class BinanceTrade:
    def __init__(self, symbol):
        self.api_key = os.getenv('Bin_API_KEY')
        self.api_secret = os.getenv('Bin_SECRET_KEY')
        self.client = UMFutures(key=self.api_key, secret=self.api_secret)
        self.fee = (0.06 / 100) 
        self.symbol = symbol
        self.leverage = None
        self.log_file = f"{symbol}_trade_log.txt"  # Log file named after the symbol
        self.tele_alarm = TelegramFosa()

    def market_open_position(self, side, position_side, calced_quantity):
        self.order(symbol=self.symbol.upper(), side=side, position_side=position_side, quantity=calced_quantity)
        return calced_quantity

    def order_sl_tp(self, balance, entry_price, position, atr, calced_quantity, atr_multiplier_tp, atr_multiplier_sl):
        balance *= self.leverage
        simple_fee_usdt = balance * (0.07 / 100) # 0.07은 fee%
        fee_proifit_val = simple_fee_usdt / calced_quantity
        one_percent_calc = entry_price * (0.01 / self.leverage)
        long_minimum_tp = entry_price + fee_proifit_val + one_percent_calc
        short_minimum_tp = entry_price - fee_proifit_val - one_percent_calc
        print(calced_quantity)
        if position == "long":
            tp_price = entry_price + (atr * atr_multiplier_tp)
            sl_price = entry_price - (atr * atr_multiplier_sl)
            take_profit_price = max(tp_price, long_minimum_tp)

            #  datatype 정상화
            sl_price = str(round(sl_price,1))
            take_profit_price = str(round(take_profit_price,1))

            self.order(symbol=self.symbol.upper(), side="SELL", position_side="LONG", quantity=calced_quantity, order_type="TAKE_PROFIT", price=take_profit_price, stop_price=take_profit_price, close_position=True)
            self.order(symbol=self.symbol.upper(), side="SELL", position_side="LONG", quantity=calced_quantity, order_type="STOP", price=sl_price, stop_price=sl_price, close_position=True)
        elif position == "short":
            tp_price = entry_price - (atr * atr_multiplier_tp)
            sl_price = entry_price + (atr * atr_multiplier_sl)
            take_profit_price = min(tp_price, short_minimum_tp)

            #  datatype 정상화
            sl_price = str(round(sl_price,1))
            take_profit_price = str(round(take_profit_price,1))

            self.order(symbol=self.symbol.upper(), side="BUY", position_side="SHORT", quantity=calced_quantity, order_type="TAKE_PROFIT", price=take_profit_price, stop_price=take_profit_price, close_position=True)
            self.order(symbol=self.symbol.upper(), side="BUY", position_side="SHORT", quantity=calced_quantity, order_type="STOP", price=sl_price, stop_price=sl_price, close_position=True)
        
        self.log_trade(f"Opened {position} position at {entry_price}, TP: {take_profit_price}, SL: {sl_price}")
        asyncio.run(self.tele_alarm.message(f"{position} 포지션 {entry_price}에 열림\n목표가: {take_profit_price}\n손절가: {sl_price}"))

    def get_symbol_info(self, symbol: str):
        exchange_info = self.client.exchange_info()
        symbol = symbol.upper()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                tick_size = s['filters'][0]['tickSize']
                price_precision = s['pricePrecision']
                quantity_precision = s['quantityPrecision']
        return None

    def fetch_balance(self):
        try:
            response = self.client.balance()
            available_balance = next(x for x in response if x['asset'] == "USDT")['availableBalance']
            return round(float(available_balance), 3) // 0.01 / 100  # floor down to 2 decimal
        except ClientError as e:
            print(f"Error fetching balance: {e}")
            return None

    def calculate_quantity(self, price):
        available_balance = self.fetch_balance()
        max_quantity = round((((available_balance * (1 - (self.leverage * 0.005))) * self.leverage) / price) * 0.92, 3)
        return max_quantity, available_balance

    def set_leverage(self, leverage):
        try:
            response = self.client.change_leverage(symbol=self.symbol.upper(), leverage=leverage)
            self.leverage = leverage
            print(f"Leverage set to {response['leverage']}")
        except ClientError as e:
            print(f"Error setting leverage: {e}")

    def order(self, symbol, side, position_side, quantity, order_type="MARKET", price=None, stop_price=None, close_position=False):
        # quantity = quantity // 0.01 / 10  # floor to 2 decimal
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "positionSide": position_side,
                "quantity": quantity,
                "type": order_type,
                "timestamp": int(time.time() * 1000)
            }
            if close_position:
                params.update({"close_position": True})
            if order_type in ["STOP", "TAKE_PROFIT"]:
                if stop_price:
                    params.update({"stopPrice": stop_price})
                if price:
                    params.update({"price": price})

            # print('================================================================')
            # print(f"Placing order with params: {params}")
            print('================================================================')

            response = self.client.new_order(**params)
            print(f"Order placed: {response}")
            self.log_trade(f"Order placed: {params}")
            
            return response

        except ClientError as e:
            print(f"API error placing order: {e}")
            if 'timestamp' in str(e):
                print("Timestamp error detected. Resyncing time and retrying...")
                time.sleep(5)
                self.order(symbol, side, position_side, quantity, order_type, price, stop_price, close_position)
            return None

    def log_trade(self, message):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{current_time} - {message}"
        with open(self.log_file, "a") as file:
            file.write(log_message + "\n")
        print(log_message)  # Optional: Also print the log message to the console

    def check_open_orders(self):
        all_orders = self.client.get_orders(symbol=self.symbol)
        if len(all_orders) > 0:
            return True
        else:
            False

    def close_one_order(self):   
        all_orders = self.client.get_orders(symbol=self.symbol)
        if len(all_orders) == 1:
            self.client.cancel_order(symbol=self.symbol, orderId=all_orders[0]['orderId'], origClientOrderId=all_orders[0]['clientOrderId'])
            return True
        else:
            pass

if __name__ == "__main__":
    trader = BinanceTrade("btcusdt")
    symbol_info = trader.get_symbol_info("btcusdt")
    if symbol_info:
        print(f"Symbol Info: {symbol_info}")
    else:
        print("Symbol not found.")
