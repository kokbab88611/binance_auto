import os 
from binance.um_futures import UMFutures
from binance.error import ClientError
import time
from datetime import datetime

class BinanceTrade:
    def __init__(self):
        self.api_key = os.getenv('Bin_API_KEY')
        self.api_secret = os.getenv('Bin_SECRET_KEY')
        self.client = UMFutures(key=self.api_key, secret=self.api_secret)
        self.leverage = 15
        self.fee = (0.06 / 100) 
        self.exchange_info = self.client.exchange_info()
    
    def market_open_position(self, side = "BUY", position_side="LONG", calced_quantity = 0):
        self.order(symbol=self.symbol.upper(), side = side, position_side = position_side, quantity=calced_quantity)

    def set_atr_based_sl_tp(self, entry_price, position, atr, quantity, atr_multiplier_tp, atr_multiplier_sl):
        balance = self.fetch_balance() 
        balance *= self.leverage
        fee_approx = balance
        simple_fee_usdt = balance * (0.07 / 100) # 0.07은 fee%
        fee_proifit_val = simple_fee_usdt/quantity
        one_percent_calc = entry_price * (0.01/self.leverage)
        long_minimum_tp = entry_price + fee_proifit_val + one_percent_calc
        short_minimum_tp = entry_price - fee_proifit_val - one_percent_calc

        if position == "long":
            tp_price = entry_price + (atr * atr_multiplier_tp)
            sl_price = entry_price - (atr * atr_multiplier_sl)
            take_profit_price = max(tp_price, long_minimum_tp)
        elif position == "short":
            tp_price = entry_price - (atr * atr_multiplier_tp)
            sl_price = entry_price + (atr * atr_multiplier_sl)
            take_profit_price = max(tp_price, long_minimum_tp)

        return str(round(take_profit_price, 2)), str(round(sl_price,2))

    def get_symbol_info(self, symbol: str):
        symbol = symbol.upper()
        for s in self.exchange_info['symbols']:
            if s['symbol'] == symbol:
                # print(s)
                tick_size = s['filters'][0]['tickSize']
                price_prcision = s['pricePrecision']
                quantity_prcision = s['quantityPrecision']
                # print(f"Tick Size: {tick_size}, Price Precision: {price_prcision}, Quantity precision: {quantity_prcision}")
        return None

    def fetch_balance(self):
        try:
            response = self.client.balance()
            # balance = next(x for x in response if x['asset'] == "USDT")['balance']
            available_balance = next(x for x in response if x['asset'] == "USDT")['availableBalance']
            print(f"Available Balance: {available_balance}")
            return round(float(available_balance),3) // 0.01 / 100 # floor down to 2 decimal
        except ClientError as e:
            print(f"Error fetching balance: {e}")
            return None

    def calculate_quantity(self, available_balance, price):
        max_quantity = round((((available_balance * (1 - (self.leverage * 0.005))) * self.leverage) / price) * 0.95 ,3)
        return max_quantity

    def set_leverage(self):
        try:
            response = self.client.change_leverage(symbol=self.symbol.upper(), leverage=self.leverage)
            print(f"Leverage set to {response['leverage']}")
        except ClientError as e:
            print(f"Error setting leverage: {e}")

    def order(self, symbol, side, position_side, quantity, order_type="MARKET", price=None, stop_price=None, close_position=False):
        quantity = quantity // 0.01 / 100 # floor to 2 dec
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
            # print('================================================================')

            # response = self.client.new_order(**params)
            # print(f"Order placed: {response}")
            # return response
        except ClientError as e:
            print(f"API error placing order: {e}")
            if 'timestamp' in str(e):
                print("Timestamp error detected. Resyncing time and retrying...")
                time.sleep(5)
                self.order(symbol, side, position_side, quantity, order_type, price, stop_price, close_position)
            return None

    def long(self, current_price, atr):
        if self.in_cooldown:
            return
        self.set_leverage()
        available_balance = self.fetch_balance()
        if available_balance is None or available_balance <= 0:
            print("Insufficient available balance to place order")
            return

        calced_quantity = self.calculate_quantity(available_balance, current_price)
        self.market_open_position(quantity=calced_quantity) # 빠른 시장가 매매
        in_atr = round(atr.iloc[-1], 2)
        enter_price = current_price
        price_profit, price_stoploss = self.set_atr_based_sl_tp(enter_price, in_atr, "long", balance=available_balance, quantity=calced_quantity)
        print(price_profit, price_stoploss)
        self.order(symbol=self.symbol.upper(), side="BUY", position_side="LONG", quantity=calced_quantity)
        time.sleep(1)
        self.order(symbol=self.symbol.upper(), side="SELL", position_side="LONG", quantity=calced_quantity, order_type="TAKE_PROFIT", price=price_profit, stop_price=price_profit, close_position=True)
        self.order(symbol=self.symbol.upper(), side="SELL", position_side="LONG", quantity=calced_quantity, order_type="STOP", price=price_stoploss, stop_price=price_stoploss, close_position=True)

        self.save_result(f"Opened long position at {current_price}")
        print('================================================================')
        print(f"Opened long position at {current_price}, Target Profit Price: {price_profit}, Stop Loss Price: {price_stoploss}")
        print('================================================================')

    def short(self, current_price, atr):
        if self.in_cooldown:
            return
        self.set_leverage()
        available_balance = self.fetch_balance()
        if available_balance is None or available_balance <= 0:
            print("Insufficient available balance to place order")
            return
        calced_quantity = self.calculate_quantity(available_balance, current_price)
        in_atr = round(atr.iloc[-1], 2)
        enter_price = current_price
        price_profit, price_stoploss = self.set_atr_based_sl_tp(enter_price, in_atr, "short", balance=available_balance, quantity= calced_quantity)

        self.order(symbol=self.symbol.upper(), side="SELL", position_side="SHORT", quantity=calced_quantity)
        time.sleep(1)
        self.order(symbol=self.symbol.upper(), side="BUY", position_side="SHORT", quantity=calced_quantity, order_type="TAKE_PROFIT", price=price_profit, stop_price=price_profit, close_position=True)
        self.order(symbol=self.symbol.upper(), side="BUY", position_side="SHORT", quantity=calced_quantity, order_type="STOP", price=price_stoploss, stop_price=price_stoploss, close_position=True)

        self.save_result(f"Opened short position at {current_price}")
        print('================================================================')
        print(f"Opened short position at {current_price}, Target Profit Price: {price_profit}, Stop Loss Price: {price_stoploss}")
        print('================================================================')

    def save_result(self, message):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{current_time} - {message}"
        with open(self.results_file, "a") as file:
            file.write(log_message + "\n")

if __name__ == "__main__":
    trader = BinanceTrade()
    symbol_info = trader.get_symbol_info("BTCUSDT")
    if symbol_info:
        print(f"Symbol Info: {symbol_info}")
    else:
        print("Symbol not found.")

    has_open_orders = trader.check_open_orders("BTCUSDT")
    print(f"Has open orders: {has_open_orders}")