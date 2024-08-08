import os 
from binance.um_futures import UMFutures
from binance.error import ClientError
import time

class BinanceTrade:
    def __init__(self):
        self.api_key = os.getenv('Bin_API_KEY')
        self.api_secret = os.getenv('Bin_SECRET_KEY')
        self.client = UMFutures(key=self.api_key, secret=self.api_secret)
        self.leverage = 15
        self.fee = (0.06 / 100) 
        self.exchange_info = self.client.exchange_info()

    def set_atr_based_sl_tp(self, entry_price, position, balance = 0, quantity = 0):
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
            return round(float(available_balance),3)
        except ClientError as e:
            print(f"Error fetching balance: {e}")
            return None

    def calculate_quantity(self, available_balance, price):
        max_quantity = round((((available_balance * (1 - (self.leverage * 0.005))) * self.leverage) / price) * 0.9 ,3)
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

if __name__ == "__main__":
    trader = BinanceTrade()
    symbol_info = trader.get_symbol_info("BTCUSDT")
    if symbol_info:
        print(f"Symbol Info: {symbol_info}")
    else:
        print("Symbol not found.")

    has_open_orders = trader.check_open_orders("BTCUSDT")
    print(f"Has open orders: {has_open_orders}")