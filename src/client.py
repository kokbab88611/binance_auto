import logging
from binance.um_futures import UMFutures
from binance.error import ClientError
import config

class BinanceClient:
    def __init__(self):
        self.um_futures_client = UMFutures(key=config.API_KEY, secret=config.SECRET_KEY)

    def change_leverage(self, leverage):
        try:
            response = self.um_futures_client.change_leverage(symbol=config.SYMBOL, leverage=leverage)
            logging.info(response)
        except ClientError as error:
            logging.error(f"Error: {error.status_code} Code: {error.error_code} Message: {error.error_message}")

    def get_balance(self):
        try:
            response = self.um_futures_client.balance()
            balance = float(next(x for x in response if x['asset'] == "USDT")['balance'])
            return balance * 0.9
        except ClientError as error:
            logging.error(f"Error: {error.status_code} Code: {error.error_code} Message: {error.error_message}")
            return 0.0
    
    def place_order(self, symbol, side, reduceOnly, quantity):
        try:
            quantity = float(round(quantity, 3))
            response = self.um_futures_client.new_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                reduceOnly=reduceOnly,
            )
            print(response)
        except ClientError as error:
            logging.error(f"Error: {error.status_code} Code: {error.error_code} Message: {error.error_message}")
