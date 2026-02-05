import logging
from binance.um_futures import UMFutures
from binance.error import ClientError
import config

class BinanceClient:
    def __init__(self):
        self.client = UMFutures(key=config.API_KEY, secret=config.SECRET_KEY)

    def change_leverage(self, symbol, leverage):
        try:
            response = self.client.change_leverage(symbol=symbol, leverage=leverage)
            logging.info(f"Leverage changed: {response}")
            return response
        except ClientError as error:
            logging.error(f"Error changing leverage: {error.status_code} - {error.error_message}")
            return None

    def get_balance(self):
        try:
            response = self.client.balance()
            # Find USDT balance
            balance_data = next((x for x in response if x['asset'] == "USDT"), None)
            if balance_data:
                balance = float(balance_data['balance'])
                # Using 90% of available balance as per original logic
                return balance * 0.9
            return 0.0
        except ClientError as error:
            logging.error(f"Error getting balance: {error.status_code} - {error.error_message}")
            return 0.0
    
    def place_order(self, symbol, side, reduce_only, quantity):
        """
        Place a MARKET order.
        side: 'BUY' or 'SELL'
        """
        try:
            quantity = float(round(quantity, 3))
            response = self.client.new_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                reduceOnly=reduce_only, 
            )
            logging.info(f"Order placed: {response}")
            return response
        except ClientError as error:
            logging.error(f"Error placing order: {error.status_code} - {error.error_message}")
            return None
