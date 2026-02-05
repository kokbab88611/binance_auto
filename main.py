import logging
from threading import Thread
from src.bot import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trade_bot.log"),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    try:
        bot = TradingBot()
        websocket_thread = Thread(target=bot.websocket_thread)
        websocket_thread.start()
        websocket_thread.join()
    except KeyboardInterrupt:
        logging.info("Stopping bot...")
    except Exception as e:
        logging.error(f"Error: {e}")
