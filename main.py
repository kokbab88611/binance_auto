import logging
from threading import Thread
from src.bot import TradingBot

# Configure Logging
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
        logging.info("Starting Binance Auto Trading Bot...")
        bot = TradingBot()
        
        # Start WebSocket in a separate thread
        websocket_thread = Thread(target=bot.start_socket)
        websocket_thread.start()
        
        logging.info("Bot is running. Press Ctrl+C to stop.")
        websocket_thread.join()
        
    except KeyboardInterrupt:
        logging.info("Stopping bot...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
