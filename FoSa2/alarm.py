import telegram
import asyncio
import os

class TelegramFosa:
    def __init__(self) -> None:
        self.token = os.getenv('TELE_FOSA')
        self.bot = telegram.Bot(self.token)

    async def message(self, content):
        await self.bot.send_message(chat_id="6161371214", text=content)

    # async def run(self, content):
    #     await self.message(content)