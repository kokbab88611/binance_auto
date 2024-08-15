import asyncio
from alarm import TelegramFosa

# async def main():
#     a = TelegramFosa()
#     await a.message("테스트 중입니다")

if __name__ == "__main__":
    a = TelegramFosa()
    
    asyncio.run(a.message("테스트? \n 중입니다"))

    