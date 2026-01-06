import json
import asyncio
import os
import sys
from dotenv import load_dotenv
from pocketoptionapi_async import AsyncPocketOptionClient

async def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get SSID from environment variable
    SSID = os.getenv("POCKETOPTION_SSID")
    
    if not SSID:
        sys.exit(1)
    
    # Auto-detect demo status from SSID
    is_demo = True  # Default to demo
    if SSID.startswith('42["auth",'):
        try:
            json_start = SSID.find("{")
            json_end = SSID.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_part = SSID[json_start:json_end]
                data = json.loads(json_part)
                is_demo = bool(data.get("isDemo", 1))
        except Exception:
            pass
    
    client = AsyncPocketOptionClient(SSID, is_demo=is_demo, enable_logging=False)
    
    connected = await client.connect()
    
    if not connected or not client.is_connected:
        sys.exit(1)
    
    await asyncio.sleep(1)  # Give connection time to initialize
    
    try:
        balance = await client.get_balance()
        print(f"{balance.balance} {balance.currency}")
    except Exception:
        sys.exit(1)
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

