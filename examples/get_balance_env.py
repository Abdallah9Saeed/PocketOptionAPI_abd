import json
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from pocketoptionapi_async import AsyncPocketOptionClient

def write_log(message: str, log_file: str = "balance_log.txt"):
    """كتابة رسالة إلى ملف السجل"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception:
        pass  # Ignore logging errors

async def main():
    # Load environment variables from .env file
    load_dotenv()
    write_log("Starting balance check...")
    
    # Get SSID from environment variable
    SSID = os.getenv("POCKETOPTION_SSID")
    
    if not SSID:
        write_log("ERROR: POCKETOPTION_SSID not found in .env file")
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
    
    account_type = "Demo" if is_demo else "Real"
    write_log(f"Connecting to PocketOption ({account_type} account)...")
    
    client = AsyncPocketOptionClient(SSID, is_demo=is_demo, enable_logging=False)
    
    connected = await client.connect()
    
    if not connected or not client.is_connected:
        write_log("ERROR: Failed to connect to PocketOption")
        sys.exit(1)
    
    write_log("Connection established successfully")
    await asyncio.sleep(1)  # Give connection time to initialize
    
    try:
        balance = await client.get_balance()
        balance_str = f"{balance.balance} {balance.currency}"
        print(balance_str)
        write_log(f"Balance retrieved: {balance_str}")
    except Exception as e:
        write_log(f"ERROR: Failed to get balance - {e}")
        sys.exit(1)
    finally:
        await client.disconnect()
        write_log("Disconnected from PocketOption")

if __name__ == "__main__":
    asyncio.run(main())

