import json
import asyncio
from pocketoptionapi_async import AsyncPocketOptionClient

async def main():
    SSID = input("Enter your SSID: ")
    
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
                print(f"Detected account type: {'Demo' if is_demo else 'Real'}")
        except Exception as e:
            print(f"Warning: Could not parse SSID demo status, using default (demo=True): {e}")
    
    client = AsyncPocketOptionClient(SSID, is_demo=is_demo, enable_logging=True)
    
    print("Connecting to PocketOption...")
    connected = await client.connect()
    
    if not connected:
        print("ERROR: Failed to connect to PocketOption. Please check your SSID and try again.")
        return
    
    if not client.is_connected:
        print("ERROR: Connection was not established successfully.")
        return
    
    print("Connection successful! Waiting a moment for initialization...")
    await asyncio.sleep(1)  # Give connection time to initialize
    
    try:
        print("Fetching balance...")
        balance = await client.get_balance()
        print(f"Your balance is: {balance.balance}, currency: {balance.currency}")
    except Exception as e:
        print(f"ERROR: Failed to get balance: {e}")
        print(f"Connection status: {client.is_connected}")
    finally:
        await client.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())