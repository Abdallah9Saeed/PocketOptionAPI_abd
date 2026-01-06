#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import asyncio
import os
import sys
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from pocketoptionapi_async import AsyncPocketOptionClient
from pocketoptionapi_async.constants import ASSETS

# ==========================
# Utils
# ==========================

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas[:period]]
    losses = [-d if d < 0 else 0 for d in deltas[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for d in deltas[period:]:
        avg_gain = ((period - 1) * avg_gain + max(d, 0)) / period
        avg_loss = ((period - 1) * avg_loss + max(-d, 0)) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_ema(prices: List[float], period: int = 50) -> Optional[float]:
    if len(prices) < period:
        return None

    ema = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)

    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def is_strong_candle(candle) -> bool:
    body = abs(candle.close - candle.open)
    total = candle.high - candle.low
    return total > 0 and (body / total) > 0.7


def has_trend_context(candles, direction: str) -> bool:
    if len(candles) < 4:
        return False

    c1 = candles[-2]
    c2 = candles[-3]

    if direction == "bullish":
        return c1.close < c1.open and c2.close < c2.open

    if direction == "bearish":
        return c1.close > c1.open and c2.close > c2.open

    return False


def is_pin_bar(candle) -> Optional[str]:
    body = abs(candle.close - candle.open)
    total = candle.high - candle.low
    if total == 0:
        return None

    upper = candle.high - max(candle.open, candle.close)
    lower = min(candle.open, candle.close) - candle.low

    if lower > body * 2 and upper < body * 0.5:
        return "bullish"

    if upper > body * 2 and lower < body * 0.5:
        return "bearish"

    return None


def is_engulfing(curr, prev) -> Optional[str]:
    if prev.close < prev.open and curr.close > curr.open:
        if curr.open < prev.close and curr.close > prev.open:
            return "bullish"

    if prev.close > prev.open and curr.close < curr.open:
        if curr.open > prev.close and curr.close < prev.open:
            return "bearish"

    return None


def get_otc_assets():
    return [a for a in ASSETS.keys() if a.endswith("_otc")]

# ==========================
# Scanner
# ==========================

async def scan_asset(client, asset: str, timeframe="5m") -> Optional[Dict]:
    candles = await client.get_candles(asset, timeframe, count=70)
    if len(candles) < 60:
        return None

    current = candles[-1]
    previous = candles[-2]

    # فلتر الشمعة القوية
    if is_strong_candle(current):
        return None

    closes = [c.close for c in candles]

    rsi = calculate_rsi(closes)
    prev_rsi = calculate_rsi(closes[:-1])
    ema50 = calculate_ema(closes)

    if rsi is None or prev_rsi is None or ema50 is None:
        return None

    # ===== ATR Filter =====
    atr = calculate_atr(candles)
    if atr is None:
        return None

    candle_range = current.high - current.low
    if candle_range == 0 or atr < candle_range * 0.8:
        return None

    # ===== Close Position Filter =====
    close_pos = (current.close - current.low) / candle_range

    score = 0
    signal = None
    pattern = ""

    pin = is_pin_bar(current)
    engulf = is_engulfing(current, previous)

    # ================= CALL =================
    if rsi <= 35:

        # RSI must start rising
        if rsi <= prev_rsi:
            return None

        # Close must be strong
        if close_pos < 0.7:
            return None

        # RSI score
        score += 30 if rsi <= 30 else 15

        # EMA filter
        if current.close < ema50:
            score += 20
        else:
            return None

        # Context filter
        if has_trend_context(candles, "bullish"):
            score += 20
        else:
            return None

        # Price Action
        if pin == "bullish":
            score += 20
            pattern = "Bullish Pin Bar"
        elif engulf == "bullish":
            score += 30
            pattern = "Bullish Engulfing"
        else:
            return None

        if score >= 75:
            signal = "CALL"

    # ================= PUT =================
    elif rsi >= 65:
        score = 0

        # RSI must start falling
        if rsi >= prev_rsi:
            return None

        # Close must be strong
        if close_pos > 0.3:
            return None

        # RSI score
        score += 30 if rsi >= 70 else 15

        # EMA filter
        if current.close > ema50:
            score += 20
        else:
            return None

        # Context filter
        if has_trend_context(candles, "bearish"):
            score += 20
        else:
            return None

        # Price Action
        if pin == "bearish":
            score += 20
            pattern = "Bearish Pin Bar"
        elif engulf == "bearish":
            score += 30
            pattern = "Bearish Engulfing"
        else:
            return None

        if score >= 75:
            signal = "PUT"

    if signal:
        return {
            "asset": asset,
            "signal": signal,
            "score": score,
            "pattern": pattern,
            "rsi": round(rsi, 2),
            "price": current.close,
            "timeframe": timeframe
        }

    return None

# ==========================
# MAIN
# ==========================

# إصلاح مشاكل الترميز UTF-8 في التيرمنال
if sys.platform != 'win32':
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

os.environ['PYTHONIOENCODING'] = 'utf-8'

def safe_print(*args, **kwargs):
    """طباعة آمنة تدعم UTF-8"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        msg = ' '.join(str(arg) for arg in args)
        replacements = {
            '🎯': '[SIGNAL]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '📊': '[INFO]',
        }
        for emoji, replacement in replacements.items():
            msg = msg.replace(emoji, replacement)
        print(msg, **kwargs)

async def main():
    load_dotenv()
    SSID = os.getenv("POCKETOPTION_SSID")
    
    if not SSID:
        safe_print("ERROR: POCKETOPTION_SSID not found in .env file.", file=sys.stderr)
        sys.exit(1)
    
    # Auto-detect demo status from SSID
    is_demo = True
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
    
    safe_print("Connecting to PocketOption...")
    safe_print(f"Account type: {'Demo' if is_demo else 'Real'}")
    
    client = AsyncPocketOptionClient(SSID, is_demo=is_demo, enable_logging=True)  # Enable logging for debugging
    
    try:
        safe_print("Attempting connection...")
        connected = await client.connect()
        
        if not connected:
            safe_print("ERROR: Connection returned False", file=sys.stderr)
            safe_print("   Check:", file=sys.stderr)
            safe_print("   1. SSID is correct and not expired", file=sys.stderr)
            safe_print("   2. Internet connection is working", file=sys.stderr)
            safe_print("   3. SSID format: 42[\"auth\",{\"session\":\"...\",\"isDemo\":1,\"uid\":...,\"platform\":1}]", file=sys.stderr)
            sys.exit(1)
        
        # Wait for connection to stabilize
        safe_print("Waiting for connection to stabilize...")
        await asyncio.sleep(2)  # Increased wait time
        
        # Verify connection with retries
        max_retries = 5
        for i in range(max_retries):
            if client.is_connected:
                safe_print("✅ Connected successfully!\n")
                break
            await asyncio.sleep(0.5)
        else:
            safe_print("ERROR: Connection verification failed after multiple attempts", file=sys.stderr)
            safe_print(f"   Connection status: {client.is_connected}", file=sys.stderr)
            safe_print("   This usually means:", file=sys.stderr)
            safe_print("   - SSID is expired (get a fresh one from browser DevTools)", file=sys.stderr)
            safe_print("   - SSID format is incorrect", file=sys.stderr)
            safe_print("   - Network/firewall blocking WebSocket connections", file=sys.stderr)
            sys.exit(1)
        
        assets = get_otc_assets()
        safe_print(f"Scanning {len(assets)} OTC assets...\n")
        
        signals_found = []
        scanned = 0
        
        for asset in assets:
            scanned += 1
            if scanned % 20 == 0:
                safe_print(f"  Progress: {scanned}/{len(assets)} assets scanned...")
            
            try:
                res = await scan_asset(client, asset)
                if res:
                    signals_found.append(res)
                    safe_print(
                        f"🎯 {res['asset']} | {res['signal']} | "
                        f"Score: {res['score']} | RSI: {res['rsi']} | {res['pattern']}"
                    )
            except Exception as e:
                # Skip assets that fail
                pass
            
            await asyncio.sleep(0.05)
        
        safe_print(f"\n{'='*60}")
        safe_print(f"Scan completed: {len(signals_found)} signals found from {len(assets)} assets")
        
        # Print summary with scores
        if signals_found:
            safe_print(f"\n📊 Summary of Signals Found:")
            safe_print(f"{'='*60}")
            
            # Sort by score (highest first)
            signals_found.sort(key=lambda x: x['score'], reverse=True)
            
            for i, signal in enumerate(signals_found, 1):
                safe_print(
                    f"{i}. {signal['asset']} | {signal['signal']} | "
                    f"Score: {signal['score']} | RSI: {signal['rsi']} | {signal['pattern']}"
                )
            
            # Statistics
            call_signals = [s for s in signals_found if s['signal'] == 'CALL']
            put_signals = [s for s in signals_found if s['signal'] == 'PUT']
            
            safe_print(f"\n📈 Statistics:")
            safe_print(f"   Total Signals: {len(signals_found)}")
            safe_print(f"   CALL Signals: {len(call_signals)}")
            safe_print(f"   PUT Signals: {len(put_signals)}")
            
            if signals_found:
                avg_score = sum(s['score'] for s in signals_found) / len(signals_found)
                max_score = max(s['score'] for s in signals_found)
                min_score = min(s['score'] for s in signals_found)
                safe_print(f"   Average Score: {avg_score:.2f}")
                safe_print(f"   Highest Score: {max_score}")
                safe_print(f"   Lowest Score: {min_score}")
        else:
            safe_print("\n❌ No signals found matching the criteria.")
        
    except KeyboardInterrupt:
        safe_print("\n\nScan interrupted by user.")
        sys.exit(0)
    except Exception as e:
        safe_print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        safe_print(f"\nTraceback:", file=sys.stderr)
        safe_print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            await client.disconnect()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
