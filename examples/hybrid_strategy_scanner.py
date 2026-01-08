#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Strategy Scanner: Strategy A (Classic) + Strategy B (Enhanced)
فحص هجين: يستخدم الاستراتيجيتين معاً ويعرض فقط العملات التي تحقق الشروط معاً
"""

import json
import asyncio
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection
from pocketoptionapi_async.constants import ASSETS

# ================= CONFIG =================
TRADE_AMOUNT = 1.0        # 1$ فقط
MIN_SCORE = 75            # حد الدخول للاستراتيجية B
RSI_PERIOD = 14
RSI_BUY_MAX = 60          # لا شراء في تشبع شراء
RSI_SELL_MIN = 40         # لا بيع في تشبع بيع
TIMEFRAME = "5m"          # الإطار الزمني
MAX_ASSETS_TO_SHOW = 10   # عرض فقط أول 10 عملات

# ================= UTF-8 Setup =================
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
            '💰': '[TRADE]',
            '🔥': '[MATCH]',
        }
        for emoji, replacement in replacements.items():
            msg = msg.replace(emoji, replacement)
        print(msg, **kwargs)

# ================= UTILS =================

def smart_expiration(period: int) -> int:
    """حساب مدة الصفقة الذكية بناءً على الفريم"""
    if period <= 60:
        return 120   # 1M -> 2M
    elif period <= 120:
        return 180
    elif period <= 300:
        return 300
    return period


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    """حساب EMA"""
    if len(prices) < period:
        return None
    
    ema = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)
    
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    
    return ema


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """حساب RSI"""
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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """حساب ATR"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(period).mean()


def supertrend(df: pd.DataFrame, multiplier: float = 1.3, period: int = 13) -> pd.DataFrame:
    """حساب SuperTrend"""
    atr = calculate_atr(df, period)
    hl_avg = (df['high'] + df['low']) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=str)
    
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 'down'
        else:
            if df['close'].iloc[i] <= st.iloc[i-1]:
                st.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 'down'
            else:
                st.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 'up'
    
    df['ST'] = st
    df['STX'] = direction
    return df


def heikinashi(df: pd.DataFrame) -> pd.DataFrame:
    """تحويل الشموع إلى Heikin Ashi"""
    ha_df = df.copy()
    
    ha_df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    ha_df['open'] = 0.0
    ha_df.iloc[0, ha_df.columns.get_loc('open')] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    
    for i in range(1, len(ha_df)):
        ha_df.iloc[i, ha_df.columns.get_loc('open')] = (
            ha_df.iloc[i-1, ha_df.columns.get_loc('open')] + 
            ha_df.iloc[i-1, ha_df.columns.get_loc('close')]
        ) / 2
    
    ha_df['high'] = df[['high', 'open', 'close']].max(axis=1)
    ha_df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    return ha_df


def crossed_above(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """اكتشاف تقاطع من الأسفل"""
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossed_below(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """اكتشاف تقاطع من الأعلى"""
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def calc_score(row: Dict) -> int:
    """حساب النقاط"""
    score = 0
    # Trend
    score += 30 if row.get('trend_ok', False) else 0
    # Pullback
    score += 20 if row.get('pullback', False) else 0
    # RSI
    score += 15 if row.get('rsi_ok', False) else 0
    # Candle strength
    score += 10 if row.get('good_candle', False) else 0
    return score


def is_good_candle(row: pd.Series) -> bool:
    """التحقق من قوة الشمعة"""
    body = abs(row['close'] - row['open'])
    rng = row['high'] - row['low']
    return rng > 0 and body / rng < 0.7


# ================= STRATEGIES =================

def strategie_classic(pair: str, df: pd.DataFrame, period: int) -> Optional[str]:
    """
    Strategy A (Classic): Heikin Ashi + SuperTrend + EMA
    بدون RSI أو Score (Baseline للمقارنة)
    """
    # Heikin Ashi
    ha = heikinashi(df)
    df[['open', 'close', 'high', 'low']] = ha[['open', 'close', 'high', 'low']]
    
    # Indicators
    df = supertrend(df, 1.3, 13)
    df['ema_fast'] = df['close'].ewm(span=16, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=165, adjust=False).mean()
    
    if len(df) < 2:
        return None
    
    last = df.iloc[-1]
    
    trend_up = last['STX'] == 'up' and last['ema_fast'] > last['ema_slow']
    trend_dn = last['STX'] == 'down' and last['ema_fast'] < last['ema_slow']
    
    pullback_up = crossed_above(df['ST'], df['ema_fast']).iloc[-1] if len(df) > 1 else False
    pullback_dn = crossed_below(df['ST'], df['ema_fast']).iloc[-1] if len(df) > 1 else False
    
    if trend_up and pullback_up:
        return 'call'
    if trend_dn and pullback_dn:
        return 'put'
    return None


def strategie_scored(pair: str, df: pd.DataFrame, period: int) -> Tuple[Optional[str], int]:
    """
    Strategy B (Enhanced): Scored + RSI + Smart Expiration
    Returns: (signal, score) where signal is 'call', 'put', or None
    """
    # Heikin Ashi
    ha = heikinashi(df)
    df[['open', 'close', 'high', 'low']] = ha[['open', 'close', 'high', 'low']]
    
    # Indicators
    df = supertrend(df, 1.3, 13)
    df['ema_fast'] = df['close'].ewm(span=16, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=165, adjust=False).mean()
    
    closes = df['close'].tolist()
    rsi_values = []
    for i in range(len(closes)):
        if i < RSI_PERIOD:
            rsi_values.append(50.0)
        else:
            rsi = calculate_rsi(closes[:i+1], RSI_PERIOD)
            rsi_values.append(rsi if rsi else 50.0)
    df['rsi'] = rsi_values
    
    if len(df) < 2:
        return None, 0
    
    last = df.iloc[-1]
    
    # ===== CONDITIONS =====
    trend_up = last['STX'] == 'up' and last['ema_fast'] > last['ema_slow']
    trend_dn = last['STX'] == 'down' and last['ema_fast'] < last['ema_slow']
    
    pullback_up = crossed_above(df['ST'], df['ema_fast']).iloc[-1] if len(df) > 1 else False
    pullback_dn = crossed_below(df['ST'], df['ema_fast']).iloc[-1] if len(df) > 1 else False
    
    rsi_buy_ok = last['rsi'] < RSI_BUY_MAX
    rsi_sell_ok = last['rsi'] > RSI_SELL_MIN
    
    candle_ok = is_good_candle(last)
    
    # ===== SCORE =====
    buy_score = calc_score({
        'trend_ok': trend_up,
        'pullback': pullback_up,
        'rsi_ok': rsi_buy_ok,
        'good_candle': candle_ok
    })
    
    sell_score = calc_score({
        'trend_ok': trend_dn,
        'pullback': pullback_dn,
        'rsi_ok': rsi_sell_ok,
        'good_candle': candle_ok
    })
    
    # ===== DECISION =====
    if buy_score >= MIN_SCORE:
        return 'call', buy_score
    if sell_score >= MIN_SCORE:
        return 'put', sell_score
    return None, 0


# ================= SCANNER =================

async def scan_asset_hybrid(client: AsyncPocketOptionClient, asset: str, timeframe: str) -> Optional[Dict]:
    """فحص أصل واحد بكلتا الاستراتيجيتين"""
    try:
        # Get candles
        candles = await client.get_candles(asset, timeframe, count=200)
        
        if len(candles) < 170:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume or 0
        } for c in candles])
        
        timeframe_seconds = 300 if timeframe == "5m" else 60
        
        # Run Strategy A (Classic)
        signal_a = strategie_classic(asset, df.copy(), timeframe_seconds)
        
        # Run Strategy B (Enhanced)
        signal_b, score_b = strategie_scored(asset, df.copy(), timeframe_seconds)
        
        # فقط إذا تحقق كلا الاستراتيجيتين معاً
        if signal_a and signal_b and signal_a == signal_b:
            return {
                'asset': asset,
                'signal': signal_a,  # نفس الإشارة في كلا الاستراتيجيتين
                'score': score_b,
                'strategy_a': '✅',
                'strategy_b': '✅',
                'price': df.iloc[-1]['close']
            }
        
        return None
        
    except Exception as e:
        return None


async def main():
    """الدالة الرئيسية"""
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
    
    client = AsyncPocketOptionClient(SSID, is_demo=is_demo, enable_logging=False)
    
    connected = await client.connect()
    
    if not connected or not client.is_connected:
        safe_print("ERROR: Failed to connect to PocketOption.", file=sys.stderr)
        sys.exit(1)
    
    await asyncio.sleep(1)  # Give connection time to initialize
    
    safe_print("✅ Connected successfully!\n")
    
    try:
        # Get OTC assets
        otc_assets = [a for a in ASSETS.keys() if a.endswith("_otc")]
        
        safe_print(f"📊 Scanning {len(otc_assets)} OTC assets...")
        safe_print(f"   Looking for assets that match BOTH strategies:")
        safe_print(f"   • Strategy A (Classic): Heikin Ashi + SuperTrend + EMA")
        safe_print(f"   • Strategy B (Enhanced): + RSI + Score >= {MIN_SCORE}")
        safe_print(f"   • Showing top {MAX_ASSETS_TO_SHOW} matches\n")
        safe_print("=" * 80)
        
        matches = []
        scanned = 0
        
        for asset in otc_assets:
            scanned += 1
            if scanned % 20 == 0:
                safe_print(f"  Progress: {scanned}/{len(otc_assets)} assets scanned...")
            
            result = await scan_asset_hybrid(client, asset, TIMEFRAME)
            if result:
                matches.append(result)
            
            await asyncio.sleep(0.05)
        
        # Sort by score (highest first) and limit to MAX_ASSETS_TO_SHOW
        matches.sort(key=lambda x: x['score'], reverse=True)
        matches = matches[:MAX_ASSETS_TO_SHOW]
        
        # Display results
        if matches:
            safe_print(f"\n🔥 Found {len(matches)} assets matching BOTH strategies:\n")
            
            for i, match in enumerate(matches, 1):
                safe_print(f"{i}. 🎯 {match['asset']}")
                safe_print(f"   Signal: {match['signal'].upper()}")
                safe_print(f"   Score (Strategy B): {match['score']}")
                safe_print(f"   Strategy A: {match['strategy_a']} | Strategy B: {match['strategy_b']}")
                safe_print(f"   Current Price: {match['price']:.5f}")
                safe_print(f"   Smart Expiration: {smart_expiration(300)}s")
                safe_print()
        else:
            safe_print(f"\n❌ No assets found matching both strategies.")
            safe_print(f"   Try adjusting MIN_SCORE or check market conditions.")
        
        safe_print("=" * 80)
        safe_print(f"✅ Scan completed: {scanned} assets scanned, {len(matches)} matches found")
        
    except KeyboardInterrupt:
        safe_print("\n\n⏹️  Scan interrupted by user.")
        sys.exit(0)
    except Exception as e:
        safe_print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        safe_print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
