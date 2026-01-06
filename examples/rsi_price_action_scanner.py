#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI + Price Action Scanner
يبحث عن عملات OTC التي تحقق شروط استراتيجية RSI + Price Action
"""
import json
import asyncio
import os
import sys
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from pocketoptionapi_async import AsyncPocketOptionClient
from pocketoptionapi_async.constants import ASSETS

# إصلاح مشاكل الترميز UTF-8 في التيرمنال
if sys.platform != 'win32':
    # Linux/Mac: تعيين UTF-8 للتيرمنال
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass

# تعيين UTF-8 للـ stdout و stderr
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# تعيين متغيرات البيئة للتيرمنال
os.environ['PYTHONIOENCODING'] = 'utf-8'

def safe_print(*args, **kwargs):
    """طباعة آمنة تدعم UTF-8 مع معالجة الأخطاء"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # في حالة فشل الطباعة بالـ emoji، استبدلها برموز ASCII
        msg = ' '.join(str(arg) for arg in args)
        # استبدال الـ emojis برموز ASCII بسيطة
        replacements = {
            '🔌': '[*]',
            '⏳': '[...]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '📊': '[INFO]',
            '🎯': '[SIGNAL]',
            '📌': '[NEAR]',
            '⚠️': '[WARN]',
            '🔵': '[CALL]',
            '🔴': '[PUT]',
        }
        for emoji, replacement in replacements.items():
            msg = msg.replace(emoji, replacement)
        print(msg, **kwargs)

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """حساب RSI لمجموعة من الأسعار (طريقة Wilder's Smoothing)"""
    if len(prices) < period + 1:
        return None
    
    # حساب التغييرات في الأسعار
    deltas = []
    for i in range(1, len(prices)):
        deltas.append(prices[i] - prices[i-1])
    
    # فصل المكاسب والخسائر للفترة الأولى
    gains = [delta if delta > 0 else 0 for delta in deltas[:period]]
    losses = [-delta if delta < 0 else 0 for delta in deltas[:period]]
    
    # حساب المتوسط الأولي
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        return 100.0
    
    # تطبيق Wilder's Smoothing على باقي القيم
    for i in range(period, len(deltas)):
        change = deltas[i]
        gain = change if change > 0 else 0
        loss = -change if change < 0 else 0
        
        # Wilder's Smoothing: New Avg = ((Period - 1) * Old Avg + New Value) / Period
        avg_gain = ((period - 1) * avg_gain + gain) / period
        avg_loss = ((period - 1) * avg_loss + loss) / period
    
    if avg_loss == 0:
        return 100.0
    
    # حساب RS و RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def is_pin_bar(candle, previous_candle) -> Tuple[bool, str]:
    """
    اكتشاف شمعة Pin Bar
    إرجاع: (هل هي pin bar, نوعها: 'bullish' أو 'bearish')
    """
    body_size = abs(candle.close - candle.open)
    total_range = candle.high - candle.low
    
    if total_range == 0:
        return False, ""
    
    body_ratio = body_size / total_range
    
    # Pin Bar يجب أن يكون الذيل (shadow) أكبر من الجسم
    # Bearish Pin Bar: ذيل علوي طويل
    if candle.open > candle.close:  # شمعة هابطة
        upper_shadow = candle.high - max(candle.open, candle.close)
        lower_shadow = min(candle.open, candle.close) - candle.low
        
        if upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5:
            return True, "bearish"
    
    # Bullish Pin Bar: ذيل سفلي طويل
    if candle.close > candle.open:  # شمعة صاعدة
        upper_shadow = candle.high - max(candle.open, candle.close)
        lower_shadow = min(candle.open, candle.close) - candle.low
        
        if lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
            return True, "bullish"
    
    return False, ""

def is_engulfing(current_candle, previous_candle) -> Tuple[bool, str]:
    """
    اكتشاف شمعة Engulfing
    إرجاع: (هل هي engulfing, نوعها: 'bullish' أو 'bearish')
    """
    # Bullish Engulfing
    if (previous_candle.close < previous_candle.open and  # الشمعة السابقة هابطة
        current_candle.close > current_candle.open and    # الشمعة الحالية صاعدة
        current_candle.open < previous_candle.close and   # الفتح أقل من إغلاق السابقة
        current_candle.close > previous_candle.open):     # الإغلاق أعلى من فتح السابقة
        return True, "bullish"
    
    # Bearish Engulfing
    if (previous_candle.close > previous_candle.open and  # الشمعة السابقة صاعدة
        current_candle.close < current_candle.open and    # الشمعة الحالية هابطة
        current_candle.open > previous_candle.close and   # الفتح أعلى من إغلاق السابقة
        current_candle.close < previous_candle.open):     # الإغلاق أقل من فتح السابقة
        return True, "bearish"
    
    return False, ""

def get_otc_assets() -> List[str]:
    """الحصول على قائمة بجميع عملات OTC"""
    otc_assets = [asset for asset in ASSETS.keys() if asset.endswith("_otc")]
    return otc_assets

async def scan_asset(client: AsyncPocketOptionClient, asset: str, timeframe: str = "1m", show_all: bool = False) -> Optional[Dict]:
    """مسح أصل واحد للبحث عن إشارات"""
    try:
        # الحصول على 50 شمعة (نحتاج على الأقل 15 لحساب RSI)
        candles = await client.get_candles(asset, timeframe, count=50)
        
        if len(candles) < 15:
            return None
        
        # الحصول على آخر شمعة وشمعتين سابقتين
        current_candle = candles[-1]
        previous_candle = candles[-2] if len(candles) > 1 else None
        
        if not previous_candle:
            return None
        
        # استخراج الأسعار (close prices) لحساب RSI
        close_prices = [candle.close for candle in candles]
        rsi = calculate_rsi(close_prices, period=14)
        
        if rsi is None:
            return None
        
        # التحقق من شروط CALL (شراء) - توسيع النطاق قليلاً
        call_signal = False
        call_pattern = ""
        call_near_signal = False  # قريب من الإشارة
        
        if rsi < 35:  # توسيع من 30 إلى 35
            # التحقق من شمعة انعكاس صاعدة
            is_pin, pin_type = is_pin_bar(current_candle, previous_candle)
            is_eng, eng_type = is_engulfing(current_candle, previous_candle)
            
            if rsi < 30:  # في المنطقة القوية
                if is_pin and pin_type == "bullish":
                    call_signal = True
                    call_pattern = "Pin Bar (Bullish)"
                elif is_eng and eng_type == "bullish":
                    call_signal = True
                    call_pattern = "Engulfing (Bullish)"
            elif rsi < 35:  # قريب من الإشارة
                call_near_signal = True
                if is_pin and pin_type == "bullish":
                    call_pattern = "Pin Bar (Bullish) - قريب"
                elif is_eng and eng_type == "bullish":
                    call_pattern = "Engulfing (Bullish) - قريب"
        
        # التحقق من شروط PUT (بيع) - توسيع النطاق قليلاً
        put_signal = False
        put_pattern = ""
        put_near_signal = False  # قريب من الإشارة
        
        if rsi > 65:  # توسيع من 70 إلى 65
            # التحقق من شمعة انعكاس هابطة
            is_pin, pin_type = is_pin_bar(current_candle, previous_candle)
            is_eng, eng_type = is_engulfing(current_candle, previous_candle)
            
            if rsi > 70:  # في المنطقة القوية
                if is_pin and pin_type == "bearish":
                    put_signal = True
                    put_pattern = "Pin Bar (Bearish)"
                elif is_eng and eng_type == "bearish":
                    put_signal = True
                    put_pattern = "Engulfing (Bearish)"
            elif rsi > 65:  # قريب من الإشارة
                put_near_signal = True
                if is_pin and pin_type == "bearish":
                    put_pattern = "Pin Bar (Bearish) - قريب"
                elif is_eng and eng_type == "bearish":
                    put_pattern = "Engulfing (Bearish) - قريب"
        
        # إرجاع النتيجة إذا كانت هناك إشارة أو إذا طلبنا عرض الكل
        if call_signal or put_signal or call_near_signal or put_near_signal or show_all:
            return {
                "asset": asset,
                "rsi": round(rsi, 2),
                "current_price": current_candle.close,
                "call_signal": call_signal,
                "call_pattern": call_pattern,
                "call_near_signal": call_near_signal,
                "put_signal": put_signal,
                "put_pattern": put_pattern,
                "put_near_signal": put_near_signal,
                "timeframe": timeframe
            }
        
        return None
        
    except Exception as e:
        # طباعة الأخطاء للمساعدة في التصحيح
        return None

async def main():
    # تحميل متغيرات البيئة
    load_dotenv()
    
    # الحصول على SSID من ملف .env
    SSID = os.getenv("POCKETOPTION_SSID")
    
    if not SSID:
        print("ERROR: POCKETOPTION_SSID not found in .env file.", file=sys.stderr)
        sys.exit(1)
    
    # الكشف التلقائي عن نوع الحساب من SSID
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
    
    # الاتصال بالعميل
    safe_print("🔌 جارٍ الاتصال بـ PocketOption...")
    
    client = AsyncPocketOptionClient(SSID, is_demo=is_demo, enable_logging=True)  # تفعيل اللوغات للمساعدة في التصحيح
    
    try:
        connected = await client.connect()
        
        # انتظار إضافي للتأكد من اكتمال الاتصال
        if connected:
            print("⏳ انتظار اكتمال الاتصال...")
            await asyncio.sleep(2)  # زيادة وقت الانتظار
        
        # التحقق من الاتصال مرة أخرى
        if not connected:
            print("❌ ERROR: فشل الاتصال الأولي", file=sys.stderr)
            sys.exit(1)
        
        # التحقق من حالة الاتصال
        max_retries = 5
        for i in range(max_retries):
            if client.is_connected:
                try:
                    print("✅ تم الاتصال بنجاح!")
                except UnicodeEncodeError:
                    print("[OK] تم الاتصال بنجاح!")
                break
            await asyncio.sleep(0.5)
        else:
            try:
                print("❌ ERROR: الاتصال لم يكتمل بعد عدة محاولات", file=sys.stderr)
                print("   تأكد من أن SSID صحيح وغير منتهي الصلاحية", file=sys.stderr)
            except UnicodeEncodeError:
                print("[ERROR] الاتصال لم يكتمل بعد عدة محاولات", file=sys.stderr)
                print("   تاكد من ان SSID صحيح وغير منتهي الصلاحية", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        try:
            print(f"❌ ERROR: فشل الاتصال - {e}", file=sys.stderr)
            print("   تحقق من:", file=sys.stderr)
            print("   1. SSID صحيح في ملف .env", file=sys.stderr)
            print("   2. SSID غير منتهي الصلاحية", file=sys.stderr)
            print("   3. الاتصال بالإنترنت يعمل", file=sys.stderr)
        except UnicodeEncodeError:
            print(f"[ERROR] فشل الاتصال - {e}", file=sys.stderr)
            print("   تحقق من:", file=sys.stderr)
            print("   1. SSID صحيح في ملف .env", file=sys.stderr)
            print("   2. SSID غير منتهي الصلاحية", file=sys.stderr)
            print("   3. الاتصال بالانترنت يعمل", file=sys.stderr)
        sys.exit(1)
    
    try:
        # الحصول على قائمة عملات OTC
        otc_assets = get_otc_assets()
        
        safe_print(f"جارٍ المسح على {len(otc_assets)} عملة OTC...")
        safe_print("=" * 80)
        
        # المسح على إطار زمني 1m
        print("\n📊 إطار زمني: 1 دقيقة")
        print("-" * 80)
        
        signals_1m = []
        near_signals_1m = []
        all_rsi_1m = []
        
        print(f"جارٍ فحص {len(otc_assets)} عملة...")
        scanned = 0
        
        for asset in otc_assets:
            scanned += 1
            if scanned % 10 == 0:
                print(f"  تم فحص {scanned}/{len(otc_assets)} عملة...")
            
            result = await scan_asset(client, asset, "1m", show_all=False)
            if result:
                if result.get('call_signal') or result.get('put_signal'):
                    signals_1m.append(result)
                elif result.get('call_near_signal') or result.get('put_near_signal'):
                    near_signals_1m.append(result)
                else:
                    # عرض RSI للأصول القريبة من الإشارات
                    if result['rsi'] < 35 or result['rsi'] > 65:
                        all_rsi_1m.append(result)
            await asyncio.sleep(0.05)  # تجنب الضغط على الخادم
        
        # عرض نتائج 1m - إشارات قوية أولاً
        if signals_1m:
            print(f"\n✅ تم العثور على {len(signals_1m)} إشارة قوية:")
            for signal in signals_1m:
                print(f"\n🎯 {signal['asset']}")
                print(f"   RSI: {signal['rsi']}")
                print(f"   السعر الحالي: {signal['current_price']}")
                if signal['call_signal']:
                    print(f"   🔵 إشارة شراء (CALL) - {signal['call_pattern']}")
                if signal['put_signal']:
                    print(f"   🔴 إشارة بيع (PUT) - {signal['put_pattern']}")
        else:
            print("\n❌ لم يتم العثور على إشارات قوية في الإطار الزمني 1m")
        
        # عرض الإشارات القريبة
        if near_signals_1m:
            print(f"\n⚠️  تم العثور على {len(near_signals_1m)} إشارة قريبة:")
            for signal in near_signals_1m:
                print(f"\n📌 {signal['asset']}")
                print(f"   RSI: {signal['rsi']}")
                print(f"   السعر الحالي: {signal['current_price']}")
                if signal.get('call_pattern'):
                    print(f"   🔵 {signal['call_pattern']}")
                if signal.get('put_pattern'):
                    print(f"   🔴 {signal['put_pattern']}")
        
        # عرض RSI للعملات القريبة من المناطق
        if all_rsi_1m and not signals_1m and not near_signals_1m:
            print(f"\n📊 العملات القريبة من مناطق الإشارات (RSI < 35 أو > 65):")
            for signal in sorted(all_rsi_1m, key=lambda x: abs(x['rsi'] - 50))[:10]:  # أول 10
                print(f"   {signal['asset']}: RSI = {signal['rsi']}")
        
        # المسح على إطار زمني 5m
        print("\n\n📊 إطار زمني: 5 دقائق")
        print("-" * 80)
        
        signals_5m = []
        near_signals_5m = []
        all_rsi_5m = []
        
        print(f"\nجارٍ فحص {len(otc_assets)} عملة...")
        scanned = 0
        
        for asset in otc_assets:
            scanned += 1
            if scanned % 10 == 0:
                print(f"  تم فحص {scanned}/{len(otc_assets)} عملة...")
            
            result = await scan_asset(client, asset, "5m", show_all=False)
            if result:
                if result.get('call_signal') or result.get('put_signal'):
                    signals_5m.append(result)
                elif result.get('call_near_signal') or result.get('put_near_signal'):
                    near_signals_5m.append(result)
                else:
                    # عرض RSI للأصول القريبة من الإشارات
                    if result['rsi'] < 35 or result['rsi'] > 65:
                        all_rsi_5m.append(result)
            await asyncio.sleep(0.05)  # تجنب الضغط على الخادم
        
        # عرض نتائج 5m - إشارات قوية أولاً
        if signals_5m:
            print(f"\n✅ تم العثور على {len(signals_5m)} إشارة قوية:")
            for signal in signals_5m:
                print(f"\n🎯 {signal['asset']}")
                print(f"   RSI: {signal['rsi']}")
                print(f"   السعر الحالي: {signal['current_price']}")
                if signal['call_signal']:
                    print(f"   🔵 إشارة شراء (CALL) - {signal['call_pattern']}")
                if signal['put_signal']:
                    print(f"   🔴 إشارة بيع (PUT) - {signal['put_pattern']}")
        else:
            print("\n❌ لم يتم العثور على إشارات قوية في الإطار الزمني 5m")
        
        # عرض الإشارات القريبة
        if near_signals_5m:
            print(f"\n⚠️  تم العثور على {len(near_signals_5m)} إشارة قريبة:")
            for signal in near_signals_5m:
                print(f"\n📌 {signal['asset']}")
                print(f"   RSI: {signal['rsi']}")
                print(f"   السعر الحالي: {signal['current_price']}")
                if signal.get('call_pattern'):
                    print(f"   🔵 {signal['call_pattern']}")
                if signal.get('put_pattern'):
                    print(f"   🔴 {signal['put_pattern']}")
        
        # عرض RSI للعملات القريبة من المناطق
        if all_rsi_5m and not signals_5m and not near_signals_5m:
            print(f"\n📊 العملات القريبة من مناطق الإشارات (RSI < 35 أو > 65):")
            for signal in sorted(all_rsi_5m, key=lambda x: abs(x['rsi'] - 50))[:10]:  # أول 10
                print(f"   {signal['asset']}: RSI = {signal['rsi']}")
        
        print("\n" + "=" * 80)
        total_strong = len(signals_1m) + len(signals_5m)
        total_near = len(near_signals_1m) + len(near_signals_5m)
        print(f"✅ انتهى المسح:")
        print(f"   • إشارات قوية: {len(signals_1m)} في 1m + {len(signals_5m)} في 5m = {total_strong} إجمالي")
        print(f"   • إشارات قريبة: {len(near_signals_1m)} في 1m + {len(near_signals_5m)} في 5m = {total_near} إجمالي")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

