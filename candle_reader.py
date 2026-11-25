"""
candle_reader.py - Advanced Candlestick Pattern Reader & SMC Analyzer.
Detects every single candlestick pattern and explains its market psychology in simple Indonesian.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

class CandleReader:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 20:
            return df
        
        df = df.copy()
        # EMAs
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=min(200, len(df)), adjust=False).mean()

        # ATR 14
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(window=14).mean().bfill()

        # RSI 14
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi14"] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ADX 14 (Trend Strength)
        up_move = df["high"].diff()
        down_move = -df["low"].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_adx = tr.replace(0, 1e-9)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_adx)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_adx)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
        df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        return df

    @staticmethod
    def read_candle_pattern(df: pd.DataFrame, idx: int = -1) -> Dict[str, Any]:
        """
        Reads any candlestick pattern for the given candle index.
        Returns pattern name, direction, confidence score, and Indonesian explanation.
        """
        if df.empty or len(df) < 3:
            return {
                "pattern": "Candle Standar",
                "bias": "NEUTRAL",
                "strength": 50,
                "description": "Data candlestick belum cukup untuk dianalisis.",
                "candle_type": "Neutral"
            }
        
        c = df.iloc[idx]
        p1 = df.iloc[idx - 1]
        p2 = df.iloc[idx - 2] if len(df) >= 3 else p1

        open_p = c["open"]
        close_p = c["close"]
        high_p = c["high"]
        low_p = c["low"]

        body = abs(close_p - open_p)
        candle_range = high_p - low_p
        if candle_range <= 0.001:
            candle_range = 0.001

        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p
        is_bullish = close_p >= open_p
        is_bearish = close_p < open_p

        body_ratio = body / candle_range
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        # Pattern evaluations
        # 1. Doji
        if body_ratio <= 0.10:
            if lower_wick_ratio >= 0.65:
                return {
                    "pattern": "DRAGONFLY DOJI",
                    "bias": "BULLISH",
                    "strength": 80,
                    "candle_type": "Reversal Naik",
                    "description": "Dragonfly Doji: Penjual sempat menekan turun sangat jauh, tapi pembeli membalas dan mengembalikan harga ke puncak. Indikasi pantulan naik kuat."
                }
            elif upper_wick_ratio >= 0.65:
                return {
                    "pattern": "GRAVESTONE DOJI",
                    "bias": "BEARISH",
                    "strength": 80,
                    "candle_type": "Reversal Turun",
                    "description": "Gravestone Doji: Pembeli sempat mendongkrak harga ke atas, namun penjual membanting harga kembali ke bawah. Indikasi pembalikan turun kuat."
                }
            else:
                return {
                    "pattern": "STANDARD DOJI",
                    "bias": "NEUTRAL",
                    "strength": 60,
                    "candle_type": "Ragu-ragu (Indecision)",
                    "description": "Doji: Keseimbangan antara pembeli dan penjual. Pasar sedang berkonsolidasi menunggu dorongan tren berikutnya."
                }

        # 2. Pinbar / Hammer / Shooting Star
        if lower_wick_ratio >= 0.60 and body_ratio <= 0.35:
            return {
                "pattern": "BULLISH PINBAR / HAMMER",
                "bias": "BULLISH",
                "strength": 88,
                "candle_type": "Rejection Bawah (Buy Signal)",
                "description": "Bullish Pinbar / Hammer: Ekor bawah sangat panjang menunjukkan penolakan keras harga murah. Tekanan beli institusi masuk menolak penurunan."
            }

        if upper_wick_ratio >= 0.60 and body_ratio <= 0.35:
            return {
                "pattern": "BEARISH SHOOTING STAR",
                "bias": "BEARISH",
                "strength": 88,
                "candle_type": "Rejection Atas (Sell Signal)",
                "description": "Bearish Shooting Star: Ekor atas panjang menandakan penolakan tajam di area resistance. Penjual menolak harga tinggi dan siap menekan turun."
            }

        # 3. Engulfing
        p1_body = abs(p1["close"] - p1["open"])
        if is_bullish and p1["close"] < p1["open"] and close_p > p1["open"] and open_p <= p1["close"]:
            return {
                "pattern": "BULLISH ENGULFING",
                "bias": "BULLISH",
                "strength": 92,
                "candle_type": "Bullish Dominance (Kuat Naik)",
                "description": "Bullish Engulfing: Candle hijau tebal menelan utuh candle merah sebelumnya. Pembeli mendominasi penuh pasar secara agresif."
            }

        if is_bearish and p1["close"] > p1["open"] and close_p < p1["open"] and open_p >= p1["close"]:
            return {
                "pattern": "BEARISH ENGULFING",
                "bias": "BEARISH",
                "strength": 92,
                "candle_type": "Bearish Dominance (Kuat Turun)",
                "description": "Bearish Engulfing: Candle merah tebal menelan tuntas candle hijau sebelumnya. Penjual mengambil alih pasar secara dominan."
            }

        # 4. Morning Star & Evening Star
        if len(df) >= 3:
            # Morning Star: Bearish candle -> Small body candle -> Strong Bullish candle
            p2_bearish = p2["close"] < p2["open"]
            p1_small = abs(p1["close"] - p1["open"]) / max(p1["high"] - p1["low"], 0.001) < 0.35
            if p2_bearish and p1_small and is_bullish and close_p > (p2["open"] + p2["close"]) / 2:
                return {
                    "pattern": "MORNING STAR",
                    "bias": "BULLISH",
                    "strength": 94,
                    "candle_type": "Major Bullish Reversal",
                    "description": "Morning Star (Bintang Fajar): Pola 3 candle pembalikan arah dari turun menjadi lonjakan naik yang sangat akurat."
                }

            # Evening Star: Bullish candle -> Small body candle -> Strong Bearish candle
            p2_bullish = p2["close"] > p2["open"]
            if p2_bullish and p1_small and is_bearish and close_p < (p2["open"] + p2["close"]) / 2:
                return {
                    "pattern": "EVENING STAR",
                    "bias": "BEARISH",
                    "strength": 94,
                    "candle_type": "Major Bearish Reversal",
                    "description": "Evening Star (Bintang Senja): Pola 3 candle pembalikan arah dari kenaikan menjadi penurunan tajam."
                }

        # 5. Marubozu (Strong Momentum)
        if body_ratio >= 0.85:
            if is_bullish:
                return {
                    "pattern": "BULLISH MARUBOZU",
                    "bias": "BULLISH",
                    "strength": 85,
                    "candle_type": "Momentum Beli Masif",
                    "description": "Bullish Marubozu: Badan candle hijau penuh hampir tanpa sumbu. Pembeli memegang kendali penuh dari pembukaan hingga penutupan."
                }
            else:
                return {
                    "pattern": "BEARISH MARUBOZU",
                    "bias": "BEARISH",
                    "strength": 85,
                    "candle_type": "Momentum Jual Masif",
                    "description": "Bearish Marubozu: Badan candle merah tebal tanpa sumbu. Penjual menekan harga tiada henti sepanjang sesi."
                }

        # 6. Rejection Wicks
        if lower_wick_ratio >= 0.45:
            return {
                "pattern": "LOWER WICK REJECTION",
                "bias": "BULLISH",
                "strength": 72,
                "candle_type": "Penolakan Harga Rendah",
                "description": "Lower Wick Rejection: Ekor bawah panjang menandakan ada minat beli yang kuat di harga bawah (zona support)."
            }

        if upper_wick_ratio >= 0.45:
            return {
                "pattern": "UPPER WICK REJECTION",
                "bias": "BEARISH",
                "strength": 72,
                "candle_type": "Penolakan Harga Tinggi",
                "description": "Upper Wick Rejection: Ekor atas panjang menandakan pasokan penjual menahan harga di area resistance atas."
            }

        # 7. Default Candle
        if is_bullish:
            return {
                "pattern": "BULLISH CANDLE",
                "bias": "BULLISH",
                "strength": 55,
                "candle_type": "Normal Bullish",
                "description": "Candle hijau standar bergerak naik secara proporsional."
            }
        else:
            return {
                "pattern": "BEARISH CANDLE",
                "bias": "BEARISH",
                "strength": 55,
                "candle_type": "Normal Bearish",
                "description": "Candle merah standar bergerak turun secara proporsional."
            }

    @staticmethod
    def analyze_smart_money(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes Order Blocks (OB), Fair Value Gap (FVG), Premium/Discount Zones.
        """
        if df.empty or len(df) < 15:
            return {
                "zone": "NEUTRAL",
                "order_block_bull": None,
                "order_block_bear": None,
                "fvg": None,
                "equilibrium": 0.0
            }

        recent = df.iloc[-30:] if len(df) >= 30 else df
        highest = recent["high"].max()
        lowest = recent["low"].min()
        curr_price = df["close"].iloc[-1]
        rng = highest - lowest if (highest - lowest) > 0 else 1.0
        
        # Equilibrium 50%
        equilibrium = lowest + (rng * 0.50)
        price_pos_pct = ((curr_price - lowest) / rng) * 100.0

        if price_pos_pct < 40:
            zone_desc = "ZONA DISCOUNT (Murah - Ideal untuk Mencari Peluang BUY)"
            zone_type = "DISCOUNT"
        elif price_pos_pct > 60:
            zone_desc = "ZONA PREMIUM (Mahal - Ideal untuk Mencari Peluang SELL)"
            zone_type = "PREMIUM"
        else:
            zone_desc = "ZONA EQUILIBRIUM (Netral / 50% Fair Value)"
            zone_type = "EQUILIBRIUM"

        # Order block detection
        bullish_ob = None
        bearish_ob = None
        for i in range(len(df) - 4, len(df) - 1):
            # Bullish OB: last down candle before strong up impulse
            if df["close"].iloc[i] < df["open"].iloc[i] and df["close"].iloc[i+1] > df["high"].iloc[i]:
                bullish_ob = {
                    "high": round(df["high"].iloc[i], 2),
                    "low": round(df["low"].iloc[i], 2),
                    "type": "Demand Zone (Area Beli Smart Money)"
                }
            # Bearish OB: last up candle before strong down impulse
            if df["close"].iloc[i] > df["open"].iloc[i] and df["close"].iloc[i+1] < df["low"].iloc[i]:
                bearish_ob = {
                    "high": round(df["high"].iloc[i], 2),
                    "low": round(df["low"].iloc[i], 2),
                    "type": "Supply Zone (Area Jual Smart Money)"
                }

        # Fair Value Gap (FVG)
        fvg = None
        if len(df) >= 3:
            c1 = df.iloc[-3]
            c2 = df.iloc[-2]
            c3 = df.iloc[-1]
            # Bullish FVG: Low of c3 > High of c1
            if c3["low"] > c1["high"]:
                fvg = {
                    "type": "BULLISH FVG (Imbalance Bawah)",
                    "top": round(c3["low"], 2),
                    "bottom": round(c1["high"], 2),
                    "note": "Celah harga naik agresif yang berpotensi ditest ulang sebagai support."
                }
            elif c1["low"] > c3["high"]:
                fvg = {
                    "type": "BEARISH FVG (Imbalance Atas)",
                    "top": round(c1["low"], 2),
                    "bottom": round(c3["high"], 2),
                    "note": "Celah harga turun agresif yang berpotensi ditest ulang sebagai resistance."
                }

        return {
            "zone_type": zone_type,
            "zone_desc": zone_desc,
            "price_pos_pct": round(price_pos_pct, 1),
            "equilibrium": round(equilibrium, 2),
            "range_high": round(highest, 2),
            "range_low": round(lowest, 2),
            "bullish_ob": bullish_ob,
            "bearish_ob": bearish_ob,
            "fvg": fvg
        }

if __name__ == "__main__":
    from market_data import MarketDataFetcher
    fetcher = MarketDataFetcher()
    data = fetcher.get_multi_timeframe_data()
    m5 = CandleReader.calculate_indicators(data["m5"])
    pattern = CandleReader.read_candle_pattern(m5)
    smc = CandleReader.analyze_smart_money(m5)
    print("Detected Pattern M5:", pattern)
    print("SMC Zone:", smc["zone_desc"])
