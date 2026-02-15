#!/usr/bin/env python3
"""
蔡森老师经典技术分析系统 - Cai Sen Technical Analysis System
应用蔡森老师的经典技术分析方法分析加密货币走势

蔡森技术分析核心方法:
1. 趋势分析 - 趋势线、通道线
2. 均线系统 - MA20, MA60, MA120 多空判断
3. 支撑阻力 - 关键价位识别
4. K线形态 - 经典K线组合
5. 量价关系 - 成交量确认
6. 技术指标 - MACD, RSI, KDJ
7. 斐波那契 - 回调与扩展
8. 波浪理论 - 艾略特波浪分析
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

ALPHA_VANTAGE_API_KEY = "IUO07N60XUPUHNTL"
TWELVE_DATA_API_KEY = "f5491ce160e64101a960e19eb8363f38"
FINNHUB_API_KEY = "d6476nhr01ql6dj2ekegd6476nhr01ql6dj2ekf0"


def sma(series, length):
    return series.rolling(window=length).mean()


class CaiSenTechnicalAnalysis:
    """蔡森技术分析系统"""

    def __init__(self):
        self.results = {}

    def fetch_crypto_data_alphavantage(
        self, symbol: str, market: str = "USD"
    ) -> pd.DataFrame:
        """使用Alpha Vantage获取加密货币数据"""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": market,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if "Time Series (Digital Currency Daily)" not in data:
                print(
                    f"Alpha Vantage API Error: {data.get('Note', data.get('Error Message', 'Unknown error'))}"
                )
                return None

            ts_data = data["Time Series (Digital Currency Daily)"]
            df = pd.DataFrame.from_dict(ts_data, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            df.columns = [
                col.split(". ")[1] if ". " in col else col for col in df.columns
            ]
            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])

            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def fetch_crypto_data_twelvedata(
        self, symbol: str, interval: str = "1day", outputsize: int = 365
    ) -> pd.DataFrame:
        """使用Twelve Data获取加密货币数据"""
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": f"{symbol}/USD",
            "interval": interval,
            "outputsize": outputsize,
            "apikey": TWELVE_DATA_API_KEY,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if "status" in data and data["status"] == "error":
                print(f"Twelve Data API Error: {data.get('message', 'Unknown error')}")
                return None

            if "values" not in data:
                print(f"Twelve Data API Error: No data returned")
                return None

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()

            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])

            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"])
            else:
                df["volume"] = 0

            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

            return df[["Open", "High", "Low", "Close", "Volume"]].dropna(
                subset=["Open", "High", "Low", "Close"]
            )

        except Exception as e:
            print(f"Error fetching data from Twelve Data: {e}")
            return None

    def calculate_ma_system(self, df: pd.DataFrame) -> Dict:
        """蔡森均线系统分析 - 核心多空判断"""
        df = df.copy()

        df["MA5"] = sma(df["Close"], 5)
        df["MA10"] = sma(df["Close"], 10)
        df["MA20"] = sma(df["Close"], 20)
        df["MA60"] = sma(df["Close"], 60)
        df["MA120"] = sma(df["Close"], 120)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        ma_status = {
            "current_price": float(last["Close"]),
            "MA5": float(last["MA5"]) if pd.notna(last["MA5"]) else None,
            "MA10": float(last["MA10"]) if pd.notna(last["MA10"]) else None,
            "MA20": float(last["MA20"]) if pd.notna(last["MA20"]) else None,
            "MA60": float(last["MA60"]) if pd.notna(last["MA60"]) else None,
            "MA120": float(last["MA120"]) if pd.notna(last["MA120"]) else None,
        }

        if last["Close"] > last["MA5"] > last["MA10"] > last["MA20"]:
            ma_status["short_trend"] = "强势多头排列"
            ma_status["short_signal"] = "买入"
        elif last["MA5"] > last["MA10"] > last["MA20"]:
            ma_status["short_trend"] = "多头排列"
            ma_status["short_signal"] = "持有"
        elif last["Close"] < last["MA5"] < last["MA10"] < last["MA20"]:
            ma_status["short_trend"] = "强势空头排列"
            ma_status["short_signal"] = "卖出"
        elif last["MA5"] < last["MA10"] < last["MA20"]:
            ma_status["short_trend"] = "空头排列"
            ma_status["short_signal"] = "观望"
        else:
            ma_status["short_trend"] = "均线纠缠"
            ma_status["short_signal"] = "等待明确"

        if last["MA20"] > last["MA60"] > last["MA120"]:
            ma_status["mid_trend"] = "中期多头趋势"
        elif last["MA20"] < last["MA60"] < last["MA120"]:
            ma_status["mid_trend"] = "中期空头趋势"
        else:
            ma_status["mid_trend"] = "中期趋势不明"

        if prev["MA20"] <= prev["MA60"] and last["MA20"] > last["MA60"]:
            ma_status["golden_cross_20_60"] = "20日均线金叉60日均线 - 买入信号"
        elif prev["MA20"] >= prev["MA60"] and last["MA20"] < last["MA60"]:
            ma_status["death_cross_20_60"] = "20日均线死叉60日均线 - 卖出信号"
        else:
            ma_status["cross_signal"] = "无交叉信号"

        return ma_status

    def calculate_macd(self, df: pd.DataFrame) -> Dict:
        """MACD指标分析 - 蔡森经典指标"""
        df = df.copy()

        macd_indicator = MACD(
            close=df["Close"], window_fast=12, window_slow=26, window_sign=9
        )
        df["MACD"] = macd_indicator.macd()
        df["MACD_Signal"] = macd_indicator.macd_signal()
        df["MACD_Hist"] = macd_indicator.macd_diff()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        macd_status = {
            "MACD": float(last["MACD"]) if pd.notna(last["MACD"]) else None,
            "Signal": float(last["MACD_Signal"])
            if pd.notna(last["MACD_Signal"])
            else None,
            "Histogram": float(last["MACD_Hist"])
            if pd.notna(last["MACD_Hist"])
            else None,
        }

        if last["MACD"] > last["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
            macd_status["signal"] = "MACD金叉 - 买入信号"
        elif last["MACD"] < last["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
            macd_status["signal"] = "MACD死叉 - 卖出信号"
        elif last["MACD"] > last["MACD_Signal"]:
            macd_status["signal"] = "MACD多头运行"
        else:
            macd_status["signal"] = "MACD空头运行"

        if last["MACD_Hist"] > 0 and prev["MACD_Hist"] > 0:
            if last["MACD_Hist"] > prev["MACD_Hist"]:
                macd_status["momentum"] = "多头动能增强"
            else:
                macd_status["momentum"] = "多头动能减弱"
        elif last["MACD_Hist"] < 0 and prev["MACD_Hist"] < 0:
            if last["MACD_Hist"] > prev["MACD_Hist"]:
                macd_status["momentum"] = "空头动能减弱"
            else:
                macd_status["momentum"] = "空头动能增强"
        else:
            macd_status["momentum"] = "动能转换中"

        return macd_status

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """RSI指标分析"""
        df = df.copy()

        rsi_indicator = RSIIndicator(close=df["Close"], window=period)
        df["RSI"] = rsi_indicator.rsi()

        last = df.iloc[-1]
        rsi_value = last["RSI"]

        rsi_status = {
            "RSI": float(rsi_value) if pd.notna(rsi_value) else 50,
        }

        rsi_val = rsi_status["RSI"]
        if rsi_val >= 80:
            rsi_status["status"] = "严重超买"
            rsi_status["signal"] = "卖出"
        elif rsi_val >= 70:
            rsi_status["status"] = "超买区"
            rsi_status["signal"] = "谨慎"
        elif rsi_val >= 50:
            rsi_status["status"] = "强势区"
            rsi_status["signal"] = "持有"
        elif rsi_val >= 30:
            rsi_status["status"] = "弱势区"
            rsi_status["signal"] = "观望"
        else:
            rsi_status["status"] = "超卖区"
            rsi_status["signal"] = "关注买入机会"

        return rsi_status

    def calculate_kdj(self, df: pd.DataFrame) -> Dict:
        """KDJ指标分析"""
        df = df.copy()

        stoch = StochasticOscillator(
            high=df["High"], low=df["Low"], close=df["Close"], window=9, smooth_window=3
        )
        df["K"] = stoch.stoch()
        df["D"] = stoch.stoch_signal()
        df["J"] = 3 * df["K"] - 2 * df["D"]

        last = df.iloc[-1]
        prev = df.iloc[-2]

        kdj_status = {
            "K": float(last["K"]) if pd.notna(last["K"]) else 50,
            "D": float(last["D"]) if pd.notna(last["D"]) else 50,
            "J": float(last["J"]) if pd.notna(last["J"]) else 50,
        }

        if last["K"] > last["D"] and prev["K"] <= prev["D"]:
            kdj_status["signal"] = "KDJ金叉 - 买入信号"
        elif last["K"] < last["D"] and prev["K"] >= prev["D"]:
            kdj_status["signal"] = "KDJ死叉 - 卖出信号"
        elif last["K"] > last["D"]:
            kdj_status["signal"] = "KDJ多头运行"
        else:
            kdj_status["signal"] = "KDJ空头运行"

        if last["J"] > 100:
            kdj_status["j_status"] = "J值超买"
        elif last["J"] < 0:
            kdj_status["j_status"] = "J值超卖"
        else:
            kdj_status["j_status"] = "J值正常"

        return kdj_status

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> Dict:
        """布林带分析"""
        df = df.copy()

        bb_indicator = BollingerBands(close=df["Close"], window=period, window_dev=2)
        df["BB_Upper"] = bb_indicator.bollinger_hband()
        df["BB_Middle"] = bb_indicator.bollinger_mavg()
        df["BB_Lower"] = bb_indicator.bollinger_lband()
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"] * 100

        last = df.iloc[-1]

        bb_status = {
            "Upper": float(last["BB_Upper"]) if pd.notna(last["BB_Upper"]) else None,
            "Middle": float(last["BB_Middle"]) if pd.notna(last["BB_Middle"]) else None,
            "Lower": float(last["BB_Lower"]) if pd.notna(last["BB_Lower"]) else None,
            "Width": float(last["BB_Width"]) if pd.notna(last["BB_Width"]) else None,
            "Current_Price": float(last["Close"]),
        }

        bb_percent = (last["Close"] - last["BB_Lower"]) / (
            last["BB_Upper"] - last["BB_Lower"]
        )
        bb_status["BB_Percent"] = float(bb_percent) if pd.notna(bb_percent) else 0.5

        if last["Close"] >= last["BB_Upper"]:
            bb_status["status"] = "价格突破上轨 - 强势/超买"
        elif last["Close"] <= last["BB_Lower"]:
            bb_status["status"] = "价格跌破下轨 - 弱势/超卖"
        elif bb_percent >= 0.8:
            bb_status["status"] = "接近上轨 - 偏强"
        elif bb_percent <= 0.2:
            bb_status["status"] = "接近下轨 - 偏弱"
        else:
            bb_status["status"] = "中轨附近运行"

        avg_width = df["BB_Width"].tail(20).mean()
        if last["BB_Width"] < avg_width * 0.5:
            bb_status["squeeze"] = "布林带收窄 - 可能即将突破"
        elif last["BB_Width"] > avg_width * 1.5:
            bb_status["squeeze"] = "布林带扩张 - 波动加剧"
        else:
            bb_status["squeeze"] = "布林带正常"

        return bb_status

    def find_support_resistance(self, df: pd.DataFrame, lookback: int = 60) -> Dict:
        """支撑阻力位识别 - 蔡森关键价位法"""
        df = df.copy()
        recent = df.tail(lookback)

        highs = recent["High"].values
        lows = recent["Low"].values

        resistance_levels = []
        support_levels = []

        for i in range(2, len(highs) - 2):
            if (
                highs[i] > highs[i - 1]
                and highs[i] > highs[i - 2]
                and highs[i] > highs[i + 1]
                and highs[i] > highs[i + 2]
            ):
                resistance_levels.append(highs[i])
            if (
                lows[i] < lows[i - 1]
                and lows[i] < lows[i - 2]
                and lows[i] < lows[i + 1]
                and lows[i] < lows[i + 2]
            ):
                support_levels.append(lows[i])

        resistance_levels = sorted(set(resistance_levels), reverse=True)[:5]
        support_levels = sorted(set(support_levels), reverse=True)[:5]

        current_price = df.iloc[-1]["Close"]

        nearest_resistance = None
        for r in resistance_levels:
            if r > current_price:
                nearest_resistance = r
                break

        nearest_support = None
        for s in reversed(support_levels):
            if s < current_price:
                nearest_support = s
                break

        return {
            "current_price": float(current_price),
            "resistance_levels": [float(x) for x in resistance_levels],
            "support_levels": [float(x) for x in support_levels],
            "nearest_resistance": float(nearest_resistance)
            if nearest_resistance
            else None,
            "nearest_support": float(nearest_support) if nearest_support else None,
            "resistance_distance_pct": float(
                (nearest_resistance - current_price) / current_price * 100
            )
            if nearest_resistance
            else None,
            "support_distance_pct": float(
                (current_price - nearest_support) / current_price * 100
            )
            if nearest_support
            else None,
        }

    def calculate_fibonacci(self, df: pd.DataFrame, lookback: int = 120) -> Dict:
        """斐波那契回撤分析"""
        recent = df.tail(lookback)

        high = recent["High"].max()
        low = recent["Low"].min()
        current = df.iloc[-1]["Close"]

        diff = high - low

        fib_levels = {
            "0%": float(high),
            "23.6%": float(high - diff * 0.236),
            "38.2%": float(high - diff * 0.382),
            "50%": float(high - diff * 0.5),
            "61.8%": float(high - diff * 0.618),
            "78.6%": float(high - diff * 0.786),
            "100%": float(low),
        }

        position = None
        for level, price in list(fib_levels.items())[1:]:
            if current >= price:
                position = f"当前价格位于 {level} 之上"
                break

        return {
            "swing_high": float(high),
            "swing_low": float(low),
            "current_price": float(current),
            "fib_levels": fib_levels,
            "position": position,
        }

    def analyze_volume(self, df: pd.DataFrame) -> Dict:
        """量价关系分析 - 蔡森量价理论"""
        df = df.copy()

        df["Volume_MA5"] = sma(df["Volume"], 5)
        df["Volume_MA20"] = sma(df["Volume"], 20)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        vol_ratio = (
            last["Volume"] / last["Volume_MA20"] if last["Volume_MA20"] > 0 else 1
        )

        volume_status = {
            "current_volume": float(last["Volume"]),
            "volume_ma5": float(last["Volume_MA5"])
            if pd.notna(last["Volume_MA5"])
            else None,
            "volume_ma20": float(last["Volume_MA20"])
            if pd.notna(last["Volume_MA20"])
            else None,
            "volume_ratio": float(vol_ratio),
        }

        price_change = (
            (last["Close"] - prev["Close"]) / prev["Close"] * 100
            if prev["Close"] > 0
            else 0
        )
        volume_change = (
            (last["Volume"] - prev["Volume"]) / prev["Volume"] * 100
            if prev["Volume"] > 0
            else 0
        )

        if price_change > 2 and volume_change > 50:
            volume_status["pattern"] = "放量上涨 - 多头强势"
            volume_status["signal"] = "买入"
        elif price_change > 0 and volume_change > 0:
            volume_status["pattern"] = "量价齐升 - 趋势健康"
            volume_status["signal"] = "持有"
        elif price_change < -2 and volume_change > 50:
            volume_status["pattern"] = "放量下跌 - 空头强势"
            volume_status["signal"] = "卖出"
        elif price_change < 0 and volume_change > 0:
            volume_status["pattern"] = "放量下跌 - 卖压增加"
            volume_status["signal"] = "谨慎"
        elif price_change > 0 and volume_change < -30:
            volume_status["pattern"] = "缩量上涨 - 上涨乏力"
            volume_status["signal"] = "观望"
        elif price_change < 0 and volume_change < -30:
            volume_status["pattern"] = "缩量下跌 - 卖压减轻"
            volume_status["signal"] = "关注"
        else:
            volume_status["pattern"] = "量价正常"
            volume_status["signal"] = "中性"

        return volume_status

    def detect_kline_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """K线形态识别"""
        patterns = []

        if len(df) < 5:
            return patterns

        last = df.iloc[-1]
        prev = df.iloc[-2]

        body = abs(last["Close"] - last["Open"])
        upper_shadow = last["High"] - max(last["Open"], last["Close"])
        lower_shadow = min(last["Open"], last["Close"]) - last["Low"]
        total_range = last["High"] - last["Low"]

        if total_range > 0:
            if body / total_range < 0.1 and upper_shadow > body and lower_shadow > body:
                if prev["Close"] > prev["Open"]:
                    patterns.append(
                        {
                            "name": "十字星",
                            "signal": "趋势可能反转",
                            "significance": "高",
                        }
                    )

        if body > 0:
            if upper_shadow < body * 0.1 and lower_shadow < body * 0.1:
                if last["Close"] > last["Open"]:
                    patterns.append(
                        {
                            "name": "光头光脚阳线",
                            "signal": "强势多头",
                            "significance": "高",
                        }
                    )
                else:
                    patterns.append(
                        {
                            "name": "光头光脚阴线",
                            "signal": "强势空头",
                            "significance": "高",
                        }
                    )

        if last["Close"] > last["Open"] and prev["Close"] < prev["Open"]:
            if last["Close"] > prev["Open"] and last["Open"] < prev["Close"]:
                patterns.append(
                    {"name": "阳包阴", "signal": "看涨反转", "significance": "高"}
                )

        if last["Close"] < last["Open"] and prev["Close"] > prev["Open"]:
            if last["Open"] > prev["Close"] and last["Close"] < prev["Open"]:
                patterns.append(
                    {"name": "阴包阳", "signal": "看跌反转", "significance": "高"}
                )

        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            patterns.append(
                {
                    "name": "锤子线/下影线",
                    "signal": "底部反转信号",
                    "significance": "中",
                }
            )

        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            patterns.append(
                {
                    "name": "流星线/上影线",
                    "signal": "顶部反转信号",
                    "significance": "中",
                }
            )

        return patterns

    def detect_trend(self, df: pd.DataFrame) -> Dict:
        """趋势分析 - 蔡森趋势线法"""
        df = df.copy()

        recent = df.tail(60)

        high_peaks = []
        low_troughs = []

        for i in range(2, len(recent) - 2):
            if (
                recent.iloc[i]["High"] > recent.iloc[i - 1]["High"]
                and recent.iloc[i]["High"] > recent.iloc[i - 2]["High"]
                and recent.iloc[i]["High"] > recent.iloc[i + 1]["High"]
                and recent.iloc[i]["High"] > recent.iloc[i + 2]["High"]
            ):
                high_peaks.append((recent.index[i], recent.iloc[i]["High"]))

            if (
                recent.iloc[i]["Low"] < recent.iloc[i - 1]["Low"]
                and recent.iloc[i]["Low"] < recent.iloc[i - 2]["Low"]
                and recent.iloc[i]["Low"] < recent.iloc[i + 1]["Low"]
                and recent.iloc[i]["Low"] < recent.iloc[i + 2]["Low"]
            ):
                low_troughs.append((recent.index[i], recent.iloc[i]["Low"]))

        trend_status = {}

        if len(high_peaks) >= 2:
            last_high = high_peaks[-1][1]
            prev_high = high_peaks[-2][1]
            if last_high > prev_high:
                trend_status["high_trend"] = "高点上移 - 多头趋势"
            else:
                trend_status["high_trend"] = "高点下移 - 空头趋势"
        else:
            trend_status["high_trend"] = "无法确定高点趋势"

        if len(low_troughs) >= 2:
            last_low = low_troughs[-1][1]
            prev_low = low_troughs[-2][1]
            if last_low > prev_low:
                trend_status["low_trend"] = "低点上移 - 多头趋势"
            else:
                trend_status["low_trend"] = "低点下移 - 空头趋势"
        else:
            trend_status["low_trend"] = "无法确定低点趋势"

        if "高点上移" in trend_status.get(
            "high_trend", ""
        ) and "低点上移" in trend_status.get("low_trend", ""):
            trend_status["overall"] = "上升趋势"
            trend_status["action"] = "逢低做多"
        elif "高点下移" in trend_status.get(
            "high_trend", ""
        ) and "低点下移" in trend_status.get("low_trend", ""):
            trend_status["overall"] = "下降趋势"
            trend_status["action"] = "逢高做空"
        else:
            trend_status["overall"] = "震荡趋势"
            trend_status["action"] = "区间操作"

        return trend_status

    def generate_prediction(self, analysis: Dict) -> Dict:
        """综合分析生成预测"""
        signals = []

        ma = analysis.get("ma_system", {})
        if "强势多头排列" in ma.get("short_trend", ""):
            signals.append(("均线", 3, "多"))
        elif "多头排列" in ma.get("short_trend", ""):
            signals.append(("均线", 2, "多"))
        elif "强势空头排列" in ma.get("short_trend", ""):
            signals.append(("均线", -3, "空"))
        elif "空头排列" in ma.get("short_trend", ""):
            signals.append(("均线", -2, "空"))
        else:
            signals.append(("均线", 0, "中"))

        macd = analysis.get("macd", {})
        if "金叉" in macd.get("signal", ""):
            signals.append(("MACD", 2, "多"))
        elif "死叉" in macd.get("signal", ""):
            signals.append(("MACD", -2, "空"))
        elif "多头" in macd.get("signal", ""):
            signals.append(("MACD", 1, "多"))
        else:
            signals.append(("MACD", -1, "空"))

        rsi = analysis.get("rsi", {})
        rsi_val = rsi.get("RSI", 50)
        if rsi_val >= 70:
            signals.append(("RSI", -1, "超买"))
        elif rsi_val <= 30:
            signals.append(("RSI", 1, "超卖"))
        elif rsi_val >= 50:
            signals.append(("RSI", 1, "多"))
        else:
            signals.append(("RSI", -1, "空"))

        kdj = analysis.get("kdj", {})
        if "金叉" in kdj.get("signal", ""):
            signals.append(("KDJ", 2, "多"))
        elif "死叉" in kdj.get("signal", ""):
            signals.append(("KDJ", -2, "空"))
        elif "多头" in kdj.get("signal", ""):
            signals.append(("KDJ", 1, "多"))
        else:
            signals.append(("KDJ", -1, "空"))

        bb = analysis.get("bollinger", {})
        if "突破上轨" in bb.get("status", ""):
            signals.append(("布林", 2, "强多"))
        elif "跌破下轨" in bb.get("status", ""):
            signals.append(("布林", -2, "强空"))
        elif "偏强" in bb.get("status", ""):
            signals.append(("布林", 1, "多"))
        elif "偏弱" in bb.get("status", ""):
            signals.append(("布林", -1, "空"))
        else:
            signals.append(("布林", 0, "中"))

        vol = analysis.get("volume", {})
        if "放量上涨" in vol.get("pattern", "") or "量价齐升" in vol.get("pattern", ""):
            signals.append(("量价", 2, "多"))
        elif "放量下跌" in vol.get("pattern", ""):
            signals.append(("量价", -2, "空"))
        else:
            signals.append(("量价", 0, "中"))

        trend = analysis.get("trend", {})
        if "上升" in trend.get("overall", ""):
            signals.append(("趋势", 2, "多"))
        elif "下降" in trend.get("overall", ""):
            signals.append(("趋势", -2, "空"))
        else:
            signals.append(("趋势", 0, "震荡"))

        total_score = sum(s[1] for s in signals)

        if total_score >= 8:
            prediction = "强烈看多"
            action = "建议买入"
            confidence = "高"
        elif total_score >= 4:
            prediction = "偏多"
            action = "可考虑买入"
            confidence = "中"
        elif total_score <= -8:
            prediction = "强烈看空"
            action = "建议卖出"
            confidence = "高"
        elif total_score <= -4:
            prediction = "偏空"
            action = "可考虑卖出"
            confidence = "中"
        else:
            prediction = "震荡/中性"
            action = "观望或轻仓操作"
            confidence = "低"

        return {
            "signals": signals,
            "total_score": total_score,
            "prediction": prediction,
            "action": action,
            "confidence": confidence,
        }

    def full_analysis(self, symbol: str, df: pd.DataFrame) -> Dict:
        """执行完整技术分析"""
        print(f"\n{'=' * 60}")
        print(f"蔡森技术分析报告 - {symbol}")
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"数据范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}"
        )
        print(f"{'=' * 60}\n")

        analysis = {
            "symbol": symbol,
            "analysis_time": datetime.now().isoformat(),
            "data_range": f"{df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}",
            "current_price": float(df.iloc[-1]["Close"]),
        }

        print("【1. 均线系统分析】")
        analysis["ma_system"] = self.calculate_ma_system(df)
        ma = analysis["ma_system"]
        print(f"  当前价格: ${ma['current_price']:,.2f}")
        if ma["MA5"]:
            print(
                f"  MA5: ${ma['MA5']:,.2f} | MA10: ${ma['MA10']:,.2f} | MA20: ${ma['MA20']:,.2f}"
            )
        if ma["MA60"]:
            print(f"  MA60: ${ma['MA60']:,.2f} | MA120: ${ma['MA120']:,.2f}")
        print(f"  短期趋势: {ma['short_trend']} - {ma['short_signal']}")
        print(f"  中期趋势: {ma['mid_trend']}")

        print("\n【2. MACD指标分析】")
        analysis["macd"] = self.calculate_macd(df)
        macd = analysis["macd"]
        if macd["MACD"]:
            print(
                f"  MACD: {macd['MACD']:.4f} | Signal: {macd['Signal']:.4f} | Hist: {macd['Histogram']:.4f}"
            )
        print(f"  信号: {macd['signal']}")
        print(f"  动能: {macd['momentum']}")

        print("\n【3. RSI指标分析】")
        analysis["rsi"] = self.calculate_rsi(df)
        rsi = analysis["rsi"]
        print(f"  RSI(14): {rsi['RSI']:.2f}")
        print(f"  状态: {rsi['status']} - {rsi['signal']}")

        print("\n【4. KDJ指标分析】")
        analysis["kdj"] = self.calculate_kdj(df)
        kdj = analysis["kdj"]
        print(f"  K: {kdj['K']:.2f} | D: {kdj['D']:.2f} | J: {kdj['J']:.2f}")
        print(f"  信号: {kdj['signal']}")
        print(f"  J值状态: {kdj['j_status']}")

        print("\n【5. 布林带分析】")
        analysis["bollinger"] = self.calculate_bollinger_bands(df)
        bb = analysis["bollinger"]
        if bb["Upper"]:
            print(
                f"  上轨: ${bb['Upper']:,.2f} | 中轨: ${bb['Middle']:,.2f} | 下轨: ${bb['Lower']:,.2f}"
            )
        print(f"  带宽: {bb['Width']:.2f}%" if bb["Width"] else "")
        print(f"  状态: {bb['status']}")
        print(f"  带宽状态: {bb['squeeze']}")

        print("\n【6. 支撑阻力位】")
        analysis["support_resistance"] = self.find_support_resistance(df)
        sr = analysis["support_resistance"]
        print(f"  当前价格: ${sr['current_price']:,.2f}")
        if sr["nearest_resistance"]:
            print(
                f"  最近阻力: ${sr['nearest_resistance']:,.2f} (距离: +{sr['resistance_distance_pct']:.2f}%)"
            )
        if sr["nearest_support"]:
            print(
                f"  最近支撑: ${sr['nearest_support']:,.2f} (距离: -{sr['support_distance_pct']:.2f}%)"
            )
        print(f"  阻力位: {[f'${x:,.2f}' for x in sr['resistance_levels']]}")
        print(f"  支撑位: {[f'${x:,.2f}' for x in sr['support_levels']]}")

        print("\n【7. 斐波那契回撤】")
        analysis["fibonacci"] = self.calculate_fibonacci(df)
        fib = analysis["fibonacci"]
        print(f"  波段高点: ${fib['swing_high']:,.2f}")
        print(f"  波段低点: ${fib['swing_low']:,.2f}")
        print(f"  当前位置: {fib['position']}")

        print("\n【8. 量价关系】")
        analysis["volume"] = self.analyze_volume(df)
        vol = analysis["volume"]
        print(f"  量比: {vol['volume_ratio']:.2f}")
        print(f"  形态: {vol['pattern']}")
        print(f"  信号: {vol['signal']}")

        print("\n【9. 趋势分析】")
        analysis["trend"] = self.detect_trend(df)
        trend = analysis["trend"]
        print(f"  高点趋势: {trend['high_trend']}")
        print(f"  低点趋势: {trend['low_trend']}")
        print(f"  整体趋势: {trend['overall']}")
        print(f"  操作建议: {trend['action']}")

        print("\n【10. K线形态】")
        analysis["kline_patterns"] = self.detect_kline_patterns(df)
        patterns = analysis["kline_patterns"]
        if patterns:
            for p in patterns:
                print(f"  {p['name']}: {p['signal']} (显著性: {p['significance']})")
        else:
            print("  未检测到明显K线形态")

        print("\n" + "=" * 60)
        print("【综合预测】")
        print("=" * 60)
        analysis["prediction"] = self.generate_prediction(analysis)
        pred = analysis["prediction"]

        print("\n信号汇总:")
        for s in pred["signals"]:
            print(f"  {s[0]}: {s[2]} ({'+' if s[1] > 0 else ''}{s[1]})")

        print(f"\n综合得分: {pred['total_score']}")
        print(f"预测方向: {pred['prediction']}")
        print(f"操作建议: {pred['action']}")
        print(f"置信度: {pred['confidence']}")
        print(f"\n{'=' * 60}\n")

        return analysis


def main():
    """主函数"""
    analyzer = CaiSenTechnicalAnalysis()

    symbols = ["BTC", "ETH"]
    results = {}

    for symbol in symbols:
        print(f"\n正在获取 {symbol} 数据...")

        df = analyzer.fetch_crypto_data_twelvedata(symbol, outputsize=365)

        if df is None or len(df) < 120:
            print(f"Twelve Data 获取失败，尝试 Alpha Vantage...")
            df = analyzer.fetch_crypto_data_alphavantage(symbol)

        if df is None or len(df) < 120:
            print(f"无法获取足够的 {symbol} 数据进行分析")
            continue

        results[symbol] = analyzer.full_analysis(symbol, df)

    if results:
        output_file = "/root/ideas/caishen/analysis_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n分析结果已保存至: {output_file}")

    return results


if __name__ == "__main__":
    main()
