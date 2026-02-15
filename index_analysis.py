#!/usr/bin/env python3
"""
蔡森技术分析系统 - 多时间框架指数分析
使用 yfinance 获取 SPX, NASDAQ100, NIKKEI225, DAX 数据
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

INDEX_SYMBOLS = {
    "SPX": {"yf": "^GSPC", "name": "S&P 500", "market": "US"},
    "ND100": {"yf": "^NDX", "name": "Nasdaq 100", "market": "US"},
    "JP225": {"yf": "^N225", "name": "Nikkei 225", "market": "JP"},
    "DAX": {"yf": "^GDAXI", "name": "DAX 40", "market": "DE"},
}


class CaiSenMultiTimeframeAnalysis:
    """蔡森多时间框架技术分析"""

    def __init__(self):
        self.timeframes = {
            "daily": {"interval": "1d", "period": "1y"},
            "1h": {"interval": "1h", "period": "1mo"},
            "15m": {"interval": "15m", "period": "1mo"},
        }

    def fetch_index_data(
        self, symbol: str, interval: str = "1d", period: str = "1y"
    ) -> pd.DataFrame:
        """使用 yfinance 获取指数数据"""
        config = INDEX_SYMBOLS.get(symbol, {})
        yf_symbol = config.get("yf", symbol)

        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"  No data returned for {symbol}")
                return None

            df = df.rename(
                columns={
                    "Open": "Open",
                    "High": "High",
                    "Low": "Low",
                    "Close": "Close",
                    "Volume": "Volume",
                }
            )

            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            return df[["Open", "High", "Low", "Close"]].dropna()

        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()

        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA10"] = df["Close"].rolling(10).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()

        try:
            macd = MACD(
                close=df["Close"], window_fast=12, window_slow=26, window_sign=9
            )
            df["MACD"] = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["MACD_Hist"] = macd.macd_diff()
        except:
            df["MACD"] = 0
            df["MACD_Signal"] = 0
            df["MACD_Hist"] = 0

        try:
            rsi = RSIIndicator(close=df["Close"], window=14)
            df["RSI"] = rsi.rsi()
        except:
            df["RSI"] = 50

        try:
            stoch = StochasticOscillator(
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                window=9,
                smooth_window=3,
            )
            df["K"] = stoch.stoch()
            df["D"] = stoch.stoch_signal()
            df["J"] = 3 * df["K"] - 2 * df["D"]
        except:
            df["K"] = 50
            df["D"] = 50
            df["J"] = 50

        try:
            bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
            df["BB_Upper"] = bb.bollinger_hband()
            df["BB_Middle"] = bb.bollinger_mavg()
            df["BB_Lower"] = bb.bollinger_lband()
            df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"] * 100
        except:
            pass

        return df

    def analyze_ma_system(self, df: pd.DataFrame) -> dict:
        """均线系统分析"""
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        result = {
            "price": float(last["Close"]),
            "MA5": float(last["MA5"]) if pd.notna(last["MA5"]) else None,
            "MA10": float(last["MA10"]) if pd.notna(last["MA10"]) else None,
            "MA20": float(last["MA20"]) if pd.notna(last["MA20"]) else None,
            "MA60": float(last["MA60"]) if pd.notna(last["MA60"]) else None,
        }

        ma5 = last["MA5"]
        ma10 = last["MA10"]
        ma20 = last["MA20"]
        close = last["Close"]

        if pd.isna(ma20):
            result["alignment"] = "数据不足"
            result["signal"] = 0
            return result

        if close > ma5 > ma10 > ma20:
            result["alignment"] = "强势多头排列"
            result["signal"] = 3
        elif ma5 > ma10 > ma20:
            result["alignment"] = "多头排列"
            result["signal"] = 2
        elif close < ma5 < ma10 < ma20:
            result["alignment"] = "强势空头排列"
            result["signal"] = -3
        elif ma5 < ma10 < ma20:
            result["alignment"] = "空头排列"
            result["signal"] = -2
        else:
            result["alignment"] = "均线纠缠"
            result["signal"] = 0

        # 金叉死叉
        if pd.notna(prev["MA5"]) and pd.notna(prev["MA10"]):
            if prev["MA5"] <= prev["MA10"] and ma5 > ma10:
                result["cross"] = "MA5金叉MA10"
                result["cross_signal"] = 1
            elif prev["MA5"] >= prev["MA10"] and ma5 < ma10:
                result["cross"] = "MA5死叉MA10"
                result["cross_signal"] = -1

        return result

    def analyze_macd(self, df: pd.DataFrame) -> dict:
        """MACD分析"""
        last = df.iloc[-1]
        prev = df.iloc[-2]

        result = {
            "MACD": float(last["MACD"]) if pd.notna(last["MACD"]) else 0,
            "Signal": float(last["MACD_Signal"])
            if pd.notna(last["MACD_Signal"])
            else 0,
            "Histogram": float(last["MACD_Hist"]) if pd.notna(last["MACD_Hist"]) else 0,
        }

        if prev["MACD"] <= prev["MACD_Signal"] and last["MACD"] > last["MACD_Signal"]:
            result["signal"] = "金叉"
            result["score"] = 2
        elif prev["MACD"] >= prev["MACD_Signal"] and last["MACD"] < last["MACD_Signal"]:
            result["signal"] = "死叉"
            result["score"] = -2
        elif last["MACD"] > last["MACD_Signal"]:
            result["signal"] = "多头运行"
            result["score"] = 1
        else:
            result["signal"] = "空头运行"
            result["score"] = -1

        if last["MACD"] > 0 and last["MACD_Signal"] > 0:
            result["position"] = "零轴上方"
        elif last["MACD"] < 0 and last["MACD_Signal"] < 0:
            result["position"] = "零轴下方"
        else:
            result["position"] = "零轴附近"

        return result

    def analyze_rsi(self, df: pd.DataFrame) -> dict:
        """RSI分析"""
        last = df.iloc[-1]
        rsi = last["RSI"] if pd.notna(last["RSI"]) else 50

        result = {"value": float(rsi)}

        if rsi >= 80:
            result["status"] = "严重超买"
            result["signal"] = -2
        elif rsi >= 70:
            result["status"] = "超买"
            result["signal"] = -1
        elif rsi >= 50:
            result["status"] = "强势区"
            result["signal"] = 1
        elif rsi >= 30:
            result["status"] = "弱势区"
            result["signal"] = -1
        else:
            result["status"] = "超卖"
            result["signal"] = 2

        return result

    def analyze_kdj(self, df: pd.DataFrame) -> dict:
        """KDJ分析"""
        last = df.iloc[-1]
        prev = df.iloc[-2]

        result = {
            "K": float(last["K"]) if pd.notna(last["K"]) else 50,
            "D": float(last["D"]) if pd.notna(last["D"]) else 50,
            "J": float(last["J"]) if pd.notna(last["J"]) else 50,
        }

        if prev["K"] <= prev["D"] and last["K"] > last["D"]:
            result["signal"] = "金叉"
            result["score"] = 2
        elif prev["K"] >= prev["D"] and last["K"] < last["D"]:
            result["signal"] = "死叉"
            result["score"] = -2
        elif last["K"] > last["D"]:
            result["signal"] = "多头"
            result["score"] = 1
        else:
            result["signal"] = "空头"
            result["score"] = -1

        return result

    def analyze_bollinger(self, df: pd.DataFrame) -> dict:
        """布林带分析"""
        last = df.iloc[-1]

        result = {
            "Upper": float(last["BB_Upper"])
            if "BB_Upper" in df.columns and pd.notna(last["BB_Upper"])
            else None,
            "Middle": float(last["BB_Middle"])
            if "BB_Middle" in df.columns and pd.notna(last["BB_Middle"])
            else None,
            "Lower": float(last["BB_Lower"])
            if "BB_Lower" in df.columns and pd.notna(last["BB_Lower"])
            else None,
            "Width": float(last["BB_Width"])
            if "BB_Width" in df.columns and pd.notna(last["BB_Width"])
            else None,
        }

        if result["Upper"] and result["Lower"]:
            bb_percent = (last["Close"] - result["Lower"]) / (
                result["Upper"] - result["Lower"]
            )
            result["BB_Percent"] = float(bb_percent)

            if last["Close"] >= result["Upper"]:
                result["status"] = "突破上轨"
                result["signal"] = 2
            elif last["Close"] <= result["Lower"]:
                result["status"] = "跌破下轨"
                result["signal"] = -2
            elif bb_percent >= 0.8:
                result["status"] = "接近上轨"
                result["signal"] = 1
            elif bb_percent <= 0.2:
                result["status"] = "接近下轨"
                result["signal"] = -1
            else:
                result["status"] = "中轨运行"
                result["signal"] = 0

        return result

    def find_levels(self, df: pd.DataFrame, lookback: int = 60) -> dict:
        """支撑阻力位"""
        recent = df.tail(min(lookback, len(df)))

        highs = recent["High"].values
        lows = recent["Low"].values

        resistance = []
        support = []

        for i in range(2, len(highs) - 2):
            if (
                highs[i] > highs[i - 1]
                and highs[i] > highs[i - 2]
                and highs[i] > highs[i + 1]
                and highs[i] > highs[i + 2]
            ):
                resistance.append(highs[i])
            if (
                lows[i] < lows[i - 1]
                and lows[i] < lows[i - 2]
                and lows[i] < lows[i + 1]
                and lows[i] < lows[i + 2]
            ):
                support.append(lows[i])

        resistance = sorted(set(resistance), reverse=True)[:5]
        support = sorted(set(support), reverse=True)[:5]

        current = df.iloc[-1]["Close"]

        nearest_r = None
        for r in resistance:
            if r > current:
                nearest_r = r
                break

        nearest_s = None
        for s in reversed(support):
            if s < current:
                nearest_s = s
                break

        return {
            "current": float(current),
            "resistance": [float(x) for x in resistance],
            "support": [float(x) for x in support],
            "nearest_resistance": float(nearest_r) if nearest_r else None,
            "nearest_support": float(nearest_s) if nearest_s else None,
        }

    def analyze_trend(self, df: pd.DataFrame) -> dict:
        """趋势分析"""
        recent = df.tail(min(60, len(df)))

        result = {}

        if (
            len(recent) >= 20
            and pd.notna(recent.iloc[-1]["MA20"])
            and pd.notna(recent.iloc[-20]["MA20"])
        ):
            ma20_slope = (
                (recent.iloc[-1]["MA20"] - recent.iloc[-20]["MA20"])
                / recent.iloc[-20]["MA20"]
                * 100
            )
            result["ma_slope"] = float(ma20_slope)
        else:
            result["ma_slope"] = 0

        # 高低点分析
        highs = recent["High"].values
        lows = recent["Low"].values

        high_peaks = []
        low_troughs = []

        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                high_peaks.append(highs[i])
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                low_troughs.append(lows[i])

        if len(high_peaks) >= 2:
            if high_peaks[-1] > high_peaks[-2]:
                result["high_trend"] = "高点上移"
            else:
                result["high_trend"] = "高点下移"

        if len(low_troughs) >= 2:
            if low_troughs[-1] > low_troughs[-2]:
                result["low_trend"] = "低点上移"
            else:
                result["low_trend"] = "低点下移"

        if (
            result.get("high_trend") == "高点上移"
            and result.get("low_trend") == "低点上移"
        ):
            result["trend"] = "上升趋势"
            result["score"] = 2
        elif (
            result.get("high_trend") == "高点下移"
            and result.get("low_trend") == "低点下移"
        ):
            result["trend"] = "下降趋势"
            result["score"] = -2
        else:
            result["trend"] = "震荡趋势"
            result["score"] = 0

        return result

    def generate_signal(self, analysis: dict) -> dict:
        """生成综合信号"""
        score = 0
        signals = []

        ma = analysis.get("ma", {})
        ma_signal = ma.get("signal", 0)
        score += ma_signal * 2
        signals.append(("均线", ma_signal * 2, ma.get("alignment", "-")))

        macd = analysis.get("macd", {})
        macd_score = macd.get("score", 0)
        score += macd_score
        signals.append(("MACD", macd_score, macd.get("signal", "-")))

        rsi = analysis.get("rsi", {})
        rsi_signal = rsi.get("signal", 0)
        score += rsi_signal
        signals.append(("RSI", rsi_signal, rsi.get("status", "-")))

        kdj = analysis.get("kdj", {})
        kdj_score = kdj.get("score", 0)
        score += kdj_score
        signals.append(("KDJ", kdj_score, kdj.get("signal", "-")))

        bb = analysis.get("bollinger", {})
        bb_signal = bb.get("signal", 0)
        score += bb_signal
        signals.append(("布林", bb_signal, bb.get("status", "-")))

        trend = analysis.get("trend", {})
        trend_score = trend.get("score", 0)
        score += trend_score
        signals.append(("趋势", trend_score, trend.get("trend", "-")))

        if score >= 8:
            direction = "强烈看多"
            action = "买入"
            confidence = "高"
        elif score >= 4:
            direction = "偏多"
            action = "可买"
            confidence = "中"
        elif score <= -8:
            direction = "强烈看空"
            action = "卖出"
            confidence = "高"
        elif score <= -4:
            direction = "偏空"
            action = "可卖"
            confidence = "中"
        else:
            direction = "震荡"
            action = "观望"
            confidence = "低"

        return {
            "total_score": score,
            "direction": direction,
            "action": action,
            "confidence": confidence,
            "signals": signals,
        }

    def analyze_timeframe(self, df: pd.DataFrame, tf_name: str) -> dict:
        """单时间框架分析"""
        df = self.calculate_indicators(df)

        analysis = {
            "timeframe": tf_name,
            "current_price": float(df.iloc[-1]["Close"]),
            "date": df.index[-1].strftime("%Y-%m-%d %H:%M"),
        }

        analysis["ma"] = self.analyze_ma_system(df)
        analysis["macd"] = self.analyze_macd(df)
        analysis["rsi"] = self.analyze_rsi(df)
        analysis["kdj"] = self.analyze_kdj(df)
        analysis["bollinger"] = self.analyze_bollinger(df)
        analysis["levels"] = self.find_levels(df)
        analysis["trend"] = self.analyze_trend(df)
        analysis["signal"] = self.generate_signal(analysis)

        return analysis

    def full_analysis(self, symbol: str) -> dict:
        """完整多时间框架分析"""
        config = INDEX_SYMBOLS.get(symbol, {})
        name = config.get("name", symbol)

        print(f"\n{'=' * 70}")
        print(f"蔡森技术分析 - {name} ({symbol})")
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}")

        result = {
            "symbol": symbol,
            "name": name,
            "analysis_time": datetime.now().isoformat(),
            "timeframes": {},
        }

        for tf_name, tf_config in self.timeframes.items():
            print(f"\n【{tf_name.upper()} 时间框架】")

            df = self.fetch_index_data(
                symbol, interval=tf_config["interval"], period=tf_config["period"]
            )

            if df is None or len(df) < 30:
                print(f"  数据不足，跳过 {tf_name}")
                continue

            tf_analysis = self.analyze_timeframe(df, tf_name)
            result["timeframes"][tf_name] = tf_analysis

            sig = tf_analysis["signal"]
            ma = tf_analysis["ma"]
            macd = tf_analysis["macd"]
            rsi = tf_analysis["rsi"]
            levels = tf_analysis["levels"]
            trend = tf_analysis["trend"]

            print(f"  当前价格: {tf_analysis['current_price']:,.2f}")
            print(f"  均线: {ma.get('alignment', '-')}")
            print(f"  MACD: {macd.get('signal', '-')} ({macd.get('position', '-')})")
            print(f"  RSI: {rsi['value']:.1f} ({rsi.get('status', '-')})")
            print(f"  趋势: {trend.get('trend', '-')}")

            if levels.get("nearest_resistance"):
                print(f"  阻力: {levels['nearest_resistance']:,.2f}")
            if levels.get("nearest_support"):
                print(f"  支撑: {levels['nearest_support']:,.2f}")

            print(
                f"  ━━━ 综合得分: {sig['total_score']} | 方向: {sig['direction']} | 操作: {sig['action']} ━━━"
            )

        return result


def generate_trading_plan(analyses: dict) -> dict:
    """生成下周交易计划"""
    print(f"\n{'=' * 70}")
    print("蔡森技术分析 - 下周交易计划")
    print("=" * 70)

    plan = {"week": "下周", "generated_at": datetime.now().isoformat(), "indices": {}}

    for symbol, analysis in analyses.items():
        if "timeframes" not in analysis or not analysis["timeframes"]:
            continue

        config = INDEX_SYMBOLS.get(symbol, {})
        name = config.get("name", symbol)

        daily = analysis["timeframes"].get("daily", {})
        h1 = analysis["timeframes"].get("1h", {})
        m15 = analysis["timeframes"].get("15m", {})

        if not daily:
            continue

        daily_sig = daily.get("signal", {})
        h1_sig = h1.get("signal", {}) if h1 else {}

        current_price = daily.get("current_price", 0)
        daily_score = daily_sig.get("total_score", 0)
        h1_score = h1_sig.get("total_score", 0) if h1_sig else 0

        levels = daily.get("levels", {})

        index_plan = {
            "name": name,
            "current_price": current_price,
            "daily_score": daily_score,
            "h4_score": h1_score,
            "direction": daily_sig.get("direction", "震荡"),
            "action": daily_sig.get("action", "观望"),
            "confidence": daily_sig.get("confidence", "低"),
        }

        # 设置交易参数
        if daily_score >= 6:
            index_plan["bias"] = "多头"
            index_plan["entry_type"] = "回调买入"
            if levels.get("nearest_support"):
                sup = levels["nearest_support"]
                index_plan["entry_zone"] = f"{sup * 1.005:,.2f} - {sup * 1.015:,.2f}"
                index_plan["stop_loss"] = sup * 0.98
            else:
                index_plan["entry_zone"] = (
                    f"{current_price * 0.99:,.2f} - {current_price * 0.995:,.2f}"
                )
                index_plan["stop_loss"] = current_price * 0.97
            if levels.get("nearest_resistance"):
                res = levels["nearest_resistance"]
                index_plan["target1"] = res * 0.995
                index_plan["target2"] = res * 1.02
            else:
                index_plan["target1"] = current_price * 1.02
                index_plan["target2"] = current_price * 1.04
            index_plan["position_size"] = "标准仓 (2-3%)"

        elif daily_score >= 3:
            index_plan["bias"] = "偏多"
            index_plan["entry_type"] = "轻仓买入"
            if levels.get("nearest_support"):
                sup = levels["nearest_support"]
                index_plan["entry_zone"] = f"{sup * 1.005:,.2f} - {sup * 1.02:,.2f}"
                index_plan["stop_loss"] = sup * 0.97
            else:
                index_plan["entry_zone"] = (
                    f"{current_price * 0.985:,.2f} - {current_price * 0.995:,.2f}"
                )
                index_plan["stop_loss"] = current_price * 0.97
            if levels.get("nearest_resistance"):
                res = levels["nearest_resistance"]
                index_plan["target1"] = res * 0.99
                index_plan["target2"] = res * 1.01
            else:
                index_plan["target1"] = current_price * 1.015
                index_plan["target2"] = current_price * 1.03
            index_plan["position_size"] = "轻仓 (1-2%)"

        elif daily_score <= -6:
            index_plan["bias"] = "空头"
            index_plan["entry_type"] = "反弹卖出"
            if levels.get("nearest_resistance"):
                res = levels["nearest_resistance"]
                index_plan["entry_zone"] = f"{res * 0.985:,.2f} - {res * 0.995:,.2f}"
                index_plan["stop_loss"] = res * 1.02
            else:
                index_plan["entry_zone"] = (
                    f"{current_price * 1.005:,.2f} - {current_price * 1.01:,.2f}"
                )
                index_plan["stop_loss"] = current_price * 1.03
            if levels.get("nearest_support"):
                sup = levels["nearest_support"]
                index_plan["target1"] = sup * 1.005
                index_plan["target2"] = sup * 0.98
            else:
                index_plan["target1"] = current_price * 0.98
                index_plan["target2"] = current_price * 0.96
            index_plan["position_size"] = "标准仓 (2-3%)"

        elif daily_score <= -3:
            index_plan["bias"] = "偏空"
            index_plan["entry_type"] = "轻仓卖出"
            if levels.get("nearest_resistance"):
                res = levels["nearest_resistance"]
                index_plan["entry_zone"] = f"{res * 0.98:,.2f} - {res * 0.995:,.2f}"
                index_plan["stop_loss"] = res * 1.03
            else:
                index_plan["entry_zone"] = (
                    f"{current_price * 1.005:,.2f} - {current_price * 1.015:,.2f}"
                )
                index_plan["stop_loss"] = current_price * 1.03
            if levels.get("nearest_support"):
                sup = levels["nearest_support"]
                index_plan["target1"] = sup * 1.01
                index_plan["target2"] = sup * 0.99
            else:
                index_plan["target1"] = current_price * 0.985
                index_plan["target2"] = current_price * 0.97
            index_plan["position_size"] = "轻仓 (1-2%)"

        else:
            index_plan["bias"] = "震荡"
            index_plan["entry_type"] = "区间操作"
            sup = levels.get("nearest_support", current_price * 0.97)
            res = levels.get("nearest_resistance", current_price * 1.03)
            index_plan["entry_zone"] = f"{sup:,.2f} - {res:,.2f}"
            index_plan["stop_loss"] = "区间外1-2%"
            index_plan["target1"] = "区间另一端"
            index_plan["target2"] = "延展1%"
            index_plan["position_size"] = "观望或极轻仓 (0.5-1%)"

        plan["indices"][symbol] = index_plan

        # 打印
        print(f"\n【{name} ({symbol})】")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  当前价格: {current_price:,.2f}")
        print(f"  日线得分: {daily_score} | 1H得分: {h1_score}")
        print(f"  方向判断: {index_plan['bias']} ({daily_sig.get('direction', '-')})")
        print(f"  置信度: {index_plan['confidence']}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  操作建议: {index_plan['entry_type']}")
        print(f"  入场区域: {index_plan.get('entry_zone', '-')}")
        print(f"  止损位: {index_plan.get('stop_loss', '-')}")
        print(f"  目标1: {index_plan.get('target1', '-')}")
        print(f"  目标2: {index_plan.get('target2', '-')}")
        print(f"  仓位建议: {index_plan['position_size']}")

    return plan


def main():
    """主函数"""
    analyzer = CaiSenMultiTimeframeAnalysis()
    analyses = {}

    for symbol in INDEX_SYMBOLS.keys():
        analyses[symbol] = analyzer.full_analysis(symbol)

    # 生成交易计划
    plan = generate_trading_plan(analyses)

    # 保存结果
    output = {"analyses": analyses, "trading_plan": plan}

    output_file = "/root/ideas/caishen/index_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n\n分析结果已保存至: {output_file}")

    return output


if __name__ == "__main__":
    main()
