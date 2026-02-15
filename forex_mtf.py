#!/usr/bin/env python3
"""
蔡森技术分析系统 - 外汇多时间框架分析
分析 EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CNH
时间框架: 1/4/15/30/60/240 分钟
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

TWELVE_DATA_API_KEY = "f5491ce160e64101a960e19eb8363f38"

PAIRS = {
    "EUR": {"symbol": "EUR/USD", "name": "欧元/美元"},
    "GBP": {"symbol": "GBP/USD", "name": "英镑/美元"},
    "JPY": {"symbol": "USD/JPY", "name": "美元/日元"},
    "AUD": {"symbol": "AUD/USD", "name": "澳元/美元"},
    "CAD": {"symbol": "USD/CAD", "name": "美元/加元"},
    "CNH": {"symbol": "USD/CNH", "name": "美元/人民币"},
}

TIMEFRAMES = {
    "1m": {"interval": "1min", "outputsize": 200},
    "4m": {"interval": "4min", "outputsize": 200},
    "15m": {"interval": "15min", "outputsize": 200},
    "30m": {"interval": "30min", "outputsize": 200},
    "60m": {"interval": "1h", "outputsize": 200},
    "240m": {"interval": "4h", "outputsize": 200},
}


def fetch_forex_data(symbol, interval, outputsize=200):
    """获取外汇数据"""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if "status" in data and data["status"] == "error":
            return None, data.get("message", "Unknown error")

        if "values" not in data:
            return None, "No data"

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])

        df = df.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
        )

        return df[["Open", "High", "Low", "Close"]].dropna(), None

    except Exception as e:
        return None, str(e)


def calculate_indicators(df):
    """计算技术指标"""
    df = df.copy()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    try:
        macd = MACD(close=df["Close"], window_fast=12, window_slow=26, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_Sig"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()
    except:
        df["MACD"] = 0
        df["MACD_Sig"] = 0
        df["MACD_Hist"] = 0

    try:
        rsi = RSIIndicator(close=df["Close"], window=14)
        df["RSI"] = rsi.rsi()
    except:
        df["RSI"] = 50

    try:
        stoch = StochasticOscillator(
            high=df["High"], low=df["Low"], close=df["Close"], window=9, smooth_window=3
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
        df["BB_Lower"] = bb.bollinger_lband()
        df["BB_Mid"] = bb.bollinger_mavg()
    except:
        pass

    return df


def analyze_timeframe(df, tf_name):
    """分析单时间框架"""
    df = calculate_indicators(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    result = {
        "tf": tf_name,
        "price": float(last["Close"]),
        "time": last.name.strftime("%Y-%m-%d %H:%M")
        if hasattr(last.name, "strftime")
        else str(last.name),
    }

    # 均线系统
    if pd.notna(last["MA20"]):
        if last["Close"] > last["MA5"] > last["MA10"] > last["MA20"]:
            result["ma"] = "多头排列"
            result["ma_score"] = 2
        elif last["MA5"] > last["MA10"] > last["MA20"]:
            result["ma"] = "偏多排列"
            result["ma_score"] = 1
        elif last["Close"] < last["MA5"] < last["MA10"] < last["MA20"]:
            result["ma"] = "空头排列"
            result["ma_score"] = -2
        elif last["MA5"] < last["MA10"] < last["MA20"]:
            result["ma"] = "偏空排列"
            result["ma_score"] = -1
        else:
            result["ma"] = "均线纠缠"
            result["ma_score"] = 0
    else:
        result["ma"] = "N/A"
        result["ma_score"] = 0

    # MACD
    if pd.notna(prev["MACD"]) and pd.notna(prev["MACD_Sig"]):
        if prev["MACD"] <= prev["MACD_Sig"] and last["MACD"] > last["MACD_Sig"]:
            result["macd"] = "金叉"
            result["macd_score"] = 2
        elif prev["MACD"] >= prev["MACD_Sig"] and last["MACD"] < last["MACD_Sig"]:
            result["macd"] = "死叉"
            result["macd_score"] = -2
        elif last["MACD"] > last["MACD_Sig"]:
            result["macd"] = "多头"
            result["macd_score"] = 1
        else:
            result["macd"] = "空头"
            result["macd_score"] = -1
    else:
        result["macd"] = "N/A"
        result["macd_score"] = 0

    result["macd_pos"] = (
        "零轴上" if last["MACD"] > 0 else "零轴下" if last["MACD"] < 0 else "零轴"
    )

    # RSI
    rsi = last["RSI"] if pd.notna(last["RSI"]) else 50
    result["rsi"] = float(rsi)
    if rsi >= 80:
        result["rsi_status"] = "严重超买"
        result["rsi_score"] = -2
    elif rsi >= 70:
        result["rsi_status"] = "超买"
        result["rsi_score"] = -1
    elif rsi >= 50:
        result["rsi_status"] = "强势"
        result["rsi_score"] = 1
    elif rsi >= 30:
        result["rsi_status"] = "弱势"
        result["rsi_score"] = -1
    else:
        result["rsi_status"] = "超卖"
        result["rsi_score"] = 2

    # KDJ
    if pd.notna(prev["K"]) and pd.notna(prev["D"]):
        if prev["K"] <= prev["D"] and last["K"] > last["D"]:
            result["kdj"] = "金叉"
            result["kdj_score"] = 2
        elif prev["K"] >= prev["D"] and last["K"] < last["D"]:
            result["kdj"] = "死叉"
            result["kdj_score"] = -2
        elif last["K"] > last["D"]:
            result["kdj"] = "多头"
            result["kdj_score"] = 1
        else:
            result["kdj"] = "空头"
            result["kdj_score"] = -1
    else:
        result["kdj"] = "N/A"
        result["kdj_score"] = 0

    # 趋势
    lookback = min(60, len(df))
    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values

    high_peaks = []
    low_troughs = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            high_peaks.append(highs[i])
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            low_troughs.append(lows[i])

    if len(high_peaks) >= 2 and len(low_troughs) >= 2:
        if high_peaks[-1] > high_peaks[-2] and low_troughs[-1] > low_troughs[-2]:
            result["trend"] = "上升"
            result["trend_score"] = 2
        elif high_peaks[-1] < high_peaks[-2] and low_troughs[-1] < low_troughs[-2]:
            result["trend"] = "下降"
            result["trend_score"] = -2
        else:
            result["trend"] = "震荡"
            result["trend_score"] = 0
    else:
        result["trend"] = "不明"
        result["trend_score"] = 0

    # 支撑阻力
    resist = []
    suppt = []
    for i in range(2, len(recent) - 2):
        if (
            highs[i] > highs[i - 1]
            and highs[i] > highs[i + 1]
            and highs[i] > highs[i - 2]
            and highs[i] > highs[i + 2]
        ):
            resist.append(highs[i])
        if (
            lows[i] < lows[i - 1]
            and lows[i] < lows[i + 1]
            and lows[i] < lows[i - 2]
            and lows[i] < lows[i + 2]
        ):
            suppt.append(lows[i])

    resist = sorted(set(resist), reverse=True)[:3]
    suppt = sorted(set(suppt), reverse=True)[:3]

    nr = next((r for r in resist if r > last["Close"]), None)
    ns = next((s for s in reversed(suppt) if s < last["Close"]), None)

    result["resistance"] = [float(x) for x in resist]
    result["support"] = [float(x) for x in suppt]
    result["near_r"] = float(nr) if nr else None
    result["near_s"] = float(ns) if ns else None

    # 布林带位置
    if "BB_Upper" in df.columns and pd.notna(last["BB_Upper"]):
        bb_pct = (last["Close"] - last["BB_Lower"]) / (
            last["BB_Upper"] - last["BB_Lower"]
        )
        result["bb_pct"] = float(bb_pct)
        if bb_pct >= 0.8:
            result["bb"] = "上轨区"
            result["bb_score"] = -1
        elif bb_pct <= 0.2:
            result["bb"] = "下轨区"
            result["bb_score"] = 1
        else:
            result["bb"] = "中轨区"
            result["bb_score"] = 0

    # 综合得分
    total = (
        result["ma_score"] * 2
        + result["macd_score"]
        + result["rsi_score"]
        + result["kdj_score"]
        + result["trend_score"]
        + result.get("bb_score", 0)
    )
    result["total"] = total

    if total >= 6:
        result["dir"] = "强烈看多"
        result["act"] = "买入"
    elif total >= 3:
        result["dir"] = "看多"
        result["act"] = "可买"
    elif total <= -6:
        result["dir"] = "强烈看空"
        result["act"] = "卖出"
    elif total <= -3:
        result["dir"] = "看空"
        result["act"] = "可卖"
    else:
        result["dir"] = "震荡"
        result["act"] = "观望"

    return result


def analyze_pair(pair_code):
    """分析单个货币对"""
    config = PAIRS[pair_code]
    symbol = config["symbol"]
    name = config["name"]

    print(f"\n{'=' * 70}")
    print(f"蔡森技术分析 - {name} ({symbol})")
    print(f"{'=' * 70}")

    results = {"code": pair_code, "symbol": symbol, "name": name, "timeframes": {}}

    for tf_name, tf_config in TIMEFRAMES.items():
        print(f"\n【{tf_name}】", end=" ")

        df, error = fetch_forex_data(
            symbol, tf_config["interval"], tf_config["outputsize"]
        )

        if df is None or len(df) < 30:
            print(f"数据不足: {error}")
            continue

        analysis = analyze_timeframe(df, tf_name)
        results["timeframes"][tf_name] = analysis

        print(
            f"价格:{analysis['price']:.5f} | 均线:{analysis['ma']} | MACD:{analysis['macd']} | "
            f"RSI:{analysis['rsi']:.1f} | 趋势:{analysis['trend']} | 得分:{analysis['total']} | {analysis['dir']}"
        )

        time.sleep(1)  # 避免速率限制

    return results


def generate_summary(all_results):
    """生成汇总"""
    print(f"\n{'=' * 70}")
    print("多时间框架综合分析汇总")
    print(f"{'=' * 70}")

    summary = []

    for code, data in all_results.items():
        if not data["timeframes"]:
            continue

        # 计算各时间框架得分
        scores = {}
        for tf, analysis in data["timeframes"].items():
            scores[tf] = analysis["total"]

        # 加权综合得分 (较长周期权重更高)
        weights = {
            "1m": 0.05,
            "4m": 0.08,
            "15m": 0.12,
            "30m": 0.15,
            "60m": 0.25,
            "240m": 0.35,
        }
        weighted_score = sum(scores.get(tf, 0) * w for tf, w in weights.items())

        # 统计多空信号
        bullish = sum(1 for s in scores.values() if s > 0)
        bearish = sum(1 for s in scores.values() if s < 0)
        neutral = len(scores) - bullish - bearish

        # 获取240m数据
        tf240 = data["timeframes"].get("240m", {})
        tf60 = data["timeframes"].get("60m", {})

        if weighted_score >= 4:
            direction = "看多"
            action = "买入"
        elif weighted_score >= 1:
            direction = "偏多"
            action = "可买"
        elif weighted_score <= -4:
            direction = "看空"
            action = "卖出"
        elif weighted_score <= -1:
            direction = "偏空"
            action = "可卖"
        else:
            direction = "震荡"
            action = "观望"

        summary.append(
            {
                "code": code,
                "name": data["name"],
                "price": data["timeframes"]
                .get("240m", {})
                .get("price", data["timeframes"].get("60m", {}).get("price", 0)),
                "score_240m": scores.get("240m", 0),
                "score_60m": scores.get("60m", 0),
                "score_30m": scores.get("30m", 0),
                "weighted_score": weighted_score,
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "direction": direction,
                "action": action,
                "near_r": tf240.get("near_r") or tf60.get("near_r"),
                "near_s": tf240.get("near_s") or tf60.get("near_s"),
            }
        )

        print(f"\n{data['name']} ({code})")
        print(f"  价格: {summary[-1]['price']:.5f}")
        print(
            f"  时间框架得分: 240m={scores.get('240m', 'N/A'):2d} | 60m={scores.get('60m', 'N/A'):2d} | "
            f"30m={scores.get('30m', 'N/A'):2d} | 15m={scores.get('15m', 'N/A'):2d}"
        )
        print(
            f"  加权得分: {weighted_score:.2f} | 多头信号:{bullish} | 空头信号:{bearish}"
        )
        print(f"  综合判断: {direction} | 操作: {action}")
        if summary[-1]["near_r"]:
            print(f"  阻力: {summary[-1]['near_r']:.5f}")
        if summary[-1]["near_s"]:
            print(f"  支撑: {summary[-1]['near_s']:.5f}")

    return summary


def generate_trading_plan(summary, all_results):
    """生成交易计划"""
    print(f"\n{'=' * 70}")
    print("下周外汇交易计划")
    print(f"{'=' * 70}")

    plan = {"week": "下周", "generated": datetime.now().isoformat(), "pairs": {}}

    for s in summary:
        code = s["code"]
        name = s["name"]
        price = s["price"]
        score = s["weighted_score"]
        direction = s["direction"]

        pair_plan = {
            "name": name,
            "current_price": price,
            "direction": direction,
            "action": s["action"],
            "weighted_score": score,
        }

        print(f"\n【{name} ({code})】")
        print(f"  当前价格: {price:.5f}")
        print(f"  方向判断: {direction}")

        if score >= 4:  # 看多
            pair_plan["bias"] = "多头"
            pair_plan["entry_type"] = "回调买入"

            # 使用支撑位作为入场参考
            near_s = s["near_s"] or price * 0.998
            pair_plan["entry_zone"] = f"{near_s * 1.0002:.5f} - {near_s * 1.0008:.5f}"
            pair_plan["stop_loss"] = near_s * 0.999

            near_r = s["near_r"] or price * 1.003
            pair_plan["target1"] = near_r * 0.9995
            pair_plan["target2"] = near_r * 1.001

            pair_plan["position"] = "标准仓 (1-2%)"
            pair_plan["confidence"] = "高"

        elif score >= 1:  # 偏多
            pair_plan["bias"] = "偏多"
            pair_plan["entry_type"] = "轻仓买入"

            near_s = s["near_s"] or price * 0.997
            pair_plan["entry_zone"] = f"{near_s * 1.0005:.5f} - {near_s * 1.002:.5f}"
            pair_plan["stop_loss"] = near_s * 0.9985

            near_r = s["near_r"] or price * 1.002
            pair_plan["target1"] = near_r * 0.999
            pair_plan["target2"] = near_r * 1.0005

            pair_plan["position"] = "轻仓 (0.5-1%)"
            pair_plan["confidence"] = "中"

        elif score <= -4:  # 看空
            pair_plan["bias"] = "空头"
            pair_plan["entry_type"] = "反弹卖出"

            near_r = s["near_r"] or price * 1.002
            pair_plan["entry_zone"] = f"{near_r * 0.9992:.5f} - {near_r * 0.9998:.5f}"
            pair_plan["stop_loss"] = near_r * 1.001

            near_s = s["near_s"] or price * 0.997
            pair_plan["target1"] = near_s * 1.0005
            pair_plan["target2"] = near_s * 0.999

            pair_plan["position"] = "标准仓 (1-2%)"
            pair_plan["confidence"] = "高"

        elif score <= -1:  # 偏空
            pair_plan["bias"] = "偏空"
            pair_plan["entry_type"] = "轻仓卖出"

            near_r = s["near_r"] or price * 1.001
            pair_plan["entry_zone"] = f"{near_r * 0.998:.5f} - {near_r * 0.9998:.5f}"
            pair_plan["stop_loss"] = near_r * 1.0015

            near_s = s["near_s"] or price * 0.998
            pair_plan["target1"] = near_s * 1.001
            pair_plan["target2"] = near_s * 0.999

            pair_plan["position"] = "轻仓 (0.5-1%)"
            pair_plan["confidence"] = "中"

        else:  # 震荡
            pair_plan["bias"] = "震荡"
            pair_plan["entry_type"] = "区间操作"

            near_s = s["near_s"] or price * 0.998
            near_r = s["near_r"] or price * 1.002
            pair_plan["entry_zone"] = f"{near_s:.5f} - {near_r:.5f}"
            pair_plan["stop_loss"] = "区间外0.3%"
            pair_plan["target1"] = "区间另一端"
            pair_plan["target2"] = "延展0.2%"
            pair_plan["position"] = "极轻仓或观望"
            pair_plan["confidence"] = "低"

        plan["pairs"][code] = pair_plan

        print(f"  操作建议: {pair_plan['entry_type']}")
        print(f"  入场区域: {pair_plan['entry_zone']}")
        if isinstance(pair_plan["stop_loss"], (int, float)):
            print(f"  止损: {pair_plan['stop_loss']:.5f}")
        else:
            print(f"  止损: {pair_plan['stop_loss']}")
        if isinstance(pair_plan["target1"], (int, float)):
            print(f"  目标1: {pair_plan['target1']:.5f}")
        else:
            print(f"  目标1: {pair_plan['target1']}")
        if isinstance(pair_plan["target2"], (int, float)):
            print(f"  目标2: {pair_plan['target2']:.5f}")
        else:
            print(f"  目标2: {pair_plan['target2']}")
        print(f"  仓位: {pair_plan['position']}")
        print(f"  置信度: {pair_plan['confidence']}")

    return plan


def main():
    """主函数"""
    print(f"{'=' * 70}")
    print(f"蔡森技术分析系统 - 外汇多时间框架分析")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"货币对: EUR, GBP, JPY, AUD, CAD, CNH")
    print(f"时间框架: 1/4/15/30/60/240 分钟")
    print(f"{'=' * 70}")

    all_results = {}

    for code in PAIRS.keys():
        all_results[code] = analyze_pair(code)
        time.sleep(2)  # 避免速率限制

    # 生成汇总
    summary = generate_summary(all_results)

    # 生成交易计划
    plan = generate_trading_plan(summary, all_results)

    # 保存结果
    output = {"analysis": all_results, "summary": summary, "trading_plan": plan}

    output_file = "/root/ideas/caishen/forex_mtf.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n\n结果已保存至: {output_file}")

    return output


if __name__ == "__main__":
    main()
