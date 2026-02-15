#!/usr/bin/env python3
"""
蔡森技术分析系统 - 综合市场分析
涵盖: 全球指数、商品、外汇、加密货币
"""

import json
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

MARKETS = {
    "indices": {
        "SPX": {"yf": "^GSPC", "name": "S&P 500"},
        "ND100": {"yf": "^NDX", "name": "Nasdaq 100"},
        "DJI": {"yf": "^DJI", "name": "Dow Jones"},
        "JP225": {"yf": "^N225", "name": "Nikkei 225"},
        "DAX": {"yf": "^GDAXI", "name": "DAX 40"},
        "FTSE": {"yf": "^FTSE", "name": "FTSE 100"},
        "CAC": {"yf": "^FCHI", "name": "CAC 40"},
        "HSI": {"yf": "^HSI", "name": "Hang Seng"},
        "ASX": {"yf": "^AXJO", "name": "ASX 200"},
        "SSEC": {"yf": "000001.SS", "name": "Shanghai Composite"},
    },
    "commodities": {
        "XAU": {"yf": "GC=F", "name": "Gold"},
        "XAG": {"yf": "SI=F", "name": "Silver"},
        "WTI": {"yf": "CL=F", "name": "Crude Oil WTI"},
        "BRENT": {"yf": "BZ=F", "name": "Brent Crude"},
        "NG": {"yf": "NG=F", "name": "Natural Gas"},
        "COPPER": {"yf": "HG=F", "name": "Copper"},
    },
    "crypto": {
        "BTC": {"yf": "BTC-USD", "name": "Bitcoin"},
        "ETH": {"yf": "ETH-USD", "name": "Ethereum"},
    },
    "forex": {
        "EURUSD": {"yf": "EURUSD=X", "name": "EUR/USD"},
        "GBPUSD": {"yf": "GBPUSD=X", "name": "GBP/USD"},
        "USDJPY": {"yf": "USDJPY=X", "name": "USD/JPY"},
        "AUDUSD": {"yf": "AUDUSD=X", "name": "AUD/USD"},
        "USDCAD": {"yf": "USDCAD=X", "name": "USD/CAD"},
        "USDCNH": {"yf": "USDCNH=X", "name": "USD/CNH"},
    },
}


def fetch_data(symbol, interval="1d", period="1y", retry=3):
    for attempt in range(retry):
        try:
            time.sleep(5)
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return None
            df = df.sort_index()
            return df
        except Exception as e:
            print(f"  Error (attempt {attempt + 1}/{retry}): {e}")
            if attempt < retry - 1:
                time.sleep(10)
    return None


def calc_indicators(df):
    df = df.copy()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Sig"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    rsi = RSIIndicator(close=df["Close"])
    df["RSI"] = rsi.rsi()

    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"])
    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    bb = BollingerBands(close=df["Close"])
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Mid"] = bb.bollinger_mavg()

    return df


def find_pivots(df, lookback=60):
    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values

    resist = []
    suppt = []
    for i in range(2, len(highs) - 2):
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

    return sorted(set(resist), reverse=True), sorted(set(suppt), reverse=True)


def fib_levels(high, low):
    diff = high - low
    return {
        "0.236": high - diff * 0.236,
        "0.382": high - diff * 0.382,
        "0.5": high - diff * 0.5,
        "0.618": high - diff * 0.618,
        "0.786": high - diff * 0.786,
    }


def analyze(df, tf):
    df = calc_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    result = {"tf": tf, "price": float(last["Close"]), "date": str(last.name)}

    # 均线
    if pd.notna(last["MA20"]):
        if last["Close"] > last["MA5"] > last["MA10"] > last["MA20"]:
            result["ma"] = "多头排列"
            result["ma_score"] = 2
        elif last["Close"] < last["MA5"] < last["MA10"] < last["MA20"]:
            result["ma"] = "空头排列"
            result["ma_score"] = -2
        else:
            result["ma"] = "均线纠缠"
            result["ma_score"] = 0
        result["ma5"] = float(last["MA5"]) if pd.notna(last["MA5"]) else None
        result["ma10"] = float(last["MA10"]) if pd.notna(last["MA10"]) else None
        result["ma20"] = float(last["MA20"])
    else:
        result["ma"] = "N/A"
        result["ma_score"] = 0

    # MACD
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
    result["macd_pos"] = "零轴上" if last["MACD"] > 0 else "零轴下"
    result["macd_val"] = float(last["MACD"])
    result["macd_sig"] = float(last["MACD_Sig"])
    result["macd_hist"] = float(last["MACD_Hist"])

    # RSI
    rsi = last["RSI"]
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
        result["rsi_score"] = 1

    # KDJ
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
    result["k"] = float(last["K"]) if pd.notna(last["K"]) else None
    result["d"] = float(last["D"]) if pd.notna(last["D"]) else None
    result["j"] = float(last["J"]) if pd.notna(last["J"]) else None

    # 布林带
    result["bb_upper"] = float(last["BB_Upper"])
    result["bb_lower"] = float(last["BB_Lower"])
    result["bb_mid"] = float(last["BB_Mid"])
    bb_pos = (last["Close"] - last["BB_Lower"]) / (last["BB_Upper"] - last["BB_Lower"])
    result["bb_pos"] = float(bb_pos)
    if bb_pos > 0.8:
        result["bb_status"] = "上轨附近"
    elif bb_pos < 0.2:
        result["bb_status"] = "下轨附近"
    else:
        result["bb_status"] = "中轨附近"

    # 趋势
    recent = df.tail(60)
    highs = recent["High"].values
    lows = recent["Low"].values
    hp = [
        highs[i]
        for i in range(2, len(highs) - 2)
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]
    ]
    lp = [
        lows[i]
        for i in range(2, len(lows) - 2)
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]
    ]

    if len(hp) >= 2 and len(lp) >= 2:
        if hp[-1] > hp[-2] and lp[-1] > lp[-2]:
            result["trend"] = "上升"
            result["trend_score"] = 2
        elif hp[-1] < hp[-2] and lp[-1] < lp[-2]:
            result["trend"] = "下降"
            result["trend_score"] = -2
        else:
            result["trend"] = "震荡"
            result["trend_score"] = 0
    else:
        result["trend"] = "不明"
        result["trend_score"] = 0

    # 支撑阻力
    resist, suppt = find_pivots(df)
    resist = resist[:5]
    suppt = suppt[:5]

    nr = next((r for r in resist if r > last["Close"]), None)
    ns = next((s for s in reversed(suppt) if s < last["Close"]), None)

    result["resistance"] = [float(x) for x in resist]
    result["support"] = [float(x) for x in suppt]
    result["near_r"] = float(nr) if nr else None
    result["near_s"] = float(ns) if ns else None

    # 斐波那契
    if len(df) >= 60:
        high60 = df["High"].tail(60).max()
        low60 = df["Low"].tail(60).min()
        result["fib"] = {k: float(v) for k, v in fib_levels(high60, low60).items()}
        result["high60"] = float(high60)
        result["low60"] = float(low60)

    # 成交量
    vol_ma = df["Volume"].tail(20).mean()
    if pd.notna(vol_ma) and vol_ma > 0:
        result["vol_ratio"] = float(last["Volume"] / vol_ma)
        result["vol_status"] = (
            "放量"
            if last["Volume"] > vol_ma * 1.5
            else ("缩量" if last["Volume"] < vol_ma * 0.7 else "正常")
        )
    else:
        result["vol_ratio"] = None
        result["vol_status"] = "N/A"

    # 综合得分
    total = (
        result["ma_score"] * 2
        + result["macd_score"]
        + result["rsi_score"]
        + result["kdj_score"]
        + result["trend_score"]
    )
    result["total"] = total

    if total >= 6:
        result["dir"] = "强烈看多"
        result["act"] = "积极买入"
    elif total >= 4:
        result["dir"] = "看多"
        result["act"] = "买入"
    elif total >= 2:
        result["dir"] = "偏多"
        result["act"] = "可买"
    elif total <= -6:
        result["dir"] = "强烈看空"
        result["act"] = "积极卖出"
    elif total <= -4:
        result["dir"] = "看空"
        result["act"] = "卖出"
    elif total <= -2:
        result["dir"] = "偏空"
        result["act"] = "可卖"
    else:
        result["dir"] = "震荡"
        result["act"] = "观望"

    return result


def gen_trading_plan(sym, name, daily, hourly=None):
    plan = {"symbol": sym, "name": name}

    if not daily:
        return None

    price = daily["price"]
    score = daily["total"]
    nr = daily.get("near_r")
    ns = daily.get("near_s")
    ma20 = daily.get("ma20")
    fib = daily.get("fib", {})

    plan["current_price"] = price
    plan["score"] = score
    plan["direction"] = daily["dir"]
    plan["action"] = daily["act"]

    if score >= 4:
        plan["bias"] = "LONG"
        plan["confidence"] = "HIGH" if score >= 6 else "MEDIUM"

        if ns:
            plan["entry_zone"] = [round(ns * 1.002, 4), round(ns * 1.01, 4)]
            plan["stop_loss"] = round(ns * 0.985, 4)
        else:
            plan["entry_zone"] = [round(price * 0.995, 4), round(price * 1.005, 4)]
            plan["stop_loss"] = round(price * 0.97, 4)

        targets = []
        if nr:
            targets.append(round(nr * 0.995, 4))
            targets.append(round(nr * 1.01, 4))
        if fib.get("0.618"):
            targets.append(round(fib["0.618"], 4))
        if fib.get("0.5"):
            targets.append(round(fib["0.5"], 4))
        plan["targets"] = list(dict.fromkeys(targets))[:3]

        plan["position_size"] = "Standard (2-3%)"
        plan["risk_reward"] = "1:2 or better"

    elif score >= 2:
        plan["bias"] = "LONG"
        plan["confidence"] = "LOW"

        if ns:
            plan["entry_zone"] = [round(ns * 1.005, 4), round(ns * 1.015, 4)]
            plan["stop_loss"] = round(ns * 0.98, 4)
        else:
            plan["entry_zone"] = [round(price * 0.99, 4), round(price * 1.01, 4)]
            plan["stop_loss"] = round(price * 0.975, 4)

        targets = []
        if nr:
            targets.append(round(nr, 4))
        plan["targets"] = targets[:2]

        plan["position_size"] = "Light (1-2%)"
        plan["risk_reward"] = "1:1.5"

    elif score <= -4:
        plan["bias"] = "SHORT"
        plan["confidence"] = "HIGH" if score <= -6 else "MEDIUM"

        if nr:
            plan["entry_zone"] = [round(nr * 0.99, 4), round(nr * 0.998, 4)]
            plan["stop_loss"] = round(nr * 1.015, 4)
        else:
            plan["entry_zone"] = [round(price * 0.995, 4), round(price * 1.005, 4)]
            plan["stop_loss"] = round(price * 1.03, 4)

        targets = []
        if ns:
            targets.append(round(ns * 1.005, 4))
            targets.append(round(ns * 0.99, 4))
        if fib.get("0.382"):
            targets.append(round(fib["0.382"], 4))
        plan["targets"] = list(dict.fromkeys(targets))[:3]

        plan["position_size"] = "Standard (2-3%)"
        plan["risk_reward"] = "1:2 or better"

    elif score <= -2:
        plan["bias"] = "SHORT"
        plan["confidence"] = "LOW"

        if nr:
            plan["entry_zone"] = [round(nr * 0.985, 4), round(nr * 1.0, 4)]
            plan["stop_loss"] = round(nr * 1.02, 4)
        else:
            plan["entry_zone"] = [round(price * 0.99, 4), round(price * 1.01, 4)]
            plan["stop_loss"] = round(price * 1.025, 4)

        targets = []
        if ns:
            targets.append(round(ns, 4))
        plan["targets"] = targets[:2]

        plan["position_size"] = "Light (1-2%)"
        plan["risk_reward"] = "1:1.5"

    else:
        plan["bias"] = "NEUTRAL"
        plan["confidence"] = "N/A"

        if ns and nr:
            plan["range"] = [round(ns, 4), round(nr, 4)]
            plan["range_strategy"] = "Buy at support, sell at resistance"

        plan["position_size"] = "Very Light or None"
        plan["entry_zone"] = None
        plan["stop_loss"] = None
        plan["targets"] = []

    if hourly:
        plan["hourly_trend"] = hourly["dir"]
        plan["hourly_score"] = hourly["total"]
        if hourly["dir"] == daily["dir"]:
            plan["timeframe_alignment"] = "ALIGNED - High confidence"
        else:
            plan["timeframe_alignment"] = "MISALIGNED - Reduce position size"

    plan["key_levels"] = {
        "resistance": daily.get("resistance", [])[:3],
        "support": daily.get("support", [])[:3],
        "fib_levels": fib,
    }

    return plan


def main():
    results = {"generated_at": datetime.now().isoformat(), "markets": {}}
    trading_plans = {"generated_at": datetime.now().isoformat(), "plans": {}}

    for market_type, symbols in MARKETS.items():
        print(f"\n{'=' * 70}")
        print(f"蔡森技术分析 - {market_type.upper()}")
        print(f"{'=' * 70}")

        results["markets"][market_type] = {}
        trading_plans["plans"][market_type] = {}

        for sym, cfg in symbols.items():
            print(f"\n--- {cfg['name']} ({sym}) ---")

            yf_sym = cfg["yf"]
            results["markets"][market_type][sym] = {"name": cfg["name"], "tfs": {}}

            daily_df = None
            hourly_df = None

            print(f"  Fetching daily...")
            daily_df = fetch_data(yf_sym, "1d", "1y")

            if daily_df is not None and len(daily_df) >= 30:
                daily_r = analyze(daily_df, "daily")
                results["markets"][market_type][sym]["tfs"]["daily"] = daily_r

                print(
                    f"  Daily: {daily_r['price']:,.2f} | {daily_r['dir']} | Score: {daily_r['total']}"
                )

            print(f"  Fetching hourly...")
            hourly_df = fetch_data(yf_sym, "1h", "1mo")

            if hourly_df is not None and len(hourly_df) >= 30:
                hourly_r = analyze(hourly_df, "1h")
                results["markets"][market_type][sym]["tfs"]["1h"] = hourly_r

                print(
                    f"  1H: {hourly_r['price']:,.2f} | {hourly_r['dir']} | Score: {hourly_r['total']}"
                )

            if daily_df is not None and len(daily_df) >= 30:
                daily_r = results["markets"][market_type][sym]["tfs"].get("daily")
                hourly_r = results["markets"][market_type][sym]["tfs"].get("1h")
                plan = gen_trading_plan(sym, cfg["name"], daily_r, hourly_r)
                if plan:
                    trading_plans["plans"][market_type][sym] = plan

    with open("/root/ideas/caishen/comprehensive_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open("/root/ideas/caishen/trading_plans.json", "w") as f:
        json.dump(trading_plans, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("分析完成!")
    print(f"{'=' * 70}")
    print(f"结果保存至: comprehensive_analysis.json, trading_plans.json")


if __name__ == "__main__":
    main()
