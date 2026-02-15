#!/usr/bin/env python3
"""
蔡森技术分析系统 - 外汇多时间框架分析
使用 Alpha Vantage API (更稳定的免费API)
分析 EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CNH
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

ALPHA_VANTAGE_KEY = "IUO07N60XUPUHNTL"

PAIRS = {
    "EUR": {"from": "EUR", "to": "USD", "name": "欧元/美元"},
    "GBP": {"from": "GBP", "to": "USD", "name": "英镑/美元"},
    "JPY": {"from": "USD", "to": "JPY", "name": "美元/日元"},
    "AUD": {"from": "AUD", "to": "USD", "name": "澳元/美元"},
    "CAD": {"from": "USD", "to": "CAD", "name": "美元/加元"},
    "CNH": {"from": "USD", "to": "CNY", "name": "美元/人民币"},
}


def fetch_fx_intraday(from_symbol, to_symbol, interval="60min"):
    """获取外汇日内数据"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "interval": interval,
        "outputsize": "compact",
        "apikey": ALPHA_VANTAGE_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        key = f"Time Series FX ({interval})"
        if key not in data:
            return None, data.get("Note", data.get("Error Message", "No data"))

        ts = data[key]
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df.columns = ["Open", "High", "Low", "Close"]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        return df, None

    except Exception as e:
        return None, str(e)


def fetch_fx_daily(from_symbol, to_symbol):
    """获取外汇日线数据"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "outputsize": "compact",
        "apikey": ALPHA_VANTAGE_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if "Time Series FX (Daily)" not in data:
            return None, data.get("Note", data.get("Error Message", "No data"))

        ts = data["Time Series FX (Daily)"]
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df.columns = ["Open", "High", "Low", "Close"]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        return df, None

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
        macd = MACD(close=df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Sig"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()
    except:
        df["MACD"] = 0
        df["MACD_Sig"] = 0

    try:
        rsi = RSIIndicator(close=df["Close"])
        df["RSI"] = rsi.rsi()
    except:
        df["RSI"] = 50

    try:
        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"])
        df["K"] = stoch.stoch()
        df["D"] = stoch.stoch_signal()
        df["J"] = 3 * df["K"] - 2 * df["D"]
    except:
        df["K"] = 50
        df["D"] = 50

    try:
        bb = BollingerBands(close=df["Close"])
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()
    except:
        pass

    return df


def analyze_df(df, tf_name):
    """分析数据"""
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
        elif last["Close"] < last["MA5"] < last["MA10"] < last["MA20"]:
            result["ma"] = "空头排列"
            result["ma_score"] = -2
        elif last["MA5"] > last["MA10"] > last["MA20"]:
            result["ma"] = "偏多排列"
            result["ma_score"] = 1
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

    result["macd_pos"] = "零轴上" if last["MACD"] > 0 else "零轴下"

    # RSI
    rsi = last["RSI"] if pd.notna(last["RSI"]) else 50
    result["rsi"] = float(rsi)
    if rsi >= 70:
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
    resist = []
    suppt = []
    for i in range(2, len(recent) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            resist.append(highs[i])
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            suppt.append(lows[i])

    resist = sorted(set(resist), reverse=True)[:3]
    suppt = sorted(set(suppt), reverse=True)[:3]

    nr = next((r for r in resist if r > last["Close"]), None)
    ns = next((s for s in reversed(suppt) if s < last["Close"]), None)

    result["near_r"] = float(nr) if nr else None
    result["near_s"] = float(ns) if ns else None

    # 综合得分
    total = (
        result["ma_score"] * 2
        + result["macd_score"]
        + result["rsi_score"]
        + result["kdj_score"]
        + result["trend_score"]
    )
    result["total"] = total

    if total >= 5:
        result["dir"] = "看多"
        result["act"] = "买入"
    elif total >= 2:
        result["dir"] = "偏多"
        result["act"] = "可买"
    elif total <= -5:
        result["dir"] = "看空"
        result["act"] = "卖出"
    elif total <= -2:
        result["dir"] = "偏空"
        result["act"] = "可卖"
    else:
        result["dir"] = "震荡"
        result["act"] = "观望"

    return result


def analyze_pair(code, config):
    """分析单个货币对"""
    print(f"\n{'=' * 60}")
    print(f"蔡森技术分析 - {config['name']} ({code})")
    print(f"{'=' * 60}")

    results = {"code": code, "name": config["name"], "timeframes": {}}

    # 获取60分钟数据
    print("【60分钟】", end=" ")
    df60, err = fetch_fx_intraday(config["from"], config["to"], "60min")
    if df60 is not None and len(df60) >= 30:
        results["timeframes"]["60m"] = analyze_df(df60, "60m")
        r = results["timeframes"]["60m"]
        print(
            f"价格:{r['price']:.5f} | 均线:{r['ma']} | MACD:{r['macd']} | RSI:{r['rsi']:.1f} | 得分:{r['total']} | {r['dir']}"
        )
    else:
        print(f"数据不足: {err}")
    time.sleep(12)  # Alpha Vantage 限制: 5 calls/min

    # 获取30分钟数据
    print("【30分钟】", end=" ")
    df30, err = fetch_fx_intraday(config["from"], config["to"], "30min")
    if df30 is not None and len(df30) >= 30:
        results["timeframes"]["30m"] = analyze_df(df30, "30m")
        r = results["timeframes"]["30m"]
        print(
            f"价格:{r['price']:.5f} | 均线:{r['ma']} | MACD:{r['macd']} | RSI:{r['rsi']:.1f} | 得分:{r['total']} | {r['dir']}"
        )
    else:
        print(f"数据不足: {err}")
    time.sleep(12)

    # 获取15分钟数据
    print("【15分钟】", end=" ")
    df15, err = fetch_fx_intraday(config["from"], config["to"], "15min")
    if df15 is not None and len(df15) >= 30:
        results["timeframes"]["15m"] = analyze_df(df15, "15m")
        r = results["timeframes"]["15m"]
        print(
            f"价格:{r['price']:.5f} | 均线:{r['ma']} | MACD:{r['macd']} | RSI:{r['rsi']:.1f} | 得分:{r['total']} | {r['dir']}"
        )
    else:
        print(f"数据不足: {err}")
    time.sleep(12)

    # 获取日线数据
    print("【日线】", end=" ")
    dfd, err = fetch_fx_daily(config["from"], config["to"])
    if dfd is not None and len(dfd) >= 30:
        results["timeframes"]["daily"] = analyze_df(dfd, "daily")
        r = results["timeframes"]["daily"]
        print(
            f"价格:{r['price']:.5f} | 均线:{r['ma']} | MACD:{r['macd']} | RSI:{r['rsi']:.1f} | 得分:{r['total']} | {r['dir']}"
        )
    else:
        print(f"数据不足: {err}")
    time.sleep(12)

    return results


def generate_summary_and_plan(all_results):
    """生成汇总和交易计划"""
    print(f"\n{'=' * 60}")
    print("蔡森技术分析 - 下周外汇交易计划")
    print(f"{'=' * 60}")

    summary = []

    for code, data in all_results.items():
        tfs = data.get("timeframes", {})
        if not tfs:
            continue

        # 获取各时间框架得分
        scores = {tf: a["total"] for tf, a in tfs.items()}

        # 加权得分
        weights = {"15m": 0.15, "30m": 0.25, "60m": 0.35, "daily": 0.25}
        weighted = sum(scores.get(tf, 0) * w for tf, w in weights.items())

        # 统计多空
        bullish = sum(1 for s in scores.values() if s > 0)
        bearish = sum(1 for s in scores.values() if s < 0)

        # 获取关键价位
        daily = tfs.get("daily", {})
        h60 = tfs.get("60m", {})
        price = daily.get("price") or h60.get("price", 0)
        near_r = daily.get("near_r") or h60.get("near_r")
        near_s = daily.get("near_s") or h60.get("near_s")

        if weighted >= 4:
            direction = "看多"
            action = "买入"
        elif weighted >= 1:
            direction = "偏多"
            action = "可买"
        elif weighted <= -4:
            direction = "看空"
            action = "卖出"
        elif weighted <= -1:
            direction = "偏空"
            action = "可卖"
        else:
            direction = "震荡"
            action = "观望"

        summary.append(
            {
                "code": code,
                "name": data["name"],
                "price": price,
                "score_60m": scores.get("60m", 0),
                "score_30m": scores.get("30m", 0),
                "score_15m": scores.get("15m", 0),
                "score_daily": scores.get("daily", 0),
                "weighted": weighted,
                "direction": direction,
                "action": action,
                "near_r": near_r,
                "near_s": near_s,
            }
        )

        print(f"\n【{data['name']} ({code})】")
        print(f"  当前价格: {price:.5f}")
        print(
            f"  时间框架得分: 日线={scores.get('daily', 'N/A')} | 60m={scores.get('60m', 'N/A')} | 30m={scores.get('30m', 'N/A')} | 15m={scores.get('15m', 'N/A')}"
        )
        print(f"  加权得分: {weighted:.2f} | 多头:{bullish} | 空头:{bearish}")
        print(f"  综合判断: {direction} | 操作: {action}")

        # 交易计划
        if weighted >= 4:
            entry_s = near_s or price * 0.998
            print(f"  ━━━ 交易计划 ━━━")
            print(f"  方向: 做多")
            print(f"  入场: {entry_s * 1.0005:.5f} - {entry_s * 1.002:.5f}")
            print(f"  止损: {entry_s * 0.998:.5f}")
            if near_r:
                print(f"  目标: {near_r * 0.999:.5f} -> {near_r * 1.001:.5f}")
            print(f"  仓位: 标准(1-2%)")
        elif weighted >= 1:
            entry_s = near_s or price * 0.997
            print(f"  ━━━ 交易计划 ━━━")
            print(f"  方向: 偏多")
            print(f"  入场: {entry_s * 1.001:.5f} - {entry_s * 1.003:.5f}")
            print(f"  止损: {entry_s * 0.997:.5f}")
            print(f"  仓位: 轻仓(0.5-1%)")
        elif weighted <= -4:
            entry_r = near_r or price * 1.002
            print(f"  ━━━ 交易计划 ━━━")
            print(f"  方向: 做空")
            print(f"  入场: {entry_r * 0.998:.5f} - {entry_r * 0.9995:.5f}")
            print(f"  止损: {entry_r * 1.002:.5f}")
            if near_s:
                print(f"  目标: {near_s * 1.001:.5f} -> {near_s * 0.999:.5f}")
            print(f"  仓位: 标准(1-2%)")
        elif weighted <= -1:
            entry_r = near_r or price * 1.001
            print(f"  ━━━ 交易计划 ━━━")
            print(f"  方向: 偏空")
            print(f"  入场: {entry_r * 0.999:.5f} - {entry_r * 1.0:.5f}")
            print(f"  止损: {entry_r * 1.003:.5f}")
            print(f"  仓位: 轻仓(0.5-1%)")
        else:
            print(f"  ━━━ 交易计划 ━━━")
            print(f"  方向: 震荡 - 观望")
            if near_s and near_r:
                print(f"  区间: {near_s:.5f} - {near_r:.5f}")
            print(f"  仓位: 极轻仓或观望")

        if near_r:
            print(f"  阻力: {near_r:.5f}")
        if near_s:
            print(f"  支撑: {near_s:.5f}")

    return summary


def main():
    print(f"{'=' * 60}")
    print(f"蔡森技术分析系统 - 外汇多时间框架分析")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"货币对: EUR, GBP, JPY, AUD, CAD, CNH")
    print(f"时间框架: 15m/30m/60m/日线")
    print(f"{'=' * 60}")

    all_results = {}

    for code, config in PAIRS.items():
        all_results[code] = analyze_pair(code, config)

    summary = generate_summary_and_plan(all_results)

    # 保存结果
    output = {"analysis": all_results, "summary": summary}

    with open("/root/ideas/caishen/forex_mtf.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n\n结果已保存至: /root/ideas/caishen/forex_mtf.json")


if __name__ == "__main__":
    main()
