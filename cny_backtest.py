#!/usr/bin/env python3
"""
蔡森技术分析 - 中国春节假期外汇表现回测
分析过去20年春节期间 EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CNH 的表现
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator

ALPHA_VANTAGE_KEY = "IUO07N60XUPUHNTL"

PAIRS = {
    "EUR": {"from": "EUR", "to": "USD", "name": "欧元/美元"},
    "GBP": {"from": "GBP", "to": "USD", "name": "英镑/美元"},
    "JPY": {"from": "USD", "to": "JPY", "name": "美元/日元"},
    "AUD": {"from": "AUD", "to": "USD", "name": "澳元/美元"},
    "CAD": {"from": "USD", "to": "CAD", "name": "美元/加元"},
    "CNH": {"from": "USD", "to": "CNY", "name": "美元/人民币"},
}

# 过去20年春节日期 ( approximate - CNY falls between Jan 21 - Feb 20)
CNY_DATES = {
    2025: "2025-01-29",
    2024: "2024-02-10",
    2023: "2023-01-22",
    2022: "2022-02-01",
    2021: "2021-02-12",
    2020: "2020-01-25",
    2019: "2019-02-05",
    2018: "2018-02-16",
    2017: "2017-01-28",
    2016: "2016-02-08",
    2015: "2015-02-19",
    2014: "2014-01-31",
    2013: "2013-02-10",
    2012: "2012-01-23",
    2011: "2011-02-03",
    2010: "2010-02-14",
    2009: "2009-01-26",
    2008: "2008-02-07",
    2007: "2007-02-18",
    2006: "2006-01-29",
    2005: "2005-02-09",
}


def fetch_fx_daily_full(from_symbol, to_symbol):
    """获取完整日线历史数据"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "outputsize": "full",
        "apikey": ALPHA_VANTAGE_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=60)
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


def analyze_cny_performance(df, cny_date, pair_name, days_before=5, days_after=10):
    """分析单个春节期间的表现"""
    cny = pd.to_datetime(cny_date)

    results = []

    # 定义分析窗口
    # 节前: CNY前5个交易日
    # 节中: CNY当天及假期 (中国市场休市)
    # 节后: CNY后5-10个交易日

    # 找到最近的有效交易日
    trading_days = df.index

    # 节前5天
    pre_mask = (trading_days < cny) & (trading_days >= cny - timedelta(days=10))
    pre_days = trading_days[pre_mask][-5:] if pre_mask.any() else []

    # 节后10天
    post_mask = (trading_days > cny) & (trading_days <= cny + timedelta(days=20))
    post_days = trading_days[post_mask][:10] if post_mask.any() else []

    if len(pre_days) < 3 or len(post_days) < 3:
        return None

    # 节前价格
    pre_start = df.loc[pre_days[0], "Close"]
    pre_end = df.loc[pre_days[-1], "Close"]
    pre_change = (pre_end - pre_start) / pre_start * 100

    # 节后价格
    post_start = df.loc[post_days[0], "Close"]
    post_end = df.loc[post_days[-1], "Close"]
    post_change = (post_end - post_start) / post_start * 100

    # 跨节价格变化 (节前最后一天 -> 节后第一天)
    cross_start = df.loc[pre_days[-1], "Close"]
    cross_end = df.loc[post_days[0], "Close"]
    cross_change = (cross_end - cross_start) / cross_start * 100

    # 节后3天、5天、10天表现
    post_3d = df.loc[post_days[2], "Close"] if len(post_days) >= 3 else post_end
    post_5d = df.loc[post_days[4], "Close"] if len(post_days) >= 5 else post_end

    change_3d = (post_3d - post_start) / post_start * 100
    change_5d = (post_5d - post_start) / post_start * 100
    change_10d = post_change

    # 计算波动性
    post_prices = df.loc[post_days, "Close"]
    volatility = post_prices.std() / post_prices.mean() * 100

    # 判断方向
    direction = "UP" if post_change > 0 else "DOWN" if post_change < 0 else "FLAT"

    return {
        "year": cny.year,
        "cny_date": cny_date,
        "pre_change": pre_change,
        "cross_change": cross_change,
        "post_3d": change_3d,
        "post_5d": change_5d,
        "post_10d": change_10d,
        "volatility": volatility,
        "direction": direction,
        "pre_start": pre_start,
        "pre_end": pre_end,
        "post_start": post_start,
        "post_end": post_end,
    }


def backtest_pair(code, config):
    """回测单个货币对"""
    print(f"\n{'=' * 60}")
    print(f"回测 {config['name']} ({code}) 春节表现")
    print(f"{'=' * 60}")

    # 获取完整历史数据
    print("获取历史数据...")
    df, err = fetch_fx_daily_full(config["from"], config["to"])

    if df is None:
        print(f"  数据获取失败: {err}")
        return None

    print(
        f"  数据范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"  总交易日: {len(df)}")

    results = []

    for year, cny_date in sorted(CNY_DATES.items(), reverse=True):
        if pd.to_datetime(cny_date) < df.index[0]:
            continue
        if pd.to_datetime(cny_date) > df.index[-1]:
            continue

        perf = analyze_cny_performance(df, cny_date, config["name"])
        if perf:
            results.append(perf)

    if not results:
        print("  无有效春节数据")
        return None

    # 统计分析
    results_df = pd.DataFrame(results)

    up_count = (results_df["direction"] == "UP").sum()
    down_count = (results_df["direction"] == "DOWN").sum()

    avg_post_3d = results_df["post_3d"].mean()
    avg_post_5d = results_df["post_5d"].mean()
    avg_post_10d = results_df["post_10d"].mean()
    avg_cross = results_df["cross_change"].mean()
    avg_volatility = results_df["volatility"].mean()

    print(f"\n  【春节历史表现统计】({len(results)}年数据)")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  上涨次数: {up_count} ({up_count / len(results) * 100:.1f}%)")
    print(f"  下跌次数: {down_count} ({down_count / len(results) * 100:.1f}%)")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  平均跨节涨幅: {avg_cross:+.3f}%")
    print(f"  平均节后3天: {avg_post_3d:+.3f}%")
    print(f"  平均节后5天: {avg_post_5d:+.3f}%")
    print(f"  平均节后10天: {avg_post_10d:+.3f}%")
    print(f"  平均波动性: {avg_volatility:.3f}%")

    # 显示每年详细数据
    print(f"\n  【年度详细数据】")
    print(
        f"  {'年份':<6} {'节前%':>8} {'跨节%':>8} {'节后3d%':>9} {'节后5d%':>9} {'节后10d%':>10} {'方向':>6}"
    )
    print(f"  {'-' * 60}")
    for _, row in results_df.iterrows():
        print(
            f"  {int(row['year']):<6} {row['pre_change']:>+8.3f} {row['cross_change']:>+8.3f} "
            f"{row['post_3d']:>+9.3f} {row['post_5d']:>+9.3f} {row['post_10d']:>+10.3f} {row['direction']:>6}"
        )

    return {
        "code": code,
        "name": config["name"],
        "years_count": len(results),
        "up_count": up_count,
        "down_count": down_count,
        "up_rate": up_count / len(results) * 100,
        "avg_cross": avg_cross,
        "avg_post_3d": avg_post_3d,
        "avg_post_5d": avg_post_5d,
        "avg_post_10d": avg_post_10d,
        "avg_volatility": avg_volatility,
        "yearly_data": results,
    }


def generate_trading_plan(backtest_results):
    """生成春节交易计划"""
    print(f"\n{'=' * 70}")
    print(f"蔡森技术分析 - 2026年春节外汇交易计划")
    print(f"{'=' * 70}")

    # 按胜率排序
    sorted_results = sorted(
        [r for r in backtest_results.values() if r],
        key=lambda x: x["up_rate"] if x["avg_post_5d"] > 0 else 100 - x["up_rate"],
        reverse=True,
    )

    print(f"\n【春节历史表现排名】")
    print(
        f"{'货币对':<12} {'上涨率':>8} {'平均跨节%':>10} {'平均节后5d%':>12} {'推荐方向':>10}"
    )
    print(f"{'-' * 60}")

    for r in sorted_results:
        if r["avg_post_5d"] > 0:
            direction = "做多"
            confidence = r["up_rate"]
        else:
            direction = "做空"
            confidence = 100 - r["up_rate"]

        print(
            f"{r['name']:<12} {r['up_rate']:>7.1f}% {r['avg_cross']:>+10.3f}% "
            f"{r['avg_post_5d']:>+12.3f}% {direction:>10}"
        )

    # 生成具体交易计划
    print(f"\n{'=' * 70}")
    print(f"【2026年春节交易计划】")
    print(f"春节日期: 2026年2月17日 (农历正月初一)")
    print(f"建议持仓时间: 节前1-2天入场, 节后3-5天出场")
    print(f"{'=' * 70}")

    plan = {"cny_date": "2026-02-17", "pairs": {}}

    for r in sorted_results:
        pair_plan = {
            "name": r["name"],
            "years_analyzed": r["years_count"],
            "avg_post_5d": r["avg_post_5d"],
        }

        if r["avg_post_5d"] > 0.5 and r["up_rate"] >= 60:
            # 强烈看多
            pair_plan["direction"] = "做多"
            pair_plan["confidence"] = r["up_rate"]
            pair_plan["entry"] = "节前1-2天"
            pair_plan["exit"] = "节后3-5天"
            pair_plan["position"] = "标准仓 (1-2%)"
            pair_plan["action"] = "BUY"

        elif r["avg_post_5d"] > 0 and r["up_rate"] >= 55:
            # 偏多
            pair_plan["direction"] = "偏多"
            pair_plan["confidence"] = r["up_rate"]
            pair_plan["entry"] = "节前1天"
            pair_plan["exit"] = "节后3天"
            pair_plan["position"] = "轻仓 (0.5-1%)"
            pair_plan["action"] = "LIGHT_BUY"

        elif r["avg_post_5d"] < -0.5 and r["up_rate"] <= 40:
            # 强烈看空
            pair_plan["direction"] = "做空"
            pair_plan["confidence"] = 100 - r["up_rate"]
            pair_plan["entry"] = "节前1-2天"
            pair_plan["exit"] = "节后3-5天"
            pair_plan["position"] = "标准仓 (1-2%)"
            pair_plan["action"] = "SELL"

        elif r["avg_post_5d"] < 0 and r["up_rate"] <= 45:
            # 偏空
            pair_plan["direction"] = "偏空"
            pair_plan["confidence"] = 100 - r["up_rate"]
            pair_plan["entry"] = "节前1天"
            pair_plan["exit"] = "节后3天"
            pair_plan["position"] = "轻仓 (0.5-1%)"
            pair_plan["action"] = "LIGHT_SELL"

        else:
            # 震荡
            pair_plan["direction"] = "震荡"
            pair_plan["confidence"] = 50
            pair_plan["entry"] = "-"
            pair_plan["exit"] = "-"
            pair_plan["position"] = "观望"
            pair_plan["action"] = "HOLD"

        plan["pairs"][r["code"]] = pair_plan

        print(f"\n【{r['name']} ({r['code']})】")
        print(f"  历史数据: {r['years_count']}年")
        print(f"  节后5天平均涨幅: {r['avg_post_5d']:+.3f}%")
        print(f"  上涨概率: {r['up_rate']:.1f}%")
        print(
            f"  ━━━ 推荐方向: {pair_plan['direction']} (置信度: {pair_plan['confidence']:.0f}%) ━━━"
        )
        print(f"  入场时机: {pair_plan['entry']}")
        print(f"  出场时机: {pair_plan['exit']}")
        print(f"  仓位建议: {pair_plan['position']}")

    return plan


def main():
    print(f"{'=' * 70}")
    print(f"蔡森技术分析 - 中国春节假期外汇表现回测")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"货币对: EUR, GBP, JPY, AUD, CAD, CNH")
    print(f"回测周期: 过去20年")
    print(f"{'=' * 70}")

    backtest_results = {}

    for code, config in PAIRS.items():
        result = backtest_pair(code, config)
        if result:
            backtest_results[code] = result
        time.sleep(15)  # Alpha Vantage API限制

    if backtest_results:
        plan = generate_trading_plan(backtest_results)

        # 保存结果
        output = {"backtest_results": backtest_results, "trading_plan": plan}

        with open("/root/ideas/caishen/cny_backtest.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n\n结果已保存至: /root/ideas/caishen/cny_backtest.json")

    return backtest_results


if __name__ == "__main__":
    main()
