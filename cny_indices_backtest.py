#!/usr/bin/env python3
"""
蔡森技术分析 - 中国春节假期全球股指表现回测
分析过去20年春节期间 SPX, ND100, DJI, JP225, DAX, HSI, IWM 的表现
"""

import json
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

INDICES = {
    "SPX": {"symbol": "^GSPC", "name": "S&P 500", "market": "US"},
    "ND100": {"symbol": "^NDX", "name": "Nasdaq 100", "market": "US"},
    "DJI": {"symbol": "^DJI", "name": "Dow Jones", "market": "US"},
    "JP225": {"symbol": "^N225", "name": "Nikkei 225", "market": "JP"},
    "DAX": {"symbol": "^GDAXI", "name": "DAX 40", "market": "DE"},
    "HSI": {"symbol": "^HSI", "name": "Hang Seng", "market": "HK"},
    "IWM": {"symbol": "IWM", "name": "Russell 2000", "market": "US"},
}

# 过去20年春节日期
CNY_DATES = {
    2026: "2026-02-17",
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


def fetch_index_data(symbol, start_date="2004-01-01"):
    """获取指数历史数据"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, interval="1d")

        if df.empty:
            return None

        df = df.sort_index()
        # 移除时区信息，统一使用无时区的datetime
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return None


def analyze_cny_performance(df, cny_date, index_name, days_before=5, days_after=10):
    """分析单个春节期间的表现"""
    cny = pd.to_datetime(cny_date)

    trading_days = df.index

    # 节前5天 (中国市场休市前的交易日)
    pre_mask = (trading_days < cny) & (trading_days >= cny - timedelta(days=15))
    pre_days = trading_days[pre_mask][-5:] if pre_mask.any() else []

    # 节后10天 (中国市场重新开市后的交易日)
    post_mask = (trading_days > cny) & (trading_days <= cny + timedelta(days=25))
    post_days = trading_days[post_mask][:10] if post_mask.any() else []

    if len(pre_days) < 3 or len(post_days) < 3:
        return None

    # 节前价格变化
    pre_start = df.loc[pre_days[0], "Close"]
    pre_end = df.loc[pre_days[-1], "Close"]
    pre_change = (pre_end - pre_start) / pre_start * 100

    # 跨节价格变化 (节前最后一天 -> 节后第一天)
    cross_start = df.loc[pre_days[-1], "Close"]
    cross_end = df.loc[post_days[0], "Close"]
    cross_change = (cross_end - cross_start) / cross_start * 100

    # 节后3天、5天、10天表现
    post_start = df.loc[post_days[0], "Close"]
    post_3d = (
        df.loc[post_days[2], "Close"]
        if len(post_days) >= 3
        else df.loc[post_days[-1], "Close"]
    )
    post_5d = (
        df.loc[post_days[4], "Close"]
        if len(post_days) >= 5
        else df.loc[post_days[-1], "Close"]
    )
    post_10d = df.loc[post_days[-1], "Close"]

    change_3d = (post_3d - post_start) / post_start * 100
    change_5d = (post_5d - post_start) / post_start * 100
    change_10d = (post_10d - post_start) / post_start * 100

    # 最大回撤和最大涨幅
    post_prices = df.loc[post_days, "Close"]
    max_price = post_prices.max()
    min_price = post_prices.min()
    max_gain = (max_price - post_start) / post_start * 100
    max_drawdown = (min_price - post_start) / post_start * 100

    # 波动性
    volatility = post_prices.std() / post_prices.mean() * 100

    # 判断方向
    if change_5d > 0.5:
        direction = "UP"
    elif change_5d < -0.5:
        direction = "DOWN"
    else:
        direction = "FLAT"

    return {
        "year": cny.year,
        "cny_date": cny_date,
        "pre_change": pre_change,
        "cross_change": cross_change,
        "post_3d": change_3d,
        "post_5d": change_5d,
        "post_10d": change_10d,
        "max_gain": max_gain,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "direction": direction,
        "pre_start": float(pre_start),
        "pre_end": float(pre_end),
        "post_start": float(post_start),
        "post_end": float(post_10d),
    }


def backtest_index(code, config):
    """回测单个指数"""
    print(f"\n{'=' * 70}")
    print(f"回测 {config['name']} ({code}) 春节表现")
    print(f"{'=' * 70}")

    # 获取历史数据
    print("获取历史数据...")
    df = fetch_index_data(config["symbol"])

    if df is None or df.empty:
        print(f"  数据获取失败")
        return None

    print(
        f"  数据范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"  总交易日: {len(df)}")

    results = []

    for year, cny_date in sorted(CNY_DATES.items(), reverse=True):
        cny_ts = pd.to_datetime(cny_date)
        if cny_ts < df.index[0]:
            continue
        if cny_ts > df.index[-1]:
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
    flat_count = (results_df["direction"] == "FLAT").sum()

    avg_post_3d = results_df["post_3d"].mean()
    avg_post_5d = results_df["post_5d"].mean()
    avg_post_10d = results_df["post_10d"].mean()
    avg_cross = results_df["cross_change"].mean()
    avg_volatility = results_df["volatility"].mean()
    avg_max_gain = results_df["max_gain"].mean()
    avg_max_drawdown = results_df["max_drawdown"].mean()

    # 胜率计算 (节后5天上涨为胜)
    win_count = (results_df["post_5d"] > 0).sum()
    win_rate = win_count / len(results) * 100

    print(f"\n  【春节历史表现统计】({len(results)}年数据)")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  上涨次数: {up_count} ({up_count / len(results) * 100:.1f}%)")
    print(f"  下跌次数: {down_count} ({down_count / len(results) * 100:.1f}%)")
    print(f"  持平次数: {flat_count} ({flat_count / len(results) * 100:.1f}%)")
    print(f"  胜率(节后5天>0): {win_rate:.1f}%")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  平均跨节涨幅: {avg_cross:+.3f}%")
    print(f"  平均节后3天: {avg_post_3d:+.3f}%")
    print(f"  平均节后5天: {avg_post_5d:+.3f}%")
    print(f"  平均节后10天: {avg_post_10d:+.3f}%")
    print(f"  平均最大涨幅: {avg_max_gain:+.3f}%")
    print(f"  平均最大回撤: {avg_max_drawdown:+.3f}%")
    print(f"  平均波动性: {avg_volatility:.3f}%")

    # 显示每年详细数据
    print(f"\n  【年度详细数据】")
    print(
        f"  {'年份':<6} {'节前%':>8} {'跨节%':>8} {'节后3d':>9} {'节后5d':>9} {'节后10d':>10} {'最大涨':>8} {'最大跌':>8}"
    )
    print(f"  {'-' * 80}")
    for _, row in results_df.iterrows():
        print(
            f"  {int(row['year']):<6} {row['pre_change']:>+8.2f} {row['cross_change']:>+8.2f} "
            f"{row['post_3d']:>+9.2f} {row['post_5d']:>+9.2f} {row['post_10d']:>+10.2f} "
            f"{row['max_gain']:>+8.2f} {row['max_drawdown']:>+8.2f}"
        )

    return {
        "code": code,
        "name": config["name"],
        "market": config["market"],
        "years_count": len(results),
        "up_count": int(up_count),
        "down_count": int(down_count),
        "flat_count": int(flat_count),
        "win_rate": win_rate,
        "avg_cross": avg_cross,
        "avg_post_3d": avg_post_3d,
        "avg_post_5d": avg_post_5d,
        "avg_post_10d": avg_post_10d,
        "avg_max_gain": avg_max_gain,
        "avg_max_drawdown": avg_max_drawdown,
        "avg_volatility": avg_volatility,
        "yearly_data": results,
    }


def generate_summary_and_plan(backtest_results):
    """生成汇总和交易计划"""
    print(f"\n{'=' * 80}")
    print(f"蔡森技术分析 - 2026年春节全球股指交易计划")
    print(f"{'=' * 80}")

    # 按胜率和平均收益排序
    sorted_results = sorted(
        [r for r in backtest_results.values() if r],
        key=lambda x: (x["win_rate"], x["avg_post_5d"]),
        reverse=True,
    )

    print(f"\n【春节历史表现排名】")
    print(
        f"{'指数':<15} {'胜率':>8} {'平均跨节':>10} {'节后5d':>10} {'最大涨':>10} {'最大跌':>10} {'推荐':>8}"
    )
    print(f"{'-' * 80}")

    for r in sorted_results:
        if r["avg_post_5d"] > 0.3 and r["win_rate"] >= 55:
            recommendation = "做多"
        elif r["avg_post_5d"] < -0.3 and r["win_rate"] <= 45:
            recommendation = "做空"
        else:
            recommendation = "观望"

        print(
            f"{r['name']:<15} {r['win_rate']:>7.1f}% {r['avg_cross']:>+9.2f}% "
            f"{r['avg_post_5d']:>+9.2f}% {r['avg_max_gain']:>+9.2f}% {r['avg_max_drawdown']:>+9.2f}% {recommendation:>8}"
        )

    # 生成交易计划
    print(f"\n{'=' * 80}")
    print(f"【2026年春节交易计划】")
    print(f"春节日期: 2026年2月17日 (农历正月初一)")
    print(f"中国市场假期: 2月17日 - 2月21日")
    print(f"建议入场: 节前1-2个交易日 (2月13-14日)")
    print(f"建议出场: 节后3-5个交易日 (2月25-27日)")
    print(f"{'=' * 80}")

    plan = {
        "cny_date": "2026-02-17",
        "market_holiday": "2026-02-17 to 2026-02-21",
        "entry_date": "2026-02-13 to 2026-02-14",
        "exit_date": "2026-02-25 to 2026-02-27",
        "indices": {},
    }

    for r in sorted_results:
        pair_plan = {
            "name": r["name"],
            "market": r["market"],
            "years_analyzed": r["years_count"],
            "win_rate": r["win_rate"],
            "avg_post_5d": r["avg_post_5d"],
            "avg_max_gain": r["avg_max_gain"],
            "avg_max_drawdown": r["avg_max_drawdown"],
        }

        # 根据历史表现确定交易建议
        if r["win_rate"] >= 60 and r["avg_post_5d"] > 0.5:
            pair_plan["direction"] = "强烈做多"
            pair_plan["confidence"] = r["win_rate"]
            pair_plan["entry"] = "节前1-2天"
            pair_plan["exit"] = "节后5天"
            pair_plan["stop_loss"] = "3-5%"
            pair_plan["position"] = "标准仓 (1.5-2%)"
            pair_plan["action"] = "STRONG_BUY"

        elif r["win_rate"] >= 55 and r["avg_post_5d"] > 0.3:
            pair_plan["direction"] = "看多"
            pair_plan["confidence"] = r["win_rate"]
            pair_plan["entry"] = "节前1天"
            pair_plan["exit"] = "节后3-5天"
            pair_plan["stop_loss"] = "2-3%"
            pair_plan["position"] = "轻仓 (0.5-1%)"
            pair_plan["action"] = "BUY"

        elif r["win_rate"] <= 40 and r["avg_post_5d"] < -0.5:
            pair_plan["direction"] = "强烈做空"
            pair_plan["confidence"] = 100 - r["win_rate"]
            pair_plan["entry"] = "节前1-2天"
            pair_plan["exit"] = "节后5天"
            pair_plan["stop_loss"] = "3-5%"
            pair_plan["position"] = "标准仓 (1.5-2%)"
            pair_plan["action"] = "STRONG_SELL"

        elif r["win_rate"] <= 45 and r["avg_post_5d"] < -0.3:
            pair_plan["direction"] = "看空"
            pair_plan["confidence"] = 100 - r["win_rate"]
            pair_plan["entry"] = "节前1天"
            pair_plan["exit"] = "节后3-5天"
            pair_plan["stop_loss"] = "2-3%"
            pair_plan["position"] = "轻仓 (0.5-1%)"
            pair_plan["action"] = "SELL"

        else:
            pair_plan["direction"] = "震荡/观望"
            pair_plan["confidence"] = 50
            pair_plan["entry"] = "-"
            pair_plan["exit"] = "-"
            pair_plan["stop_loss"] = "-"
            pair_plan["position"] = "观望"
            pair_plan["action"] = "HOLD"

        plan["indices"][r["code"]] = pair_plan

        print(f"\n【{r['name']} ({r['code']})】 - {r['market']}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  历史数据: {r['years_count']}年")
        print(f"  胜率: {r['win_rate']:.1f}%")
        print(f"  节后5天平均: {r['avg_post_5d']:+.2f}%")
        print(f"  平均最大涨幅: {r['avg_max_gain']:+.2f}%")
        print(f"  平均最大回撤: {r['avg_max_drawdown']:+.2f}%")
        print(
            f"  ━━━ 推荐方向: {pair_plan['direction']} (置信度: {pair_plan['confidence']:.0f}%) ━━━"
        )
        print(f"  入场时机: {pair_plan['entry']}")
        print(f"  出场时机: {pair_plan['exit']}")
        print(f"  止损建议: {pair_plan['stop_loss']}")
        print(f"  仓位建议: {pair_plan['position']}")

    # 综合建议
    print(f"\n{'=' * 80}")
    print(f"【综合交易建议】")
    print(f"{'=' * 80}")

    buy_signals = [
        r for r in sorted_results if r["win_rate"] >= 55 and r["avg_post_5d"] > 0.3
    ]
    sell_signals = [
        r for r in sorted_results if r["win_rate"] <= 45 and r["avg_post_5d"] < -0.3
    ]

    if buy_signals:
        print(f"\n做多机会:")
        for r in buy_signals:
            print(
                f"  • {r['name']}: 胜率{r['win_rate']:.0f}%, 平均涨幅{r['avg_post_5d']:+.2f}%"
            )

    if sell_signals:
        print(f"\n做空机会:")
        for r in sell_signals:
            print(
                f"  • {r['name']}: 胜率{100 - r['win_rate']:.0f}%, 平均跌幅{r['avg_post_5d']:+.2f}%"
            )

    print(f"\n风险控制:")
    print(f"  • 总仓位不超过5%")
    print(f"  • 单品种止损2-5%")
    print(f"  • 关注中国市场复牌后的波动")

    return plan, sorted_results


def main():
    print(f"{'=' * 80}")
    print(f"蔡森技术分析 - 中国春节假期全球股指表现回测")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"指数: SPX, ND100, DJI, JP225, DAX, HSI, IWM")
    print(f"回测周期: 过去20年")
    print(f"{'=' * 80}")

    backtest_results = {}

    for code, config in INDICES.items():
        result = backtest_index(code, config)
        if result:
            backtest_results[code] = result
        time.sleep(5)  # 避免速率限制

    if backtest_results:
        plan, summary = generate_summary_and_plan(backtest_results)

        # 保存结果
        output = {
            "backtest_results": {
                k: {key: v for key, v in val.items() if key != "yearly_data"}
                for k, val in backtest_results.items()
            },
            "yearly_details": {
                k: v["yearly_data"] for k, v in backtest_results.items()
            },
            "trading_plan": plan,
        }

        with open(
            "/root/ideas/caishen/cny_indices_backtest.json", "w", encoding="utf-8"
        ) as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n\n结果已保存至: /root/ideas/caishen/cny_indices_backtest.json")

    return backtest_results


if __name__ == "__main__":
    main()
