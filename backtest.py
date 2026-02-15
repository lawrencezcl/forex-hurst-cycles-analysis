#!/usr/bin/env python3
"""
蔡森技术分析回测系统 - Backtesting System
回测蔡森技术分析方法的历史表现
"""

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator

TWELVE_DATA_API_KEY = "f5491ce160e64101a960e19eb8363f38"


def fetch_data(symbol: str, outputsize: int = 365) -> pd.DataFrame:
    """获取历史数据"""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": f"{symbol}/USD",
        "interval": "1day",
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    if "values" not in data:
        return None

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )

    return df


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """计算蔡森技术分析信号"""
    df = df.copy()

    # 均线系统
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    # MACD
    macd = MACD(close=df["Close"], window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # RSI
    rsi = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi.rsi()

    # KDJ
    stoch = StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=9, smooth_window=3
    )
    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    # 生成信号
    df["MA_Signal"] = 0
    df.loc[
        (df["Close"] > df["MA5"])
        & (df["MA5"] > df["MA10"])
        & (df["MA10"] > df["MA20"]),
        "MA_Signal",
    ] = 1
    df.loc[
        (df["Close"] < df["MA5"])
        & (df["MA5"] < df["MA10"])
        & (df["MA10"] < df["MA20"]),
        "MA_Signal",
    ] = -1

    df["MACD_Signal_Flag"] = 0
    df.loc[
        (df["MACD"] > df["MACD_Signal"])
        & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1)),
        "MACD_Signal_Flag",
    ] = 1
    df.loc[
        (df["MACD"] < df["MACD_Signal"])
        & (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1)),
        "MACD_Signal_Flag",
    ] = -1

    df["RSI_Signal"] = 0
    df.loc[df["RSI"] < 30, "RSI_Signal"] = 1
    df.loc[df["RSI"] > 70, "RSI_Signal"] = -1

    df["KDJ_Signal"] = 0
    df.loc[
        (df["K"] > df["D"]) & (df["K"].shift(1) <= df["D"].shift(1)), "KDJ_Signal"
    ] = 1
    df.loc[
        (df["K"] < df["D"]) & (df["K"].shift(1) >= df["D"].shift(1)), "KDJ_Signal"
    ] = -1

    # 综合信号
    df["Total_Signal"] = (
        df["MA_Signal"] * 2
        + df["MACD_Signal_Flag"] * 2
        + df["RSI_Signal"] * 1
        + df["KDJ_Signal"] * 2
    )

    df["Action"] = "HOLD"
    df.loc[df["Total_Signal"] >= 3, "Action"] = "BUY"
    df.loc[df["Total_Signal"] <= -3, "Action"] = "SELL"

    return df


def backtest(df: pd.DataFrame, initial_capital: float = 10000) -> dict:
    """回测策略"""
    df = df.copy()
    df = df.dropna(subset=["Total_Signal"])

    capital = initial_capital
    position = 0
    trades = []
    equity_curve = []

    for i in range(1, len(df)):
        date = df.index[i]
        price = df.iloc[i]["Close"]
        signal = df.iloc[i]["Total_Signal"]
        prev_signal = df.iloc[i - 1]["Total_Signal"]

        # 买入信号
        if signal >= 3 and prev_signal < 3 and position == 0:
            position = capital / price
            entry_price = price
            entry_date = date
            capital = 0

        # 卖出信号
        elif signal <= -3 and prev_signal > -3 and position > 0:
            capital = position * price
            pnl = (price - entry_price) / entry_price * 100
            trades.append(
                {
                    "entry_date": entry_date.strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl,
                }
            )
            position = 0

        # 计算当前权益
        if position > 0:
            equity = position * price
        else:
            equity = capital
        equity_curve.append({"date": date, "equity": equity})

    # 如果还有持仓，按最后价格平仓
    if position > 0:
        capital = position * df.iloc[-1]["Close"]
        position = 0

    final_capital = capital if capital > 0 else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital * 100

    # 计算胜率
    winning_trades = [t for t in trades if t["pnl_pct"] > 0]
    losing_trades = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

    # 计算最大回撤
    equity_df = pd.DataFrame(equity_curve)
    if len(equity_df) > 0:
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (
            (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
        )
        max_drawdown = equity_df["drawdown"].min()
    else:
        max_drawdown = 0

    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_return_pct": total_return,
        "num_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate_pct": win_rate,
        "max_drawdown_pct": max_drawdown,
        "trades": trades,
    }


def run_backtest():
    """运行回测"""
    print("=" * 60)
    print("蔡森技术分析回测报告")
    print("=" * 60)
    print(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据周期: 过去365天")
    print("=" * 60)

    results = {}

    for symbol in ["BTC", "ETH"]:
        print(f"\n正在回测 {symbol}...")

        df = fetch_data(symbol, outputsize=365)
        if df is None:
            print(f"  无法获取 {symbol} 数据")
            continue

        df = calculate_signals(df)
        result = backtest(df, initial_capital=10000)
        results[symbol] = result

        print(f"\n{'=' * 40}")
        print(f"  {symbol} 回测结果")
        print(f"{'=' * 40}")
        print(f"  初始资金: ${result['initial_capital']:,.2f}")
        print(f"  最终资金: ${result['final_capital']:,.2f}")
        print(f"  总收益率: {result['total_return_pct']:.2f}%")
        print(f"  交易次数: {result['num_trades']}")
        print(f"  盈利次数: {result['winning_trades']}")
        print(f"  亏损次数: {result['losing_trades']}")
        print(f"  胜率: {result['win_rate_pct']:.1f}%")
        print(f"  最大回撤: {result['max_drawdown_pct']:.2f}%")

        if result["trades"]:
            print(f"\n  交易记录 (最近5笔):")
            for trade in result["trades"][-5:]:
                pnl_str = (
                    f"+{trade['pnl_pct']:.2f}%"
                    if trade["pnl_pct"] > 0
                    else f"{trade['pnl_pct']:.2f}%"
                )
                print(f"    {trade['entry_date']} -> {trade['exit_date']}: {pnl_str}")

    # 保存结果
    output_file = "/root/ideas/caishen/backtest_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n回测结果已保存至: {output_file}")

    return results


if __name__ == "__main__":
    run_backtest()
