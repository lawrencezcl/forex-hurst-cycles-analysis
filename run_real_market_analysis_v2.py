#!/usr/bin/env python3
"""
获取真实市场数据的蔡森周期分析 V2
==================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hurst_cycles_analysis import ForexPairsHurstAnalyzer, plot_cycle_analysis
import yfinance as yf
import time
import warnings
warnings.filterwarnings('ignore')

# 市场配置
MARKETS = {
    'SPX': '^GSPC',
    'ND100': '^NDX',
    'IWM': 'IWM',
    'JP225': '^N225',
    'DAX': '^GDAXI',
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
}


def fetch_real_data(symbol, ticker, days=365):
    """获取真实市场数据"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        print(f"  正在获取 {symbol} 数据...")

        # 使用yfinance的Ticker方法
        ticker_obj = yf.Ticker(ticker)

        # 获取历史数据
        data = ticker_obj.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )

        if data is None or data.empty or len(data) < 50:
            print(f"  ✗ {symbol}: 数据不足")
            return None

        # 重置索引以处理MultiIndex
        data = data.reset_index()

        # 标准化列名 - 处理可能的MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [col.capitalize() for col in data.columns]

        # 设置日期为索引
        if 'Date' in data.columns:
            data = data.set_index('Date')

        # 确保有必需的列
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            print(f"  ✗ {symbol}: 缺少必需列")
            return None

        current_price = data['Close'].iloc[-1]
        print(f"  ✓ {symbol}: 成功获取 {len(data)} 天数据 (当前价格: {current_price:.2f})")

        return data

    except Exception as e:
        print(f"  ✗ {symbol}: 获取失败 - {str(e)[:100]}")
        return None


def run_analysis():
    """运行完整分析"""
    print("\n" + "="*100)
    print(" "*30 + "蔡森周期分析 - 真实市场数据")
    print(" "*25 + "HURST CYCLES - REAL DATA ANALYSIS")
    print("="*100 + "\n")

    # 获取所有市场数据
    print("正在获取最新市场数据...\n")
    markets_data = {}

    for idx, (symbol, ticker) in enumerate(MARKETS.items()):
        print(f"[{symbol}]", end=' ')
        data = fetch_real_data(symbol, ticker, days=365)

        if data is not None:
            markets_data[symbol] = data

        # 添加延迟以避免rate limiting
        if idx < len(MARKETS) - 1:
            time.sleep(1)

    if not markets_data:
        print("❌ 未能获取任何真实数据，请检查网络连接")
        return

    print(f"\n✓ 成功获取 {len(markets_data)}/{len(MARKETS)} 个市场的真实数据\n")

    # 创建分析器
    analyzer = ForexPairsHurstAnalyzer()

    # 分析所有市场
    print("-"*100)
    print("正在进行蔡森周期分析...")
    print("-"*100 + "\n")

    results = analyzer.analyze_multiple_pairs(markets_data)

    # 生成报告
    print("\n" + "="*100)
    print(" "*35 + "分析结果摘要")
    print("="*100 + "\n")

    print(f"{'市场':<10} {'当前价格':<15} {'周期趋势':<15} {'信心度':<10} {'建议'}")
    print("-"*100)

    for symbol, result in results.items():
        if result is None:
            continue

        symbol_plan = result.get('confluence', {})

        if symbol_plan:
            trend = symbol_plan.get('overall_sentiment', 'NEUTRAL')
            confidence = symbol_plan.get('confidence_level', 0) * 100
            current_price = markets_data[symbol]['Close'].iloc[-1]

            trend_emoji = "📈" if trend == 'BULLISH' else "📉" if trend == 'BEARISH' else "➡️"

            recommendation = "强烈买入" if trend == 'BULLISH' and confidence > 70 else \
                           "买入" if trend == 'BULLISH' else \
                           "强烈卖出" if trend == 'BEARISH' and confidence > 70 else \
                           "卖出" if trend == 'BEARISH' else "观望"

            print(f"{symbol:<10} {current_price:<15.2f} {trend_emoji} {trend:<12} {confidence:>6.1f}%   {recommendation}")

    print("\n" + "="*100 + "\n")

    # 生成图表
    print("正在生成分析图表...\n")

    for symbol, data in markets_data.items():
        try:
            from hurst_cycles_analysis import HurstCyclesAnalyzer
            cycle_analyzer = HurstCyclesAnalyzer(data)

            chart_path = f"/root/forex/REAL_{symbol}_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_cycle_analysis(cycle_analyzer, save_path=chart_path)
            print(f"  ✓ {symbol} 图表已生成: {chart_path}")
        except Exception as e:
            print(f"  ✗ {symbol} 图表生成失败: {e}")

    # 保存详细报告
    save_detailed_report(results, markets_data)

    print("\n✓ 分析完成！")


def save_detailed_report(results, markets_data):
    """保存详细报告"""
    filename = f"/root/forex/REAL_MARKETS_ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(" "*30 + "蔡森周期分析 - 真实市场数据报告\n")
        f.write("="*100 + "\n\n")
        f.write(f"报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据来源: Yahoo Finance (实时数据)\n")
        f.write(f"分析周期: 365天历史数据\n")
        f.write(f"分析方法: J.M. Hurst 周期理论\n\n")

        # 市场概况
        f.write("="*100 + "\n")
        f.write("【市场概况】\n")
        f.write("="*100 + "\n\n")

        for symbol, data in markets_data.items():
            close = data['Close']
            f.write(f"\n{symbol}:\n")
            f.write(f"  数据范围: {data.index[0].strftime('%Y-%m-%d')} 至 {data.index[-1].strftime('%Y-%m-%d')}\n")
            f.write(f"  数据点数: {len(data)} 天\n")
            f.write(f"  当前价格: {close.iloc[-1]:.2f}\n")
            f.write(f"  期间最高: {data['High'].max():.2f}\n")
            f.write(f"  期间最低: {data['Low'].min():.2f}\n")
            f.write(f"  期间涨跌: {((close.iloc[-1] / close.iloc[0] - 1) * 100):.2f}%\n")
            f.write(f"  年化波动率: {(close.pct_change().std() * np.sqrt(252) * 100):.2f}%\n")

        # 详细交易计划
        f.write("\n\n" + "="*100 + "\n")
        f.write("【蔡森周期分析结果】\n")
        f.write("="*100 + "\n\n")

        for symbol, result in results.items():
            if result is None:
                continue

            data = markets_data[symbol]
            close = data['Close'].iloc[-1]
            atr = calculate_atr(data)

            confluence = result.get('confluence', {})
            sentiment = confluence.get('overall_sentiment', 'NEUTRAL')
            confidence = confluence.get('confidence_level', 0)

            f.write(f"\n{'─'*100}\n")
            f.write(f"{symbol} - 蔡森周期交易计划\n")
            f.write(f"{'─'*100}\n")
            f.write(f"当前价格:        {close:.2f}\n")
            f.write(f"市场情绪:        {sentiment}\n")
            f.write(f"信心水平:        {confidence*100:.1f}%\n")

            if sentiment == 'BULLISH':
                f.write(f"交易建议:        {'强烈买入' if confidence > 0.7 else '买入'}\n")
                f.write(f"入场区间:        {close:.2f} - {close * 1.005:.2f}\n")
                f.write(f"目标价位:\n")
                f.write(f"  T1: {close * 1.01:.2f} (+1%)\n")
                f.write(f"  T2: {close * 1.02:.2f} (+2%)\n")
                f.write(f"  T3: {close * 1.03:.2f} (+3%)\n")
                f.write(f"止损位:          {close * 0.985:.2f} (-1.5%)\n")
            elif sentiment == 'BEARISH':
                f.write(f"交易建议:        {'强烈卖出' if confidence > 0.7 else '卖出'}\n")
                f.write(f"入场区间:        {close * 0.995:.2f} - {close:.2f}\n")
                f.write(f"目标价位:\n")
                f.write(f"  T1: {close * 0.99:.2f} (-1%)\n")
                f.write(f"  T2: {close * 0.98:.2f} (-2%)\n")
                f.write(f"  T3: {close * 0.97:.2f} (-3%)\n")
                f.write(f"止损位:          {close * 1.015:.2f} (+1.5%)\n")
            else:
                f.write(f"交易建议:        观望等待明确信号\n")

            f.write(f"时间周期:        Short-term (1-5 days)\n")
            f.write(f"建议仓位:        {'中等' if confidence > 0.6 else '小'}\n")

    print(f"\n✓ 详细报告已保存至: {filename}")


def calculate_atr(data, period=14):
    """计算ATR"""
    high = data['High']
    low = data['Low']
    close = data['Close']

    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]

    return atr


if __name__ == "__main__":
    run_analysis()
