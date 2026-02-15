#!/usr/bin/env python3
"""
获取真实市场数据的蔡森周期分析
================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hurst_cycles_analysis import ForexPairsHurstAnalyzer, plot_cycle_analysis
import yfinance as yf
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

        # 获取日线数据
        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )

        if data.empty or len(data) < 50:
            print(f"  ✗ {symbol}: 数据不足 (仅{len(data) if not data.empty else 0}条)")
            return None

        # 标准化列名
        data.columns = data.columns.str.capitalize()

        print(f"  ✓ {symbol}: 成功获取 {len(data)} 天数据 (价格: {data['Close'].iloc[-1]:.2f})")
        return data

    except Exception as e:
        print(f"  ✗ {symbol}: 获取失败 - {e}")
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

    for symbol, ticker in MARKETS.items():
        print(f"[{symbol}]")
        data = fetch_real_data(symbol, ticker, days=365)
        if data is not None:
            markets_data[symbol] = data
        print()

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
    print(" "*35 + "分析结果")
    print("="*100 + "\n")

    print(f"{'市场':<10} {'当前价格':<15} {'周期趋势':<15} {'信心度':<10} {'建议'}")
    print("-"*100)

    for symbol, result in results.items():
        if result is None:
            continue

        plan = analyzer.generate_trading_plan(lookforward_days=7)
        symbol_plan = plan['pair_specific_plans'].get(symbol)

        if symbol_plan:
            trend_emoji = "📈" if symbol_plan['sentiment'] == 'BULLISH' else "📉" if symbol_plan['sentiment'] == 'BEARISH' else "➡️"
            print(f"{symbol:<10} {symbol_plan['current_price']:<15.2f} {trend_emoji} {symbol_plan['sentiment']:<12} {symbol_plan['confidence']*100:>6.1f}%   {symbol_plan['recommendation']}")

    print("\n" + "="*100 + "\n")

    # 生成图表
    print("正在生成分析图表...")

    for symbol, result in results.items():
        if result is None:
            continue

        try:
            # 创建周期分析器
            from hurst_cycles_analysis import HurstCyclesAnalyzer
            data = markets_data[symbol]
            cycle_analyzer = HurstCyclesAnalyzer(data)

            chart_path = f"/root/forex/real_{symbol}_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_cycle_analysis(cycle_analyzer, save_path=chart_path)
            print(f"  ✓ {symbol} 图表已生成")
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
        f.write(f"分析周期: 365天历史数据\n\n")

        # 市场概况
        f.write("="*100 + "\n")
        f.write("【市场概况】\n")
        f.write("="*100 + "\n\n")

        for symbol, data in markets_data.items():
            f.write(f"\n{symbol}:\n")
            f.write(f"  数据范围: {data.index[0].strftime('%Y-%m-%d')} 至 {data.index[-1].strftime('%Y-%m-%d')}\n")
            f.write(f"  数据点数: {len(data)} 天\n")
            f.write(f"  当前价格: {data['Close'].iloc[-1]:.2f}\n")
            f.write(f"  期间最高: {data['High'].max():.2f}\n")
            f.write(f"  期间最低: {data['Low'].min():.2f}\n")
            f.write(f"  期间涨跌: {((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100):.2f}%\n")

        # 详细交易计划
        f.write("\n\n" + "="*100 + "\n")
        f.write("【蔡森周期分析结果】\n")
        f.write("="*100 + "\n\n")

        for symbol, result in results.items():
            if result is None:
                continue

            analyzer = ForexPairsHurstAnalyzer()
            analyzer.analyze_pair(symbol, markets_data[symbol])
            plan = analyzer.generate_trading_plan(lookforward_days=7)
            symbol_plan = plan['pair_specific_plans'].get(symbol)

            if symbol_plan:
                f.write(f"\n{'─'*100}\n")
                f.write(f"{symbol} - 蔡森周期交易计划\n")
                f.write(f"{'─'*100}\n")
                f.write(f"当前价格:        {symbol_plan['current_price']:.2f}\n")
                f.write(f"市场情绪:        {symbol_plan['sentiment']}\n")
                f.write(f"信心水平:        {symbol_plan['confidence']*100:.1f}%\n")
                f.write(f"交易建议:        {symbol_plan['recommendation']}\n")
                f.write(f"入场区间:        {symbol_plan['entry_zone']}\n")
                f.write(f"\n目标价位:\n")
                for target in symbol_plan['target_levels']:
                    f.write(f"  {target}\n")
                f.write(f"\n止损位:          {symbol_plan['stop_loss']}\n")
                f.write(f"时间周期:        {symbol_plan['time_horizon']}\n")
                f.write(f"建议仓位:        {symbol_plan['position_size']}\n")

    print(f"\n✓ 详细报告已保存至: {filename}")


if __name__ == "__main__":
    run_analysis()
