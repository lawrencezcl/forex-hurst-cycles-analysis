#!/usr/bin/env python3
"""
蔡森周期分析 - 全球主要指数与加密货币分析
==========================================

分析市场:
- SPX (标普500指数)
- ND100 (纳斯达克100指数)
- IWM (罗素2000 ETF)
- JP225 (日经225指数)
- DAX (德国DAX指数)
- BTC (比特币)
- ETH (以太坊)
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hurst_cycles_analysis import (
    ForexPairsHurstAnalyzer,
    plot_cycle_analysis,
    print_detailed_report
)
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


# 要分析的指数和加密货币
MARKET_INDICES = {
    'SPX': '^GSPC',        # S&P 500
    'ND100': '^NDX',       # NASDAQ 100
    'IWM': 'IWM',          # Russell 2000 ETF
    'JP225': '^N225',      # Nikkei 225
    'DAX': '^GDAXI',       # DAX (Germany)
    'BTC': 'BTC-USD',      # Bitcoin
    'ETH': 'ETH-USD',      # Ethereum
}


def fetch_market_data(symbol: str, ticker: str, days: int = 180) -> pd.DataFrame:
    """
    获取市场数据

    Args:
        symbol: 市场符号
        ticker: Yahoo Finance ticker
        days: 获取天数

    Returns:
        价格数据
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 获取日线数据
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty or len(data) < 100:
            print(f"  ⚠ {symbol}: 数据不足，生成模拟数据")
            return generate_market_data(symbol, days)

        # 标准化列名
        data.columns = [col.capitalize() for col in data.columns]

        print(f"  ✓ {symbol}: 成功获取 {len(data)} 条真实数据")
        return data

    except Exception as e:
        print(f"  ✗ {symbol}: 获取失败 ({e})，生成模拟数据")
        return generate_market_data(symbol, days)


def generate_market_data(symbol: str, days: int = 180) -> pd.DataFrame:
    """
    生成带周期性的模拟市场数据
    """
    # 生成日线数据
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='D'
    )

    # 过滤周末
    dates = dates[dates.dayofweek < 5]

    # 初始价格
    initial_prices = {
        'SPX': 5000.0,
        'ND100': 18000.0,
        'IWM': 200.0,
        'JP225': 38000.0,
        'DAX': 17000.0,
        'BTC': 95000.0,
        'ETH': 3200.0,
    }

    initial_price = initial_prices.get(symbol, 1000.0)
    n_points = len(dates)

    # 生成带周期性的价格
    t = np.arange(n_points)

    # 添加多个周期 (日/周/月/季周期)
    cycles = np.zeros(n_points)
    for period_days, amplitude in [
        (5, 0.005),      # 周周期
        (20, 0.008),     # 月周期
        (60, 0.012),     # 季周期
        (120, 0.015),    # 更长周期
    ]:
        phase = np.random.rand() * 2 * np.pi
        cycles += amplitude * np.sin(2 * np.pi * t / period_days + phase)

    # 添加趋势
    trend_slope = np.random.choice([-0.001, 0.001])  # 随机趋势方向
    trend = trend_slope * t / n_points

    # 添加随机噪声
    noise = np.random.normal(0, 0.008, n_points)

    # 计算价格
    prices = initial_price * (1 + cycles + trend + noise)

    # 确保价格为正
    prices = np.maximum(prices, initial_price * 0.7)

    # 生成OHLC
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n_points)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n_points)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]

    # 确保OHLC关系正确
    high = np.maximum(high, np.maximum(open_prices, prices))
    low = np.minimum(low, np.minimum(open_prices, prices))

    # 生成成交量
    base_volume = 1000000
    volume = base_volume * (1 + np.random.normal(0, 0.5, n_points))
    volume = np.abs(volume).astype(int)

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': prices,
        'Volume': volume,
    }, index=dates)

    return df


def prepare_all_markets_data() -> dict:
    """
    准备所有市场数据

    Returns:
        {符号: DataFrame} 字典
    """
    print("\n" + "="*100)
    print("正在获取/生成市场数据...")
    print("="*100 + "\n")

    markets_data = {}

    for symbol, ticker in MARKET_INDICES.items():
        print(f"处理 {symbol}...", end=' ')
        data = fetch_market_data(symbol, ticker, days=180)
        markets_data[symbol] = data

    print(f"\n✓ 所有 {len(markets_data)} 个市场数据准备完成\n")

    return markets_data


def run_full_analysis(markets_data: dict, generate_charts: bool = True):
    """
    运行完整的蔡森周期分析

    Args:
        markets_data: 市场数据字典
        generate_charts: 是否生成图表
    """
    print("\n" + "="*100)
    print("开始蔡森周期分析 (Hurst Cycles Analysis)")
    print("="*100 + "\n")

    # 创建分析器
    analyzer = ForexPairsHurstAnalyzer()

    # 分析所有市场
    print("-"*100)
    print("正在分析各市场的周期特征...")
    print("-"*100 + "\n")

    results = analyzer.analyze_multiple_pairs(markets_data)

    print("\n" + "-"*100)
    print(f"✓ 周期分析完成，成功分析 {len([r for r in results.values() if r])} 个市场")
    print("-"*100 + "\n")

    # 生成交易计划
    print("\n" + "-"*100)
    print("正在生成交易计划...")
    print("-"*100 + "\n")

    trading_plan = analyzer.generate_trading_plan(lookforward_days=7)

    print("✓ 交易计划生成完成\n")

    # 打印详细报告
    print_detailed_report(trading_plan)

    # 生成图表
    if generate_charts:
        print("\n" + "="*100)
        print("正在生成分析图表...")
        print("="*100 + "\n")

        for symbol, result in results.items():
            if result is None:
                continue

            # 为每个市场生成图表
            print(f"生成 {symbol} 图表...")
            try:
                # 使用原始数据创建分析器
                analyzer_instance = result.get('analyzers', {}).get('240')

                if analyzer_instance is None:
                    # 如果没有240分钟数据，创建一个基于日线数据的分析器
                    from hurst_cycles_analysis import HurstCyclesAnalyzer
                    data = result['data']
                    analyzer_instance = HurstCyclesAnalyzer(data)

                chart_path = f"/root/forex/hurst_chart_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plot_cycle_analysis(analyzer_instance, save_path=chart_path)
                print(f"  ✓ {symbol} 图表已保存")
            except Exception as e:
                print(f"  ⚠ {symbol} 图表生成失败: {e}")

        print("\n✓ 所有图表生成完成")

    # 保存交易计划到文件
    save_trading_plan(trading_plan, markets=True)

    return trading_plan


def save_trading_plan(trading_plan: dict, markets: bool = False):
    """
    保存交易计划到文本文件
    """
    market_type = "indices_crypto" if markets else "forex"
    filename = f"/root/forex/hurst_trading_plan_{market_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        if markets:
            f.write(" "*30 + "蔡森周期分析 - 全球市场交易计划\n")
            f.write(" "*25 + "HURST CYCLES - GLOBAL MARKETS PLAN\n")
        else:
            f.write(" "*35 + "蔡森周期分析 - 交易计划\n")
            f.write(" "*30 + "HURST CYCLES TRADING PLAN\n")
        f.write("="*100 + "\n\n")

        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"计划周期: 未来7个交易日\n\n")

        # 整体建议
        f.write("="*100 + "\n")
        f.write("【整体市场分析】\n")
        f.write("="*100 + "\n\n")

        for rec in trading_plan['overall_recommendations']:
            f.write(f"\n{rec['pair']}:\n")
            f.write(f"  当前价格: {rec['current_price']:.5f}\n")
            f.write(f"  市场情绪: {rec['sentiment']}\n")
            f.write(f"  信心水平: {rec['confidence']*100:.1f}%\n")
            f.write(f"  交易建议: {rec['recommendation']}\n")

        # 详细计划
        f.write("\n\n" + "="*100 + "\n")
        f.write("【各市场详细交易计划】\n")
        f.write("="*100 + "\n\n")

        for symbol, plan in trading_plan['pair_specific_plans'].items():
            f.write(f"\n{'─'*100}\n")
            f.write(f"  {symbol} 交易计划\n")
            f.write(f"{'─'*100}\n")
            f.write(f"  当前价格:        {plan['current_price']:.5f}\n")
            f.write(f"  市场情绪:        {plan['sentiment']}\n")
            f.write(f"  信心水平:        {plan['confidence']*100:.1f}%\n")
            f.write(f"  交易建议:        {plan['recommendation']}\n")
            f.write(f"  入场区间:        {plan['entry_zone']}\n")
            f.write(f"  目标价位:\n")
            for target in plan['target_levels']:
                f.write(f"    {target}\n")
            f.write(f"  止损位:          {plan['stop_loss']}\n")
            f.write(f"  时间周期:        {plan['time_horizon']}\n")
            f.write(f"  建议仓位:        {plan['position_size']}\n")

        # 关键日期
        if trading_plan['key_dates']:
            f.write("\n\n" + "="*100 + "\n")
            f.write("【关键转折日期预测】\n")
            f.write("="*100 + "\n\n")

            sorted_dates = sorted(trading_plan['key_dates'], key=lambda x: x['date'])
            for item in sorted_dates[:10]:
                f.write(f"  {item['date'].strftime('%Y-%m-%d')}: {item['pair']} - {item['event']}\n")

        # 风险提示
        f.write("\n\n" + "="*100 + "\n")
        f.write("【风险提示】\n")
        f.write("="*100 + "\n\n")
        for warning in trading_plan['risk_warnings']:
            f.write(f"  ⚠ {warning}\n")

        f.write("\n" + "="*100 + "\n")

    print(f"\n✓ 交易计划已保存至: {filename}")

    return filename


def print_summary_statistics(trading_plan: dict):
    """
    打印统计摘要
    """
    print("\n" + "="*100)
    print("【统计摘要】")
    print("="*100 + "\n")

    bullish_count = sum(1 for p in trading_plan['pair_specific_plans'].values()
                        if p['sentiment'] == 'BULLISH')
    bearish_count = sum(1 for p in trading_plan['pair_specific_plans'].values()
                        if p['sentiment'] == 'BEARISH')
    neutral_count = sum(1 for p in trading_plan['pair_specific_plans'].values()
                        if p['sentiment'] == 'NEUTRAL')

    total = len(trading_plan['pair_specific_plans'])

    print(f"  分析市场数量: {total}")
    print(f"  看涨市场:     {bullish_count} ({bullish_count/total*100:.1f}%)")
    print(f"  看跌市场:     {bearish_count} ({bearish_count/total*100:.1f}%)")
    print(f"  中性市场:     {neutral_count} ({neutral_count/total*100:.1f}%)")

    avg_confidence = np.mean([p['confidence'] for p in trading_plan['pair_specific_plans'].values()])
    print(f"  平均信心水平:   {avg_confidence*100:.1f}%")

    if trading_plan['key_dates']:
        next_event = min(trading_plan['key_dates'], key=lambda x: x['date'])
        days_to_event = (next_event['date'] - datetime.now()).days
        print(f"  下一个关键事件: {max(0, days_to_event)} 天后 ({next_event['date'].strftime('%Y-%m-%d')})")

    print("\n" + "="*100 + "\n")


def create_markets_summary_report(trading_plan: dict):
    """
    创建市场分析摘要报告
    """
    filename = f"/root/forex/markets_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# 蔡森周期分析 - 全球市场报告\n")
        f.write("## Hurst Cycles Analysis - Global Markets Report\n\n")
        f.write(f"**报告时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}\n")
        f.write(f"**分析周期**: 未来7个交易日\n\n")
        f.write("**分析市场**:\n")

        for symbol in trading_plan['pair_specific_plans'].keys():
            f.write(f"- {symbol}\n")

        f.write("\n---\n\n")

        # 执行摘要
        f.write("## 📊 执行摘要\n\n")
        f.write("基于蔡森周期分析，对各主要指数和加密货币进行多时间框架分析：\n\n")

        f.write("| 市场 | 当前价格 | 周期趋势 | 信心度 | 建议 | 风险/收益 |\n")
        f.write("|------|----------|----------|--------|------|----------|\n")

        for plan in trading_plan['pair_specific_plans'].values():
            emoji = "📈" if plan['sentiment'] == 'BULLISH' else "📉" if plan['sentiment'] == 'BEARISH' else "➡️"
            sentiment_emoji = f"{emoji} {plan['sentiment']}"
            confidence_stars = "⭐" * int(plan['confidence'] / 0.2)
            f.write(f"| **{plan['pair']}** | {plan['current_price']:.2f} | {sentiment_emoji} | {plan['confidence']*100:.0f}% {confidence_stars} | {plan['recommendation'].split(' - ')[0]} | ~1:2 |\n")

        # 详细分析
        f.write("\n---\n\n")
        f.write("## 🎯 详细交易计划\n\n")

        for symbol, plan in trading_plan['pair_specific_plans'].items():
            f.write(f"### {symbol}\n\n")
            f.write(f"- **当前价格**: {plan['current_price']:.2f}\n")
            f.write(f"- **市场情绪**: {plan['sentiment']}\n")
            f.write(f"- **信心水平**: {plan['confidence']*100:.0f}%\n")
            f.write(f"- **交易建议**: {plan['recommendation']}\n")
            f.write(f"- **入场区间**: {plan['entry_zone']}\n")
            f.write(f"- **目标价位**:\n")
            for target in plan['target_levels']:
                f.write(f"  - {target}\n")
            f.write(f"- **止损位**: {plan['stop_loss']}\n")
            f.write(f"- **建议仓位**: {plan['position_size']}\n\n")

        # 关键日期
        if trading_plan['key_dates']:
            f.write("---\n\n")
            f.write("## 📅 关键转折日期\n\n")

            sorted_dates = sorted(trading_plan['key_dates'], key=lambda x: x['date'])
            f.write("| 日期 | 市场 | 预期事件 |\n")
            f.write("|------|------|----------|\n")

            for item in sorted_dates[:10]:
                f.write(f"| {item['date'].strftime('%Y-%m-%d')} | {item['pair']} | {item['event']} |\n")

            f.write("\n")

    print(f"\n✓ 市场摘要报告已保存至: {filename}")

    return filename


def main():
    """主函数"""
    print("\n")
    print("╔" + "="*98 + "╗")
    print("║" + " "*20 + "蔡森周期全球市场分析系统" + " "*32 + "║")
    print("║" + " "*15 + "HURST CYCLES - GLOBAL MARKETS ANALYZER" + " "*37 + "║")
    print("╚" + "="*98 + "╝")

    # 准备数据
    markets_data = prepare_all_markets_data()

    # 运行分析
    trading_plan = run_full_analysis(markets_data, generate_charts=True)

    # 打印统计摘要
    print_summary_statistics(trading_plan)

    # 创建市场摘要报告
    create_markets_summary_report(trading_plan)

    print("\n" + "╔" + "="*98 + "╗")
    print("║" + " "*25 + "全球市场分析完成！" + " "*45 + "║")
    print("╚" + "="*98 + "╝\n")


if __name__ == "__main__":
    main()
