#!/usr/bin/env python3
"""
蔡森周期多货币对多时间框架分析主程序
=====================================

分析货币对: EUR, GBP, JPY, AUD, CAD, CNH
时间框架: 1, 4, 15, 30, 60, 240 分钟
生成下周交易计划
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
from generate_forex_data import generate_forex_data
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


# 要分析的货币对 (以USD为基准)
FOREX_PAIRS = [
    'EURUSD',  # 欧元/美元
    'GBPUSD',  # 英镑/美元
    'USDJPY',  # 美元/日元
    'AUDUSD',  # 澳元/美元
    'USDCAD',  # 美元/加元
    'USDCNY',  # 美元/人民币
]


def fetch_real_forex_data(pair: str, days: int = 90) -> pd.DataFrame:
    """
    获取真实外汇数据

    Args:
        pair: 货币对
        days: 获取天数

    Returns:
        价格数据
    """
    try:
        # 转换货币对格式
        ticker_map = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'USDCNY': 'USDCNY=X',
        }

        ticker = ticker_map.get(pair, f"{pair}=X")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 获取数据 (使用小时数据以支持多时间框架分析)
        data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)

        if data.empty or len(data) < 100:
            print(f"  ⚠ {pair}: 真实数据不足，使用模拟数据")
            return generate_realistic_data(pair, days)

        # 标准化列名
        data.columns = [col.capitalize() for col in data.columns]

        print(f"  ✓ {pair}: 成功获取 {len(data)} 条真实数据")
        return data

    except Exception as e:
        print(f"  ✗ {pair}: 获取失败 ({e})，使用模拟数据")
        return generate_realistic_data(pair, days)


def generate_realistic_data(pair: str, days: int = 90) -> pd.DataFrame:
    """
    生成更真实的模拟数据 (包含周期性模式)
    """
    # 生成小时数据
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='1H'
    )

    # 过滤周末时间
    dates = dates[dates.dayofweek < 5]

    # 初始价格
    initial_prices = {
        'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDJPY': 149.50,
        'AUDUSD': 0.6550, 'USDCAD': 1.3450, 'USDCNY': 7.1950,
    }

    initial_price = initial_prices.get(pair, 1.0000)
    n_points = len(dates)

    # 生成带周期性的价格
    t = np.arange(n_points)

    # 多个周期叠加
    cycles = [
        (24, 0.002),      # 日周期 (24小时)
        (120, 0.003),     # 周周期 (5天)
        (600, 0.005),     # 月周期
        (100, 0.0015),    # 其他周期
    ]

    price_changes = np.zeros(n_points)
    for period, amplitude in cycles:
        phase = np.random.rand() * 2 * np.pi
        price_changes += amplitude * np.sin(2 * np.pi * t / period + phase)

    # 添加趋势
    trend = 0.0001 * t / n_points

    # 添加随机噪声
    noise = np.random.normal(0, 0.001, n_points)

    # 计算价格
    prices = initial_price * (1 + price_changes + trend + noise)

    # 确保价格为正
    prices = np.maximum(prices, initial_price * 0.8)

    # 生成OHLC
    high = prices * (1 + np.abs(np.random.normal(0, 0.0005, n_points)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.0005, n_points)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]

    # 确保OHLC关系正确
    high = np.maximum(high, np.maximum(open_prices, prices))
    low = np.minimum(low, np.minimum(open_prices, prices))

    # 生成成交量
    base_volume = 50000
    volume = base_volume * (1 + np.random.normal(0, 0.3, n_points))
    volume = np.abs(volume).astype(int)

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': prices,
        'Volume': volume,
    }, index=dates)

    return df


def prepare_all_pairs_data() -> dict:
    """
    准备所有货币对的数据

    Returns:
        {货币对: DataFrame} 字典
    """
    print("\n" + "="*100)
    print("正在获取/生成货币对数据...")
    print("="*100 + "\n")

    pairs_data = {}

    for pair in FOREX_PAIRS:
        print(f"处理 {pair}...", end=' ')
        data = fetch_real_forex_data(pair, days=90)
        pairs_data[pair] = data

    print(f"\n✓ 所有 {len(pairs_data)} 个货币对数据准备完成\n")

    return pairs_data


def run_full_analysis(pairs_data: dict, generate_charts: bool = True):
    """
    运行完整的蔡森周期分析

    Args:
        pairs_data: 货币对数据字典
        generate_charts: 是否生成图表
    """
    print("\n" + "="*100)
    print("开始蔡森周期分析 (Hurst Cycles Analysis)")
    print("="*100 + "\n")

    # 创建分析器
    analyzer = ForexPairsHurstAnalyzer()

    # 分析所有货币对
    print("-"*100)
    print("正在分析各货币对的周期特征...")
    print("-"*100 + "\n")

    results = analyzer.analyze_multiple_pairs(pairs_data)

    print("\n" + "-"*100)
    print(f"✓ 周期分析完成，成功分析 {len([r for r in results.values() if r])} 个货币对")
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

        for pair, result in results.items():
            if result is None:
                continue

            # 为每个货币对生成240分钟时间框架的图表
            if '240' in result.get('analyzers', {}):
                print(f"生成 {pair} 图表...")
                try:
                    tf_analyzer = result['analyzers']['240']
                    chart_path = f"/root/forex/hurst_cycle_chart_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    plot_cycle_analysis(tf_analyzer, save_path=chart_path)
                except Exception as e:
                    print(f"  ⚠ {pair} 图表生成失败: {e}")

        print("\n✓ 所有图表生成完成")

    # 保存交易计划到文件
    save_trading_plan(trading_plan)

    return trading_plan


def save_trading_plan(trading_plan: dict):
    """
    保存交易计划到文本文件
    """
    filename = f"/root/forex/hurst_trading_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
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
        f.write("【各货币对详细交易计划】\n")
        f.write("="*100 + "\n\n")

        for pair, plan in trading_plan['pair_specific_plans'].items():
            f.write(f"\n{'─'*100}\n")
            f.write(f"  {pair} 交易计划\n")
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

    print(f"  分析货币对数量: {total}")
    print(f"  看涨货币对:     {bullish_count} ({bullish_count/total*100:.1f}%)")
    print(f"  看跌货币对:     {bearish_count} ({bearish_count/total*100:.1f}%)")
    print(f"  中性货币对:     {neutral_count} ({neutral_count/total*100:.1f}%)")

    avg_confidence = np.mean([p['confidence'] for p in trading_plan['pair_specific_plans'].values()])
    print(f"  平均信心水平:   {avg_confidence*100:.1f}%")

    if trading_plan['key_dates']:
        next_event = min(trading_plan['key_dates'], key=lambda x: x['date'])
        days_to_event = (next_event['date'] - datetime.now()).days
        print(f"  下一个关键事件: {days_to_event} 天后 ({next_event['date'].strftime('%Y-%m-%d')})")

    print("\n" + "="*100 + "\n")


def main():
    """主函数"""
    print("\n")
    print("╔" + "="*98 + "╗")
    print("║" + " "*30 + "蔡森周期外汇分析系统" + " "*32 + "║")
    print("║" + " "*25 + "HURST CYCLES FOREX ANALYZER" + " "*37 + "║")
    print("╚" + "="*98 + "╝")

    # 准备数据
    pairs_data = prepare_all_pairs_data()

    # 运行分析
    trading_plan = run_full_analysis(pairs_data, generate_charts=True)

    # 打印统计摘要
    print_summary_statistics(trading_plan)

    print("\n" + "╔" + "="*98 + "╗")
    print("║" + " "*25 + "分析完成！祝交易顺利！" + " "*45 + "║")
    print("╚" + "="*98 + "╝\n")


if __name__ == "__main__":
    main()
