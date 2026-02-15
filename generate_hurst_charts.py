#!/usr/bin/env python3
"""
生成蔡森周期分析的可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from hurst_cycles_analysis import HurstCyclesAnalyzer, MultiTimeFrameAnalyzer
from generate_forex_data import generate_forex_data
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def generate_sample_with_cycles():
    """生成带有明显周期性的样本数据"""
    # 生成90天的日数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # 使用generate_forex_data生成基础数据
    df = generate_forex_data('EURUSD', start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d'), seed=42)

    # 添加额外的周期性
    n = len(df)
    t = np.arange(n)

    # 添加多个周期
    cycles = np.zeros(n)
    for period, amplitude in [(10, 0.003), (20, 0.004), (40, 0.005)]:
        cycles += amplitude * np.sin(2 * np.pi * t / period)

    # 调整价格
    base_price = df['Close'].iloc[0]
    df['Close'] = df['Close'] * (1 + cycles)
    df['High'] = df['High'] * (1 + cycles)
    df['Low'] = df['Low'] * (1 + cycles)
    df['Open'] = df['Open'] * (1 + cycles)

    return df

def create_comprehensive_dashboard():
    """创建综合蔡森周期分析仪表板"""
    # 生成数据
    data = generate_sample_with_cycles()

    # 创建分析器
    analyzer = HurstCyclesAnalyzer(data)

    # 获取分析结果
    dominant_cycles = analyzer.identify_dominant_cycles()
    turning_points = analyzer.identify_cycle_turning_points()
    detrended = analyzer.detrend_price()
    periods, psd = analyzer.spectral_analysis()
    projection = analyzer.project_cycles_forward(n_periods=30)

    # 创建图表
    fig = plt.figure(figsize=(20, 16))

    # 1. 原始价格 vs 去趋势价格
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(analyzer.dates, analyzer.close, label='Original Price', color='blue', alpha=0.7, linewidth=1)
    ax1.plot(analyzer.dates, detrended + np.mean(analyzer.close), label='Detrended + Offset',
            color='orange', alpha=0.7, linewidth=1)
    ax1.set_title('Price vs Detrended Price (Hurst Method)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    # 2. 功率谱密度 (频谱分析)
    ax2 = plt.subplot(4, 2, 2)
    valid_mask = (periods >= 5) & (periods <= len(analyzer.close) / 2)
    ax2.plot(periods[valid_mask], psd[valid_mask], color='purple', linewidth=1.5)
    ax2.set_xlabel('Period (days)', fontsize=10)
    ax2.set_ylabel('Power Spectral Density', fontsize=10)
    ax2.set_title('Spectral Analysis - Dominant Cycles Detection', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 60)
    ax2.grid(True, alpha=0.3)

    # 标记主导周期
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, cycle in enumerate(dominant_cycles[:5]):
        ax2.axvline(x=cycle['period'], color=colors[i], linestyle='--', alpha=0.6, linewidth=1.5)
        ax2.text(cycle['period'], ax2.get_ylim()[1] * 0.9 - i * ax2.get_ylim()[1] * 0.15,
                f"C{i+1}: {cycle['period']:.1f}d\n({cycle['normalized_power']*100:.0f}%)",
                fontsize=8, color=colors[i], fontweight='bold')

    # 3. 周期转折点
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(analyzer.dates, analyzer.close, label='Price', color='navy', alpha=0.7, linewidth=1)

    for tp in turning_points:
        if tp['type'] == 'VALLEY':
            color = 'green'
            marker = '^'
        else:
            color = 'red'
            marker = 'v'
        ax3.scatter(tp['date'], tp['value'], color=color, marker=marker, s=80,
                   alpha=0.7, edgecolors='black', linewidth=0.5)

    ax3.set_title('Cycle Turning Points (Peaks & Valleys)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    # 4. 周期成分分解
    ax4 = plt.subplot(4, 2, 4)
    if dominant_cycles:
        # 显示前3个主导周期成分
        for i, cycle in enumerate(dominant_cycles[:3]):
            period = int(round(cycle['period']))
            component = analyzer.extract_cycle_component(period)
            if np.any(component != 0):
                ax4.plot(analyzer.dates, component, label=f'Cycle {i+1} ({period}d)',
                        linewidth=1, alpha=0.7)

    ax4.set_title('Individual Cycle Components', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Amplitude')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    # 5. 周期投影
    ax5 = plt.subplot(4, 2, 5)
    historical_dates = analyzer.dates
    historical_prices = analyzer.close

    # 生成未来日期
    last_date = analyzer.dates[-1]
    future_dates = pd.date_range(start=last_date, periods=len(projection)+1, freq='D')[1:]

    ax5.plot(historical_dates[-60:], historical_prices[-60:], label='Historical',
            color='blue', linewidth=1.5, alpha=0.7)
    ax5.plot(future_dates, projection, label='Projected (Cycles)',
            color='orange', linestyle='--', linewidth=2, alpha=0.8)
    ax5.axvline(x=last_date, color='red', linestyle='-', linewidth=2, alpha=0.5, label='Current')

    # 标记预测的转折点
    if turning_points:
        last_tp = turning_points[-1]
        ax5.scatter(last_tp['date'], last_tp['value'], color='purple', marker='D',
                   s=100, zorder=5, label='Last Turning Point')

    ax5.set_title('Cycle Projection (Next 30 Days)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Price')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    # 6. 周期相位分析
    ax6 = plt.subplot(4, 2, 6)
    # 计算周期相位
    if dominant_cycles:
        main_period = int(round(dominant_cycles[0]['period']))
        component = analyzer.extract_cycle_component(main_period)

        if np.any(component != 0):
            # 标准化到0-100%
            normalized = ((component - component.min()) / (component.max() - component.min()) * 100)

            # 标记区域
            ax6.fill_between(analyzer.dates, 0, 100, where=(normalized > 70),
                            color='red', alpha=0.2, label='Overbought Zone')
            ax6.fill_between(analyzer.dates, 0, 100, where=(normalized < 30),
                            color='green', alpha=0.2, label='Oversold Zone')

            ax6.plot(analyzer.dates, normalized, color='purple', linewidth=1.5, label='Cycle Phase')

            # 标记当前水平
            current_phase = normalized[-1]
            ax6.axhline(y=current_phase, color='blue', linestyle='--', alpha=0.7,
                       linewidth=1.5, label=f'Current: {current_phase:.1f}')

    ax6.set_title(f'Cycle Phase Analysis (Main Period: {main_period if dominant_cycles else "N/A"} days)',
                  fontsize=12, fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Phase (0-100%)')
    ax6.set_ylim(0, 100)
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    # 7. 多时间框架对比
    ax7 = plt.subplot(4, 2, 7)
    # 模拟不同时间框架的信号
    timeframes = ['1m', '4m', '15m', '30m', '1H', '4H']
    signals = np.random.choice(['BULL', 'BEAR'], size=len(timeframes))

    colors_bull = []
    for sig in signals:
        colors_bull.append('green' if sig == 'BULL' else 'red')

    bars = ax7.bar(range(len(timeframes)), [1]*len(timeframes), color=colors_bull, alpha=0.7, edgecolor='black')

    # 添加信号标签
    for i, (bar, sig) in enumerate(zip(bars, signals)):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height/2,
                sig, ha='center', va='center', fontweight='bold', fontsize=10)

    ax7.set_title('Multi-Timeframe Signal Confluence', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Timeframe')
    ax7.set_ylabel('Signal')
    ax7.set_xticks(range(len(timeframes)))
    ax7.set_xticklabels(timeframes)
    ax7.set_ylim(0, 1.2)
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. 交易计划摘要
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')

    # 创建交易计划表格
    summary_text = """
    HURST CYCLES TRADING PLAN SUMMARY

    ANALYSIS RESULTS:
    • Dominant Cycles: """ + str(len(dominant_cycles)) + """ identified
    • Last TP: """ + (turning_points[-1]['type'] if turning_points else 'N/A') + """
    • Current Phase: """ + analyzer._determine_current_phase() + """

    TOP 3 DOMINANT CYCLES:
    """

    for i, cycle in enumerate(dominant_cycles[:3], 1):
        summary_text += f"    {i}. Period: {cycle['period']:.1f} days (Power: {cycle['normalized_power']*100:.0f}%)\n"

    summary_text += """

    RECOMMENDATION:
    • Action: """ + ("BUY" if 'RISELING' in analyzer._determine_current_phase() else "SELL") + """
    • Confidence: """ + str(int(dominant_cycles[0]['normalized_power']*100)) + """%
    • Risk Level: Moderate

    NEXT KEY DATE:
    """ + (f"• {turning_points[-1]['date'].strftime('%Y-%m-%d')}" if turning_points else "• Pending analysis")

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 总标题
    fig.suptitle('HURST CYCLES ANALYSIS - COMPREHENSIVE DASHBOARD',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # 保存图表
    filename = f'/root/forex/hurst_cycles_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ 综合仪表板已保存至: {filename}")

    plt.close()

    return filename


def create_pair_comparison_chart():
    """创建货币对对比图表"""
    pairs_data = {
        'EURUSD': generate_sample_with_cycles(),
        'GBPUSD': generate_sample_with_cycles(),
        'USDJPY': generate_sample_with_cycles(),
        'AUDUSD': generate_sample_with_cycles(),
        'USDCAD': generate_sample_with_cycles(),
        'USDCNY': generate_sample_with_cycles(),
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (pair, data) in enumerate(pairs_data.items()):
        ax = axes[idx]
        analyzer = HurstCyclesAnalyzer(data)
        dominant_cycles = analyzer.identify_dominant_cycles()

        # 绘制价格
        ax.plot(analyzer.dates, analyzer.close, color='navy', linewidth=1, alpha=0.7)

        # 绘制去趋势价格
        detrended = analyzer.detrend_price()
        ax.plot(analyzer.dates, detrended + np.mean(analyzer.close),
                color='orange', linewidth=1, alpha=0.5, linestyle='--')

        # 标记转折点
        turning_points = analyzer.identify_cycle_turning_points()
        for tp in turning_points[-5:]:  # 只显示最近5个
            color = 'green' if tp['type'] == 'VALLEY' else 'red'
            marker = '^' if tp['type'] == 'VALLEY' else 'v'
            ax.scatter(tp['date'], tp['value'], color=color, marker=marker,
                      s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

        # 主导周期信息
        if dominant_cycles:
            main_period = dominant_cycles[0]['period']
            power = dominant_cycles[0]['normalized_power'] * 100
            ax.set_title(f'{pair}\nMain Cycle: {main_period:.1f}d (Power: {power:.0f}%)',
                        fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'{pair}\nNo dominant cycles', fontsize=10, fontweight='bold')

        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    fig.suptitle('Multi-Currency Pair Hurst Cycles Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'/root/forex/hurst_cycles_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ 货币对对比图已保存至: {filename}")

    plt.close()

    return filename


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" "*20 + "生成蔡森周期分析可视化图表")
    print("="*80 + "\n")

    print("1. 生成综合分析仪表板...")
    dashboard_file = create_comprehensive_dashboard()

    print("\n2. 生成货币对对比图...")
    comparison_file = create_pair_comparison_chart()

    print("\n" + "="*80)
    print("✓ 所有图表生成完成!")
    print("="*80)
    print(f"\n综合仪表板: {dashboard_file}")
    print(f"货币对对比: {comparison_file}\n")


if __name__ == "__main__":
    main()
