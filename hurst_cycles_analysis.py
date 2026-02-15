"""
蔡森周期分析系统 (Hurst Cycles Analysis System)
==============================================

基于J.M. Hurst的市场周期理论，提供多时间框架的周期分析功能。
用于识别市场的周期性模式，预测价格转折点。

核心原理:
1. 价格由多个不同周期的波浪叠加而成
2. 使用频谱分析识别主导周期
3. 周期叠加理论预测未来价格走势
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HurstCyclesAnalyzer:
    """蔡森周期分析器"""

    def __init__(self, data: pd.DataFrame):
        """
        初始化周期分析器

        Args:
            data: 包含OHLC数据的DataFrame
        """
        self.data = data.copy()
        self.close = data['Close'].values
        self.dates = pd.to_datetime(data.index)
        self.cycles = {}
        self.cycle_projections = {}

    def detrend_price(self, method: str = 'ma') -> np.ndarray:
        """
        去除价格趋势，提取周期性成分

        Args:
            method: 去趋势方法 ('ma'移动平均, 'linear'线性回归)

        Returns:
            去趋势后的价格序列
        """
        if method == 'ma':
            # 使用移动平均去除趋势
            window = min(len(self.close) // 4, 50)
            trend = pd.Series(self.close).rolling(window=window, center=True).mean()
            trend = trend.fillna(method='bfill').fillna(method='ffill')
            detrended = self.close - trend.values

        elif method == 'linear':
            # 使用线性回归去除趋势
            x = np.arange(len(self.close))
            z = np.polyfit(x, self.close, 1)
            p = np.poly1d(z)
            trend = p(x)
            detrended = self.close - trend

        else:
            detrended = self.close

        return detrended

    def spectral_analysis(self, data: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        频谱分析识别主导周期

        Args:
            data: 输入数据，默认使用去趋势后的价格

        Returns:
            (周期长度列表, 功率谱密度列表)
        """
        if data is None:
            data = self.detrend_price(method='ma')

        # 移除NaN值
        data = data[~np.isnan(data)]

        # 计算功率谱密度
        freqs, psd = signal.welch(data, fs=1.0, nperseg=min(256, len(data)//2))

        # 转换频率为周期
        periods = 1.0 / freqs[1:]  # 跳过0频率
        psd = psd[1:]

        return periods, psd

    def identify_dominant_cycles(self, n_cycles: int = 5) -> List[Dict]:
        """
        识别主导周期

        Args:
            n_cycles: 返回的周期数量

        Returns:
            主导周期列表，每个包含周期长度和强度信息
        """
        periods, psd = self.spectral_analysis()

        # 过滤合理的周期范围 (5到数据长度的一半)
        valid_mask = (periods >= 5) & (periods <= len(self.close) / 2)
        valid_periods = periods[valid_mask]
        valid_psd = psd[valid_mask]

        # 找到峰值
        peaks, _ = signal.find_peaks(valid_psd, height=np.max(valid_psd) * 0.1)

        # 按功率排序
        sorted_indices = peaks[np.argsort(valid_psd[peaks])[::-1]]

        dominant_cycles = []
        for idx in sorted_indices[:n_cycles]:
            dominant_cycles.append({
                'period': valid_periods[idx],
                'power': valid_psd[idx],
                'normalized_power': valid_psd[idx] / np.max(valid_psd),
                'frequency': 1.0 / valid_periods[idx]
            })

        return dominant_cycles

    def extract_cycle_component(self, period: int) -> np.ndarray:
        """
        提取特定周期的成分

        Args:
            period: 周期长度

        Returns:
            该周期的价格成分
        """
        # 使用带通滤波器提取周期成分
        data = self.detrend_price(method='ma')

        # 设计带通滤波器
        nyquist = 0.5
        low = 1.0 / (period * 1.5)
        high = 1.0 / (period * 0.5)

        if low >= nyquist or high >= nyquist:
            # 如果频率超出范围，返回零
            return np.zeros_like(data)

        try:
            b, a = signal.butter(4, [low, high], btype='band', fs=1.0)
            component = signal.filtfilt(b, a, data)
            return component
        except:
            return np.zeros_like(data)

    def project_cycles_forward(self, n_periods: int = 20) -> np.ndarray:
        """
        基于识别的周期向前投影

        Args:
            n_periods: 向前投影的周期数

        Returns:
            投影的价格序列
        """
        dominant_cycles = self.identify_dominant_cycles()

        if not dominant_cycles:
            return np.full(n_periods, self.close[-1])

        # 提取所有主导周期成分
        projections = []
        for cycle_info in dominant_cycles[:3]:  # 使用前3个主导周期
            period = int(round(cycle_info['period']))
            component = self.extract_cycle_component(period)

            # 延伸该周期
            extended = np.zeros(n_periods)
            for i in range(n_periods):
                # 基于周期性模式预测
                position_in_cycle = i % period
                if len(component) >= period:
                    # 从历史数据中提取该位置的值
                    similar_positions = component[-period:]
                    extended[i] = similar_positions[int(position_in_cycle)]

            projections.append(extended)

        # 叠加所有周期预测
        if projections:
            total_projection = np.sum(projections, axis=0)

            # 添加趋势
            trend = np.mean(self.close[-20:]) if len(self.close) >= 20 else self.close[-1]
            trend_slope = (self.close[-1] - self.close[-min(20, len(self.close))]) / min(20, len(self.close))

            trend_component = trend + trend_slope * np.arange(n_periods)
            final_projection = trend_component + total_projection

            return final_projection
        else:
            return np.full(n_periods, self.close[-1])

    def identify_cycle_turning_points(self) -> List[Dict]:
        """
        识别周期转折点

        Returns:
            转折点列表，包含日期和类型(峰/谷)
        """
        detrended = self.detrend_price(method='ma')

        # 寻找峰值和谷值
        peaks, _ = signal.find_peaks(detrended, distance=5)
        valleys, _ = signal.find_peaks(-detrended, distance=5)

        turning_points = []

        for peak in peaks:
            if peak < len(self.dates):
                turning_points.append({
                    'date': self.dates[peak],
                    'type': 'PEAK',
                    'value': self.close[peak],
                    'strength': detrended[peak]
                })

        for valley in valleys:
            if valley < len(self.dates):
                turning_points.append({
                    'date': self.dates[valley],
                    'type': 'VALLEY',
                    'value': self.close[valley],
                    'strength': -detrended[valley]
                })

        # 按日期排序
        turning_points.sort(key=lambda x: x['date'])

        return turning_points

    def generate_cycle_summary(self) -> Dict:
        """
        生成周期分析摘要

        Returns:
            包含关键周期信息的字典
        """
        dominant_cycles = self.identify_dominant_cycles()
        turning_points = self.identify_cycle_turning_points()
        projection = self.project_cycles_forward(n_periods=5)

        summary = {
            'dominant_cycles': dominant_cycles,
            'last_turning_point': turning_points[-1] if turning_points else None,
            'next_turning_point_estimate': self._estimate_next_turning_point(),
            'current_phase': self._determine_current_phase(),
            'projection_5_periods': projection,
            'current_price': self.close[-1],
            'trend': 'UP' if self.close[-1] > self.close[-min(20, len(self.close))] else 'DOWN'
        }

        return summary

    def _estimate_next_turning_point(self) -> Dict:
        """估算下一个转折点"""
        dominant_cycles = self.identify_dominant_cycles()

        if not dominant_cycles:
            return None

        # 使用最短的主导周期
        shortest_period = int(dominant_cycles[-1]['period'])
        turning_points = self.identify_cycle_turning_points()

        if not turning_points:
            return None

        last_tp = turning_points[-1]
        estimated_date = last_tp['date'] + timedelta(days=shortest_period // 2)

        return {
            'estimated_date': estimated_date,
            'type': 'VALLEY' if last_tp['type'] == 'PEAK' else 'PEAK',
            'based_on_period': shortest_period
        }

    def _determine_current_phase(self) -> str:
        """确定当前所处的周期阶段"""
        detrended = self.detrend_price(method='ma')
        current_value = detrended[-1]

        # 判断是在上升还是下降阶段
        if len(detrended) >= 5:
            recent_change = detrended[-1] - detrended[-5]
            if current_value > 0:
                return 'RISELING_PEAK' if recent_change > 0 else 'FALLING_FROM_PEAK'
            else:
                return 'FALLING_TO_VALLEY' if recent_change < 0 else 'RISING_FROM_VALLEY'

        return 'NEUTRAL'


class MultiTimeFrameAnalyzer:
    """多时间框架分析器"""

    def __init__(self, data: pd.DataFrame):
        """
        初始化多时间框架分析器

        Args:
            data: 原始价格数据 (最小时间周期)
        """
        self.base_data = data.copy()
        self.timeframes = {}
        self.analyzers = {}

    def resample_data(self, timeframe_minutes: int) -> pd.DataFrame:
        """
        重采样数据到指定时间框架

        Args:
            timeframe_minutes: 时间框架(分钟)

        Returns:
            重采样后的数据
        """
        if timeframe_minutes < 60:
            rule = f'{timeframe_minutes}min'
        elif timeframe_minutes < 1440:
            rule = f'{timeframe_minutes // 60}H'
        else:
            rule = f'{timeframe_minutes // 1440}D'

        # 标准化列名
        data = self.base_data.copy()
        data.columns = [col.capitalize() for col in data.columns]

        # 确保所需列存在
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            # 尝试小写版本
            lowercase_map = {col.lower(): col for col in required_cols}
            rename_dict = {}
            for col in data.columns:
                lower = col.lower()
                if lower in lowercase_map:
                    rename_dict[col] = lowercase_map[lower]
            if rename_dict:
                data = data.rename(columns=rename_dict)

        resampled = data.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        return resampled

    def analyze_all_timeframes(self, timeframes: List[int] = None) -> Dict[int, Dict]:
        """
        分析所有时间框架

        Args:
            timeframes: 时间框架列表(分钟)，默认为[1, 4, 15, 30, 60, 240]

        Returns:
            各时间框架的分析结果
        """
        if timeframes is None:
            timeframes = [1, 4, 15, 30, 60, 240]

        results = {}

        for tf in timeframes:
            try:
                # 重采样数据
                tf_data = self.resample_data(tf)

                if len(tf_data) < 20:  # 确保有足够数据
                    continue

                # 创建周期分析器
                analyzer = HurstCyclesAnalyzer(tf_data)
                summary = analyzer.generate_cycle_summary()

                self.timeframes[tf] = tf_data
                self.analyzers[tf] = analyzer
                results[tf] = summary

            except Exception as e:
                print(f"Error analyzing timeframe {tf}min: {e}")
                continue

        return results

    def generate_confluence_analysis(self) -> Dict:
        """
        生成多时间框架的共振分析

        Returns:
            共振分析结果
        """
        results = self.analyze_all_timeframes()

        confluence = {
            'bullish_signals': [],
            'bearish_signals': [],
            'neutral_signals': [],
            'overall_sentiment': 'NEUTRAL',
            'confidence_level': 0.0
        }

        bullish_count = 0
        bearish_count = 0

        for tf, summary in results.items():
            current_phase = summary.get('current_phase', 'NEUTRAL')
            trend = summary.get('trend', 'NEUTRAL')

            if 'RISELING' in current_phase or trend == 'UP':
                confluence['bullish_signals'].append({
                    'timeframe': f"{tf}min",
                    'reason': current_phase,
                    'confidence': summary['dominant_cycles'][0]['normalized_power'] if summary['dominant_cycles'] else 0.5
                })
                bullish_count += 1

            elif 'FALLING' in current_phase or trend == 'DOWN':
                confluence['bearish_signals'].append({
                    'timeframe': f"{tf}min",
                    'reason': current_phase,
                    'confidence': summary['dominant_cycles'][0]['normalized_power'] if summary['dominant_cycles'] else 0.5
                })
                bearish_count += 1

            else:
                confluence['neutral_signals'].append({
                    'timeframe': f"{tf}min",
                    'reason': current_phase
                })

        # 确定整体情绪
        total_signals = len(results)
        if bullish_count > bearish_count * 1.5:
            confluence['overall_sentiment'] = 'BULLISH'
            confluence['confidence_level'] = bullish_count / total_signals
        elif bearish_count > bullish_count * 1.5:
            confluence['overall_sentiment'] = 'BEARISH'
            confluence['confidence_level'] = bearish_count / total_signals
        else:
            confluence['overall_sentiment'] = 'NEUTRAL'
            confluence['confidence_level'] = max(bullish_count, bearish_count) / total_signals

        return confluence


class ForexPairsHurstAnalyzer:
    """多货币对的蔡森分析系统"""

    def __init__(self):
        self.analyzers = {}
        self.results = {}

    def analyze_pair(self, pair: str, data: pd.DataFrame) -> Dict:
        """
        分析单个货币对

        Args:
            pair: 货币对名称
            data: 价格数据

        Returns:
            分析结果
        """
        # 创建多时间框架分析器
        mtf_analyzer = MultiTimeFrameAnalyzer(data)
        mtf_results = mtf_analyzer.analyze_all_timeframes()
        confluence = mtf_analyzer.generate_confluence_analysis()

        result = {
            'pair': pair,
            'timeframe_analysis': mtf_results,
            'confluence': confluence,
            'analyzers': mtf_analyzer.analyzers,
            'data': data,
            'last_price': data['Close'].iloc[-1],
            'analysis_date': datetime.now()
        }

        self.analyzers[pair] = mtf_analyzer
        self.results[pair] = result

        return result

    def analyze_multiple_pairs(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        分析多个货币对

        Args:
            pairs_data: {货币对: 数据}的字典

        Returns:
            所有货币对的分析结果
        """
        results = {}

        for pair, data in pairs_data.items():
            print(f"正在分析 {pair}...")
            try:
                result = self.analyze_pair(pair, data)
                results[pair] = result
                print(f"✓ {pair} 分析完成")
            except Exception as e:
                print(f"✗ {pair} 分析失败: {e}")
                results[pair] = None

        return results

    def generate_trading_plan(self, lookforward_days: int = 5) -> Dict:
        """
        生成交易计划

        Args:
            lookforward_days: 向前预测的天数

        Returns:
            交易计划
        """
        trading_plan = {
            'overall_recommendations': [],
            'pair_specific_plans': {},
            'risk_warnings': [],
            'key_dates': []
        }

        for pair, result in self.results.items():
            if result is None:
                continue

            confluence = result['confluence']
            last_price = result['last_price']
            sentiment = confluence['overall_sentiment']
            confidence = confluence['confidence_level']

            # 生成具体建议
            plan = {
                'pair': pair,
                'current_price': last_price,
                'sentiment': sentiment,
                'confidence': confidence,
                'recommendation': self._generate_recommendation(sentiment, confidence, pair),
                'entry_zone': self._calculate_entry_zone(result),
                'target_levels': self._calculate_targets(result, sentiment),
                'stop_loss': self._calculate_stop_loss(result, sentiment),
                'time_horizon': 'Short-term (1-5 days)',
                'position_size': 'Moderate' if confidence > 0.6 else 'Small' if confidence > 0.4 else 'Very Small'
            }

            trading_plan['pair_specific_plans'][pair] = plan
            trading_plan['overall_recommendations'].append(plan)

            # 添加关键日期
            if result.get('timeframe_analysis'):
                for tf, tf_result in result['timeframe_analysis'].items():
                    next_tp = tf_result.get('next_turning_point_estimate')
                    if next_tp and next_tp.get('estimated_date'):
                        trading_plan['key_dates'].append({
                            'date': next_tp['estimated_date'],
                            'pair': pair,
                            'event': f"Expected {next_tp['type']} based on {tf}min timeframe"
                        })

        # 风险提示
        trading_plan['risk_warnings'] = [
            "蔡森周期分析基于历史数据，实际市场可能偏离预测",
            "建议结合其他技术分析方法和基本面分析",
            "严格止损，控制风险",
            "在重要经济数据发布前降低仓位"
        ]

        return trading_plan

    def _generate_recommendation(self, sentiment: str, confidence: float, pair: str) -> str:
        """生成交易建议"""
        if confidence < 0.4:
            return "观望 - 等待更明确的信号"

        if sentiment == 'BULLISH':
            if confidence > 0.7:
                return f"强烈买入 - {pair} 在多个时间框架显示看涨"
            else:
                return f"谨慎买入 - {pair} 显示看涨迹象"
        elif sentiment == 'BEARISH':
            if confidence > 0.7:
                return f"强烈卖出 - {pair} 在多个时间框架显示看跌"
            else:
                return f"谨慎卖出 - {pair} 显示看跌迹象"
        else:
            return "中性 - 等待方向确认"

    def _calculate_entry_zone(self, result: Dict) -> str:
        """计算入场区间"""
        last_price = result['last_price']
        atr = self._get_atr(result['data'])

        lower = last_price - 0.5 * atr
        upper = last_price + 0.5 * atr

        return f"{lower:.5f} - {upper:.5f}"

    def _calculate_targets(self, result: Dict, sentiment: str) -> List[str]:
        """计算目标价位"""
        last_price = result['last_price']
        atr = self._get_atr(result['data'])

        targets = []
        if sentiment == 'BULLISH':
            targets.append(f"Target 1: {last_price + atr:.5f}")
            targets.append(f"Target 2: {last_price + 2 * atr:.5f}")
            targets.append(f"Target 3: {last_price + 3 * atr:.5f}")
        elif sentiment == 'BEARISH':
            targets.append(f"Target 1: {last_price - atr:.5f}")
            targets.append(f"Target 2: {last_price - 2 * atr:.5f}")
            targets.append(f"Target 3: {last_price - 3 * atr:.5f}")

        return targets

    def _calculate_stop_loss(self, result: Dict, sentiment: str) -> str:
        """计算止损位"""
        last_price = result['last_price']
        atr = self._get_atr(result['data'])

        if sentiment == 'BULLISH':
            return f"{last_price - 1.5 * atr:.5f}"
        elif sentiment == 'BEARISH':
            return f"{last_price + 1.5 * atr:.5f}"
        else:
            return f"{last_price - 1.5 * atr:.5f} / {last_price + 1.5 * atr:.5f}"

    def _get_atr(self, data: pd.DataFrame, period: int = 14) -> float:
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


def plot_cycle_analysis(analyzer: HurstCyclesAnalyzer, save_path: str = None):
    """
    绘制周期分析图表

    Args:
        analyzer: 蔡森周期分析器
        save_path: 保存路径
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # 图1: 原始价格 vs 去趋势价格
    axes[0].plot(analyzer.dates, analyzer.close, label='Original Price', alpha=0.7)
    detrended = analyzer.detrend_price()
    axes[0].plot(analyzer.dates, detrended + np.mean(analyzer.close), label='Detrended + Offset', alpha=0.7)
    axes[0].set_title('Price vs Detrended Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 图2: 功率谱密度
    periods, psd = analyzer.spectral_analysis()
    axes[1].plot(periods, psd)
    axes[1].set_xlabel('Period (bars)')
    axes[1].set_ylabel('Power Spectral Density')
    axes[1].set_title('Spectral Analysis - Dominant Cycles')
    axes[1].set_xlim(0, min(100, len(analyzer.close) // 2))
    axes[1].grid(True, alpha=0.3)

    # 标记主导周期
    dominant_cycles = analyzer.identify_dominant_cycles()
    for i, cycle in enumerate(dominant_cycles):
        axes[1].axvline(x=cycle['period'], color='red', linestyle='--', alpha=0.5,
                       label=f"Cycle {i+1}: {cycle['period']:.1f}" if i < 3 else "")

    # 图3: 周期转折点
    turning_points = analyzer.identify_cycle_turning_points()
    axes[2].plot(analyzer.dates, analyzer.close, label='Price', alpha=0.7)

    for tp in turning_points:
        color = 'green' if tp['type'] == 'VALLEY' else 'red'
        marker = '^' if tp['type'] == 'VALLEY' else 'v'
        axes[2].scatter(tp['date'], tp['value'], color=color, marker=marker, s=100, alpha=0.7)

    axes[2].set_title('Cycle Turning Points')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 图4: 周期投影
    projection = analyzer.project_cycles_forward(n_periods=20)
    historical_dates = analyzer.dates
    historical_prices = analyzer.close

    # 生成未来日期
    last_date = analyzer.dates[-1]
    future_dates = pd.date_range(start=last_date, periods=len(projection)+1, freq='D')[1:]

    axes[3].plot(historical_dates[-50:], historical_prices[-50:], label='Historical', color='blue', alpha=0.7)
    axes[3].plot(future_dates, projection, label='Projected', color='orange', linestyle='--', alpha=0.7)
    axes[3].axvline(x=last_date, color='red', linestyle='-', alpha=0.3, label='Current')
    axes[3].set_title('Cycle Projection')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    plt.close()


def print_detailed_report(trading_plan: Dict):
    """
    打印详细报告

    Args:
        trading_plan: 交易计划字典
    """
    print("\n" + "="*100)
    print(" "*35 + "蔡森周期分析报告")
    print(" "*30 + "HURST CYCLES ANALYSIS REPORT")
    print("="*100 + "\n")

    # 整体建议
    print("【整体市场分析】")
    print("-"*100)
    for rec in trading_plan['overall_recommendations']:
        print(f"\n{rec['pair']}:")
        print(f"  当前价格: {rec['current_price']:.5f}")
        print(f"  市场情绪: {rec['sentiment']}")
        print(f"  信心水平: {rec['confidence']*100:.1f}%")
        print(f"  交易建议: {rec['recommendation']}")

    # 各货币对详细计划
    print("\n\n" + "="*100)
    print("【各货币对详细交易计划】")
    print("="*100 + "\n")

    for pair, plan in trading_plan['pair_specific_plans'].items():
        print(f"\n{'─'*100}")
        print(f"  {pair} 交易计划")
        print(f"{'─'*100}")
        print(f"  当前价格:        {plan['current_price']:.5f}")
        print(f"  市场情绪:        {plan['sentiment']}")
        print(f"  信心水平:        {plan['confidence']*100:.1f}%")
        print(f"  交易建议:        {plan['recommendation']}")
        print(f"  入场区间:        {plan['entry_zone']}")
        print(f"  目标价位:")
        for target in plan['target_levels']:
            print(f"    {target}")
        print(f"  止损位:          {plan['stop_loss']}")
        print(f"  时间周期:        {plan['time_horizon']}")
        print(f"  建议仓位:        {plan['position_size']}")

    # 关键日期
    if trading_plan['key_dates']:
        print("\n\n" + "="*100)
        print("【关键转折日期预测】")
        print("="*100 + "\n")

        # 按日期排序
        sorted_dates = sorted(trading_plan['key_dates'], key=lambda x: x['date'])
        for item in sorted_dates[:10]:  # 只显示前10个
            print(f"  {item['date'].strftime('%Y-%m-%d')}: {item['pair']} - {item['event']}")

    # 风险提示
    print("\n\n" + "="*100)
    print("【风险提示】")
    print("="*100 + "\n")
    for warning in trading_plan['risk_warnings']:
        print(f"  ⚠ {warning}")

    print("\n" + "="*100)
    print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100 + "\n")
