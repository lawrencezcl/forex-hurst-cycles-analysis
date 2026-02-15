"""
Comprehensive Forex Analysis and Backtesting Runner (Offline Mode)
===================================================================

This version uses generated/simulated forex data to avoid Yahoo Finance rate limits.

Forex Pairs Analyzed:
- EUR/USD, EUR/GBP, EUR/JPY, EUR/CAD
- GBP/USD, GBP/JPY, GBP/AUD
- USD/JPY, USD/CAD
- AUD/USD, AUD/JPY
- CAD/JPY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from generate_forex_data import generate_all_forex_data, generate_forex_data
from forex_backtesting_strategies import (
    SMACrossoverStrategy,
    EMACrossoverStrategy,
    RSIMeanReversionStrategy,
    MACDStrategy,
    BollingerBandsStrategy,
    StochasticStrategy,
    ADXTrendStrategy,
    MultiIndicatorStrategy,
    PriceActionStrategy,
    BacktestEngine
)
from forex_technical_analysis import TechnicalIndicators


class OfflineForexAnalyzer:
    """Offline forex analyzer using generated data"""

    def __init__(self, pairs: list, start_date: str, end_date: str, initial_capital: float = 10000):
        """
        Initialize offline analyzer

        Args:
            pairs: List of forex pair symbols
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            initial_capital: Starting capital for backtests
        """
        self.pairs = pairs
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.all_data = {}
        self.all_results = {}
        self.all_trades = {}

    def generate_data(self):
        """Generate forex data for all pairs"""
        print("\n" + "="*80)
        print("GENERATING FOREX DATA FOR ALL PAIRS")
        print("="*80 + "\n")

        self.all_data = generate_all_forex_data(self.pairs, self.start_date, self.end_date)
        print(f"\n✓ Generated data for {len(self.all_data)} pairs")
        return self.all_data

    def backtest_all_pairs(self):
        """Backtest all strategies on all pairs"""
        print("\n" + "="*80)
        print("BACKTESTING ALL STRATEGIES ON ALL PAIRS")
        print("="*80 + "\n")

        # Define strategies
        strategies = [
            SMACrossoverStrategy(fast_period=20, slow_period=50),
            EMACrossoverStrategy(fast_period=12, slow_period=26),
            RSIMeanReversionStrategy(rsi_period=14, oversold=30, overbought=70),
            MACDStrategy(fast=12, slow=26, signal=9),
            BollingerBandsStrategy(period=20, std_dev=2),
            StochasticStrategy(k_period=14, d_period=3, oversold=20, overbought=80),
            ADXTrendStrategy(adx_period=14, adx_threshold=25, ma_period=50),
            MultiIndicatorStrategy(),
            PriceActionStrategy(lookback=20),
        ]

        engine = BacktestEngine(initial_capital=self.initial_capital)

        for pair, data in self.all_data.items():
            print(f"\nBacktesting {pair}...")

            # Calculate indicators
            data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)

            # Backtest each strategy
            results = []
            for strategy in strategies:
                print(f"  - Testing {strategy.name}...")
                result = engine.backtest_strategy(strategy, data_with_indicators)
                results.append(result)

            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('total_return_pct', ascending=False)
            self.all_results[pair] = results_df

            print(f"✓ {pair} backtesting complete")

        return self.all_results

    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("GENERATING COMPARISON REPORT")
        print("="*80 + "\n")

        comparison_data = []

        for pair, results_df in self.all_results.items():
            if results_df is not None and not results_df.empty:
                for _, row in results_df.iterrows():
                    comparison_data.append({
                        'Pair': pair,
                        'Strategy': row['strategy'],
                        'Total Return (%)': row['total_return_pct'],
                        'Win Rate (%)': row['win_rate'],
                        'Sharpe Ratio': row['sharpe_ratio'],
                        'Max Drawdown (%)': abs(row['max_drawdown']),
                        'Total Trades': row['total_trades'],
                        'Final Capital': row['final_capital'],
                        'Avg Win (%)': row['avg_win'],
                        'Avg Loss (%)': row['avg_loss'],
                    })

        comparison_df = pd.DataFrame(comparison_data)

        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('Total Return (%)', ascending=False)
            comparison_df.reset_index(drop=True, inplace=True)

        return comparison_df

    def find_best_strategies(self, comparison_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """Find best performing strategies"""
        print("\n" + "="*80)
        print(f"TOP {top_n} PERFORMING STRATEGIES")
        print("="*80 + "\n")

        if comparison_df.empty:
            print("No comparison data available")
            return pd.DataFrame()

        # Calculate composite score
        comparison_df['Score'] = (
            comparison_df['Total Return (%)'] * 0.4 +
            comparison_df['Win Rate (%)'] * 0.3 +
            comparison_df['Sharpe Ratio'] * 10 -
            comparison_df['Max Drawdown (%)'] * 0.2
        )

        best_strategies = comparison_df.nlargest(top_n, 'Score')
        print(best_strategies.to_string(index=False))

        return best_strategies

    def analyze_by_strategy_type(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by strategy type"""
        print("\n" + "="*80)
        print("STRATEGY PERFORMANCE ACROSS ALL PAIRS")
        print("="*80 + "\n")

        if comparison_df.empty:
            return pd.DataFrame()

        strategy_summary = comparison_df.groupby('Strategy').agg({
            'Total Return (%)': ['mean', 'std', 'min', 'max', 'count'],
            'Win Rate (%)': 'mean',
            'Sharpe Ratio': 'mean',
            'Max Drawdown (%)': 'mean',
        }).round(2)

        strategy_summary.columns = ['_'.join(col).strip() for col in strategy_summary.columns.values]
        strategy_summary = strategy_summary.sort_values('Total Return (%)_mean', ascending=False)

        print(strategy_summary.to_string())
        return strategy_summary

    def analyze_by_pair(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by forex pair"""
        print("\n" + "="*80)
        print("PAIR PERFORMANCE SUMMARY")
        print("="*80 + "\n")

        if comparison_df.empty:
            return pd.DataFrame()

        pair_summary = comparison_df.groupby('Pair').agg({
            'Total Return (%)': ['mean', 'std', 'min', 'max'],
            'Win Rate (%)': 'mean',
            'Sharpe Ratio': 'mean',
            'Max Drawdown (%)': 'mean',
        }).round(2)

        pair_summary.columns = ['_'.join(col).strip() for col in pair_summary.columns.values]
        pair_summary = pair_summary.sort_values('Total Return (%)_mean', ascending=False)

        print(pair_summary.to_string())
        return pair_summary

    def save_results(self, comparison_df: pd.DataFrame, best_strategies: pd.DataFrame):
        """Save results to CSV files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        comparison_df.to_csv(f'forex_comparison_{timestamp}.csv', index=False)
        print(f"\n✓ Full comparison saved to forex_comparison_{timestamp}.csv")

        best_strategies.to_csv(f'best_strategies_{timestamp}.csv', index=False)
        print(f"✓ Best strategies saved to best_strategies_{timestamp}.csv")

    def generate_visualizations(self, comparison_df: pd.DataFrame):
        """Generate visualization charts"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")

        if comparison_df.empty:
            print("No data to visualize")
            return

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        fig.suptitle('Comprehensive Forex Analysis Results', fontsize=18, fontweight='bold')

        # 1. Top 20 Strategies by Return
        ax1 = fig.add_subplot(gs[0, 0])
        top_20 = comparison_df.head(20)
        colors = ['green' if x > 0 else 'red' for x in top_20['Total Return (%)']]
        ax1.barh(range(len(top_20)), top_20['Total Return (%)'], color=colors)
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels([f"{row['Pair']}\n{row['Strategy']}" for _, row in top_20.iterrows()], fontsize=7)
        ax1.set_xlabel('Total Return (%)')
        ax1.set_title('Top 20 Strategies by Return', fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='x', alpha=0.3)

        # 2. Win Rate vs Return
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(comparison_df['Win Rate (%)'], comparison_df['Total Return (%)'],
                             c=comparison_df['Sharpe Ratio'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
        ax2.set_xlabel('Win Rate (%)', fontweight='bold')
        ax2.set_ylabel('Total Return (%)', fontweight='bold')
        ax2.set_title('Win Rate vs Return (colored by Sharpe Ratio)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Sharpe Ratio', fontweight='bold')

        # 3. Strategy Performance Comparison
        ax3 = fig.add_subplot(gs[1, 0])
        strategy_perf = comparison_df.groupby('Strategy')['Total Return (%)'].mean().sort_values(ascending=False)
        strategy_perf.plot(kind='bar', ax=ax3, color='steelblue', edgecolor='black')
        ax3.set_xlabel('Strategy', fontweight='bold')
        ax3.set_ylabel('Average Return (%)', fontweight='bold')
        ax3.set_title('Average Return by Strategy Type', fontweight='bold')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)

        # 4. Pair Performance Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        pair_perf = comparison_df.groupby('Pair')['Total Return (%)'].mean().sort_values(ascending=False)
        pair_perf.plot(kind='bar', ax=ax4, color='coral', edgecolor='black')
        ax4.set_xlabel('Forex Pair', fontweight='bold')
        ax4.set_ylabel('Average Return (%)', fontweight='bold')
        ax4.set_title('Average Return by Forex Pair', fontweight='bold')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)

        # 5. Win Rate by Strategy
        ax5 = fig.add_subplot(gs[2, 0])
        win_rate_by_strategy = comparison_df.groupby('Strategy')['Win Rate (%)'].mean().sort_values(ascending=False)
        win_rate_by_strategy.plot(kind='bar', ax=ax5, color='purple', edgecolor='black')
        ax5.set_xlabel('Strategy', fontweight='bold')
        ax5.set_ylabel('Win Rate (%)', fontweight='bold')
        ax5.set_title('Win Rate by Strategy Type', fontweight='bold')
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
        ax5.grid(axis='y', alpha=0.3)
        ax5.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% Baseline')
        ax5.legend()

        # 6. Sharpe Ratio by Strategy
        ax6 = fig.add_subplot(gs[2, 1])
        sharpe_by_strategy = comparison_df.groupby('Strategy')['Sharpe Ratio'].mean().sort_values(ascending=False)
        sharpe_by_strategy.plot(kind='bar', ax=ax6, color='darkgreen', edgecolor='black')
        ax6.set_xlabel('Strategy', fontweight='bold')
        ax6.set_ylabel('Sharpe Ratio', fontweight='bold')
        ax6.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontweight='bold')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
        ax6.grid(axis='y', alpha=0.3)
        ax6.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Good (>1)')
        ax6.legend()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'forex_analysis_charts_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Charts saved to forex_analysis_charts_{timestamp}.png")

        plt.show()

    def generate_detailed_report(self, comparison_df: pd.DataFrame):
        """Generate detailed text report"""
        print("\n" + "="*80)
        print("DETAILED ANALYSIS REPORT")
        print("="*80 + "\n")

        if comparison_df.empty:
            return

        print("SUMMARY STATISTICS:")
        print("-" * 80)
        print(f"Total Strategy-Pair Combinations Tested: {len(comparison_df)}")
        print(f"Profitable Combinations: {len(comparison_df[comparison_df['Total Return (%)'] > 0])}")
        print(f"Unprofitable Combinations: {len(comparison_df[comparison_df['Total Return (%)'] <= 0])}")
        print(f"Average Return Across All: {comparison_df['Total Return (%)'].mean():.2f}%")
        print(f"Best Return: {comparison_df['Total Return (%)'].max():.2f}%")
        print(f"Worst Return: {comparison_df['Total Return (%)'].min():.2f}%")
        print(f"Average Win Rate: {comparison_df['Win Rate (%)'].mean():.2f}%")
        print(f"Average Sharpe Ratio: {comparison_df['Sharpe Ratio'].mean():.2f}")

        print("\n\nTOP 5 MOST PROFITABLE COMBINATIONS:")
        print("-" * 80)
        top_5 = comparison_df.head(5)
        for i, row in top_5.iterrows():
            print(f"{i+1}. {row['Pair']} - {row['Strategy']}")
            print(f"   Return: {row['Total Return (%)']:.2f}% | Win Rate: {row['Win Rate (%)']:.2f}% | Sharpe: {row['Sharpe Ratio']:.2f}")

        print("\n\nTOP 5 BEST WIN RATES:")
        print("-" * 80)
        top_win_rate = comparison_df.nlargest(5, 'Win Rate (%)')
        for i, row in top_win_rate.iterrows():
            print(f"{i+1}. {row['Pair']} - {row['Strategy']}")
            print(f"   Win Rate: {row['Win Rate (%)']:.2f}% | Return: {row['Total Return (%)']:.2f}% | Sharpe: {row['Sharpe Ratio']:.2f}")

        print("\n\nTOP 5 BEST SHARPE RATIOS (Risk-Adjusted):")
        print("-" * 80)
        top_sharpe = comparison_df.nlargest(5, 'Sharpe Ratio')
        for i, row in top_sharpe.iterrows():
            print(f"{i+1}. {row['Pair']} - {row['Strategy']}")
            print(f"   Sharpe: {row['Sharpe Ratio']:.2f} | Return: {row['Total Return (%)']:.2f}% | Win Rate: {row['Win Rate (%)']:.2f}%")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("COMPREHENSIVE FOREX ANALYSIS SYSTEM (OFFLINE MODE)")
        print("="*80)
        print(f"Analysis Period: {self.start_date} to {self.end_date}")
        print(f"Pairs Analyzed: {', '.join(self.pairs)}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print("="*80)

        # Phase 1: Generate Data
        self.generate_data()

        # Phase 2: Backtest
        self.backtest_all_pairs()

        # Phase 3: Generate Comparison Report
        comparison_df = self.generate_comparison_report()

        # Phase 4: Find Best Strategies
        best_strategies = self.find_best_strategies(comparison_df, top_n=15)

        # Phase 5: Analyze by Strategy Type
        strategy_summary = self.analyze_by_strategy_type(comparison_df)

        # Phase 6: Analyze by Pair
        pair_summary = self.analyze_by_pair(comparison_df)

        # Phase 7: Generate Detailed Report
        self.generate_detailed_report(comparison_df)

        # Phase 8: Save Results
        self.save_results(comparison_df, best_strategies)

        # Phase 9: Generate Visualizations
        self.generate_visualizations(comparison_df)

        return {
            'comparison_df': comparison_df,
            'best_strategies': best_strategies,
            'strategy_summary': strategy_summary,
            'pair_summary': pair_summary,
        }


def main():
    """Main execution function"""
    # Define analysis parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data

    # Major forex pairs to analyze
    pairs = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
        'EURGBP', 'EURJPY', 'GBPJPY', 'EURCAD', 'GBPAUD',
        'AUDJPY', 'CADJPY'
    ]

    # Initialize analyzer
    analyzer = OfflineForexAnalyzer(
        pairs=pairs,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        initial_capital=10000
    )

    # Run full analysis
    results = analyzer.run_full_analysis()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print(f"✓ Analyzed {len(pairs)} forex pairs")
    print(f"✓ Tested 9 different trading strategies")
    print(f"✓ Tested {len(results['comparison_df'])} strategy-pair combinations")
    print(f"✓ Identified {len(results['best_strategies'])} top-performing strategy combinations")
    print(f"✓ Generated detailed reports and visualizations")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if not results['best_strategies'].empty:
        best = results['best_strategies'].iloc[0]
        print(f"\n🏆 BEST OVERALL STRATEGY: {best['Strategy']} on {best['Pair']}")
        print(f"   - Total Return: {best['Total Return (%)']:.2f}%")
        print(f"   - Win Rate: {best['Win Rate (%)']:.2f}%")
        print(f"   - Sharpe Ratio: {best['Sharpe Ratio']:.2f}")
        print(f"   - Max Drawdown: {best['Max Drawdown (%)']:.2f}%")

        print(f"\n📊 BEST STRATEGY TYPE (Average across all pairs):")
        best_strategy_type = results['strategy_summary'].index[0]
        print(f"   - {best_strategy_type}")
        print(f"   - Average Return: {results['strategy_summary'].iloc[0]['Total Return (%)_mean']:.2f}%")
        print(f"   - Average Win Rate: {results['strategy_summary'].iloc[0]['Win Rate (%)_mean']:.2f}%")

        print(f"\n💱 BEST PERFORMING PAIR (Average across all strategies):")
        best_pair = results['pair_summary'].index[0]
        print(f"   - {best_pair}")
        print(f"   - Average Return: {results['pair_summary'].iloc[0]['Total Return (%)_mean']:.2f}%")

    print("\n" + "="*80)
    print("IMPORTANT NOTES:")
    print("="*80)
    print("1. This analysis uses simulated forex data for demonstration")
    print("2. In live trading, use real market data from reliable sources")
    print("3. Past performance doesn't guarantee future results")
    print("4. Always use proper risk management and position sizing")
    print("5. Consider transaction costs, slippage, and market conditions")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
