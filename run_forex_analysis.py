"""
Comprehensive Forex Analysis and Backtesting Runner
===================================================

This script performs comprehensive technical analysis and backtesting
across multiple forex pairs to identify the most profitable strategies.

Forex Pairs Analyzed:
- EUR/USD, EUR/GBP, EUR/JPY, EUR/CAD
- GBP/USD, GBP/JPY, GBP/AUD
- USD/JPY, USD/CAD
- AUD/USD, AUD/JPY
- CAD/JPY

Author: Forex Analysis System
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from forex_technical_analysis import ForexAnalyzer, ForexDataFetcher
from forex_backtesting_strategies import (
    run_comprehensive_backtest,
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


class ComprehensiveForexAnalyzer:
    """Comprehensive forex analysis across multiple pairs"""

    def __init__(self, pairs: list, start_date: str, end_date: str, initial_capital: float = 10000):
        """
        Initialize comprehensive analyzer

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
        self.all_results = {}
        self.all_trades = {}
        self.analysis_summary = {}

    def analyze_all_pairs(self) -> dict:
        """Analyze all forex pairs with technical indicators"""
        print("\n" + "="*80)
        print("PHASE 1: TECHNICAL ANALYSIS OF ALL PAIRS")
        print("="*80 + "\n")

        for pair in self.pairs:
            try:
                print(f"Analyzing {pair}...")
                analyzer = ForexAnalyzer(pair, self.start_date, self.end_date)

                if analyzer.fetch_data():
                    analyzer.calculate_indicators()
                    signals = analyzer.get_current_signals()
                    self.analysis_summary[pair] = {
                        'current_price': analyzer.indicators['Close'].iloc[-1],
                        'signals': signals,
                        'analyzer': analyzer
                    }
                    print(f"✓ {pair} analysis complete")
                else:
                    print(f"✗ Failed to fetch data for {pair}")

            except Exception as e:
                print(f"✗ Error analyzing {pair}: {e}")

        return self.analysis_summary

    def backtest_all_pairs(self) -> tuple:
        """Backtest all strategies on all pairs"""
        print("\n" + "="*80)
        print("PHASE 2: BACKTESTING ALL STRATEGIES ON ALL PAIRS")
        print("="*80 + "\n")

        for pair in self.pairs:
            try:
                print(f"\nBacktesting {pair}...")
                results_df, trades_dict = run_comprehensive_backtest(
                    pair, self.start_date, self.end_date, self.initial_capital
                )
                self.all_results[pair] = results_df
                self.all_trades[pair] = trades_dict
                print(f"✓ {pair} backtesting complete")

            except Exception as e:
                print(f"✗ Error backtesting {pair}: {e}")
                self.all_results[pair] = None

        return self.all_results, self.all_trades

    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive comparison report across all pairs and strategies"""
        print("\n" + "="*80)
        print("PHASE 3: GENERATING COMPARISON REPORT")
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
            # Sort by total return
            comparison_df = comparison_df.sort_values('Total Return (%)', ascending=False)
            comparison_df.reset_index(drop=True, inplace=True)

        return comparison_df

    def find_best_strategies(self, comparison_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Find the best performing strategies"""
        print("\n" + "="*80)
        print(f"TOP {top_n} PERFORMING STRATEGIES")
        print("="*80 + "\n")

        if comparison_df.empty:
            print("No comparison data available")
            return pd.DataFrame()

        # Score strategies based on multiple metrics
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
        """Analyze performance by strategy type across all pairs"""
        print("\n" + "="*80)
        print("STRATEGY PERFORMANCE ACROSS ALL PAIRS")
        print("="*80 + "\n")

        if comparison_df.empty:
            print("No comparison data available")
            return pd.DataFrame()

        strategy_summary = comparison_df.groupby('Strategy').agg({
            'Total Return (%)': ['mean', 'std', 'min', 'max'],
            'Win Rate (%)': 'mean',
            'Sharpe Ratio': 'mean',
            'Max Drawdown (%)': 'mean',
            'Total Trades': 'mean',
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
            print("No comparison data available")
            return pd.DataFrame()

        pair_summary = comparison_df.groupby('Pair').agg({
            'Total Return (%)': ['mean', 'std', 'min', 'max'],
            'Win Rate (%)': 'mean',
            'Sharpe Ratio': 'mean',
            'Max Drawdown (%)': 'mean',
            'Total Trades': 'sum',
        }).round(2)

        pair_summary.columns = ['_'.join(col).strip() for col in pair_summary.columns.values]
        pair_summary = pair_summary.sort_values('Total Return (%)_mean', ascending=False)

        print(pair_summary.to_string())

        return pair_summary

    def save_results(self, comparison_df: pd.DataFrame, best_strategies: pd.DataFrame):
        """Save results to CSV files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save full comparison
        comparison_df.to_csv(f'forex_comparison_{timestamp}.csv', index=False)
        print(f"\n✓ Full comparison saved to forex_comparison_{timestamp}.csv")

        # Save best strategies
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

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Forex Analysis Results', fontsize=16, fontweight='bold')

        # 1. Top 20 Strategies by Return
        ax1 = axes[0, 0]
        top_20 = comparison_df.head(20)
        colors = ['green' if x > 0 else 'red' for x in top_20['Total Return (%)']]
        ax1.barh(range(len(top_20)), top_20['Total Return (%)'], color=colors)
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels([f"{row['Pair']}\n{row['Strategy']}" for _, row in top_20.iterrows()], fontsize=8)
        ax1.set_xlabel('Total Return (%)')
        ax1.set_title('Top 20 Strategies by Return')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='x', alpha=0.3)

        # 2. Win Rate Distribution
        ax2 = axes[0, 1]
        ax2.scatter(comparison_df['Win Rate (%)'], comparison_df['Total Return (%)'],
                   c=comparison_df['Sharpe Ratio'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
        ax2.set_xlabel('Win Rate (%)')
        ax2.set_ylabel('Total Return (%)')
        ax2.set_title('Win Rate vs Return (colored by Sharpe Ratio)')
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Sharpe Ratio')

        # 3. Strategy Performance Comparison
        ax3 = axes[1, 0]
        strategy_perf = comparison_df.groupby('Strategy')['Total Return (%)'].mean().sort_values(ascending=False)
        strategy_perf.plot(kind='bar', ax=ax3, color='steelblue', edgecolor='black')
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Average Return (%)')
        ax3.set_title('Average Return by Strategy Type')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)

        # 4. Pair Performance Comparison
        ax4 = axes[1, 1]
        pair_perf = comparison_df.groupby('Pair')['Total Return (%)'].mean().sort_values(ascending=False)
        pair_perf.plot(kind='bar', ax=ax4, color='coral', edgecolor='black')
        ax4.set_xlabel('Forex Pair')
        ax4.set_ylabel('Average Return (%)')
        ax4.set_title('Average Return by Forex Pair')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'forex_analysis_charts_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Charts saved to forex_analysis_charts_{timestamp}.png")

        plt.show()

    def generate_current_signals_report(self):
        """Generate report of current signals for all pairs"""
        print("\n" + "="*80)
        print("CURRENT TRADING SIGNALS - ALL PAIRS")
        print("="*80 + "\n")

        for pair, data in self.analysis_summary.items():
            print(f"\n{'='*80}")
            print(f"{pair} - Current Price: {data['current_price']:.5f}")
            print(f"{'='*80}")

            for indicator, signal in data['signals'].items():
                print(f"{indicator:20s}: {signal}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("COMPREHENSIVE FOREX ANALYSIS SYSTEM")
        print("="*80)
        print(f"Analysis Period: {self.start_date} to {self.end_date}")
        print(f"Pairs Analyzed: {', '.join(self.pairs)}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print("="*80)

        # Phase 1: Technical Analysis
        self.analyze_all_pairs()

        # Phase 2: Backtesting
        self.backtest_all_pairs()

        # Phase 3: Generate Comparison Report
        comparison_df = self.generate_comparison_report()

        # Phase 4: Find Best Strategies
        best_strategies = self.find_best_strategies(comparison_df, top_n=15)

        # Phase 5: Analyze by Strategy Type
        strategy_summary = self.analyze_by_strategy_type(comparison_df)

        # Phase 6: Analyze by Pair
        pair_summary = self.analyze_by_pair(comparison_df)

        # Phase 7: Save Results
        self.save_results(comparison_df, best_strategies)

        # Phase 8: Generate Visualizations
        self.generate_visualizations(comparison_df)

        # Phase 9: Current Signals
        self.generate_current_signals_report()

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
    start_date = end_date - timedelta(days=365)  # 1 year of data

    # Major forex pairs to analyze
    pairs = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
        'EURGBP', 'EURJPY', 'GBPJPY', 'EURCAD', 'GBPAUD',
        'AUDJPY', 'CADJPY'
    ]

    # Initialize analyzer
    analyzer = ComprehensiveForexAnalyzer(
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
    print(f"✓ Identified {len(results['best_strategies'])} top-performing strategy combinations")
    print(f"✓ Generated detailed reports and visualizations")
    print("\nRecommendations:")
    print("1. Focus on strategies with the highest Score metric")
    print("2. Consider win rate and Sharpe ratio alongside returns")
    print("3. Be cautious of strategies with high drawdowns")
    print("4. Always use proper risk management in live trading")
    print("\n⚠️  Disclaimer: Past performance does not guarantee future results.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
