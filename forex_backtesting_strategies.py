"""
Forex Trading Strategies and Backtesting Engine
===============================================

This module implements multiple forex trading strategies with comprehensive backtesting capabilities.

Strategies Implemented:
1. Simple Moving Average Crossover (SMA)
2. Exponential Moving Average Crossover (EMA)
3. RSI Mean Reversion
4. MACD Strategy
5. Bollinger Bands Breakout
6. Stochastic Strategy
7. ADX Trend Following
8. Multi-Indicator Confirmation Strategy
9. Price Action Strategy
10. Carry Trade Strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from forex_technical_analysis import ForexDataFetcher, TechnicalIndicators, ForexAnalyzer


class TradingStrategy:
    """Base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name
        self.trades = []
        self.position = None
        self.entry_price = None
        self.entry_date = None

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals (1 = buy, -1 = sell, 0 = hold)"""
        raise NotImplementedError("Subclasses must implement generate_signals()")

    def execute_trades(self, data: pd.DataFrame, signals: pd.Series, initial_capital: float = 10000) -> pd.DataFrame:
        """Execute trades based on signals"""
        capital = initial_capital
        position = 0
        trades = []

        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]

            if signals.iloc[i] == 1 and position == 0:  # Buy signal
                position = capital / current_price
                trades.append({
                    'date': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': position,
                    'capital': capital
                })
                capital = 0

            elif signals.iloc[i] == -1 and position > 0:  # Sell signal
                capital = position * current_price
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': position,
                    'capital': capital
                })
                position = 0

        # Close final position if still open
        if position > 0:
            capital = position * data['Close'].iloc[-1]
            trades.append({
                'date': data.index[-1],
                'action': 'SELL',
                'price': data['Close'].iloc[-1],
                'quantity': position,
                'capital': capital
            })

        return pd.DataFrame(trades)


class SMACrossoverStrategy(TradingStrategy):
    """Simple Moving Average Crossover Strategy"""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__("SMA Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate SMA crossover signals"""
        sma_fast = TechnicalIndicators.sma(data['Close'], self.fast_period)
        sma_slow = TechnicalIndicators.sma(data['Close'], self.slow_period)

        signals = pd.Series(0, index=data.index)
        signals[sma_fast > sma_slow] = 1  # Buy
        signals[sma_fast < sma_slow] = -1  # Sell

        return signals


class EMACrossoverStrategy(TradingStrategy):
    """Exponential Moving Average Crossover Strategy"""

    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        super().__init__("EMA Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate EMA crossover signals"""
        ema_fast = TechnicalIndicators.ema(data['Close'], self.fast_period)
        ema_slow = TechnicalIndicators.ema(data['Close'], self.slow_period)

        signals = pd.Series(0, index=data.index)
        signals[ema_fast > ema_slow] = 1  # Buy
        signals[ema_fast < ema_slow] = -1  # Sell

        return signals


class RSIMeanReversionStrategy(TradingStrategy):
    """RSI Mean Reversion Strategy"""

    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__("RSI Mean Reversion")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI mean reversion signals"""
        rsi = TechnicalIndicators.rsi(data['Close'], self.rsi_period)

        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1  # Buy when oversold
        signals[rsi > self.overbought] = -1  # Sell when overbought

        return signals


class MACDStrategy(TradingStrategy):
    """MACD Strategy"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate MACD signals"""
        macd, signal_line, histogram = TechnicalIndicators.macd(data['Close'], self.fast, self.slow, self.signal)

        signals = pd.Series(0, index=data.index)
        signals[macd > signal_line] = 1  # Buy when MACD crosses above signal
        signals[macd < signal_line] = -1  # Sell when MACD crosses below signal

        return signals


class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands Breakout Strategy"""

    def __init__(self, period: int = 20, std_dev: int = 2):
        super().__init__("Bollinger Bands")
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Bollinger Bands signals"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(data['Close'], self.period, self.std_dev)

        signals = pd.Series(0, index=data.index)
        signals[data['Close'] < lower] = 1  # Buy when below lower band
        signals[data['Close'] > upper] = -1  # Sell when above upper band

        return signals


class StochasticStrategy(TradingStrategy):
    """Stochastic Oscillator Strategy"""

    def __init__(self, k_period: int = 14, d_period: int = 3, oversold: int = 20, overbought: int = 80):
        super().__init__("Stochastic")
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Stochastic signals"""
        stoch_k, stoch_d = TechnicalIndicators.stochastic(data['High'], data['Low'], data['Close'],
                                                          self.k_period, self.d_period)

        signals = pd.Series(0, index=data.index)
        signals[(stoch_k < self.oversold) & (stoch_d < self.oversold)] = 1  # Buy when oversold
        signals[(stoch_k > self.overbought) & (stoch_d > self.overbought)] = -1  # Sell when overbought

        return signals


class ADXTrendStrategy(TradingStrategy):
    """ADX Trend Following Strategy"""

    def __init__(self, adx_period: int = 14, adx_threshold: int = 25, ma_period: int = 50):
        super().__init__("ADX Trend")
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.ma_period = ma_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate ADX trend signals"""
        adx = TechnicalIndicators.adx(data['High'], data['Low'], data['Close'], self.adx_period)
        sma = TechnicalIndicators.sma(data['Close'], self.ma_period)

        signals = pd.Series(0, index=data.index)

        # Buy: Strong trend and price above MA
        signals[(adx > self.adx_threshold) & (data['Close'] > sma)] = 1
        # Sell: Strong trend and price below MA
        signals[(adx > self.adx_threshold) & (data['Close'] < sma)] = -1

        return signals


class MultiIndicatorStrategy(TradingStrategy):
    """Multi-Indicator Confirmation Strategy"""

    def __init__(self):
        super().__init__("Multi-Indicator")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on multiple indicators"""
        rsi = TechnicalIndicators.rsi(data['Close'], 14)
        macd, signal_line, _ = TechnicalIndicators.macd(data['Close'])
        sma_fast = TechnicalIndicators.sma(data['Close'], 20)
        sma_slow = TechnicalIndicators.sma(data['Close'], 50)

        signals = pd.Series(0, index=data.index)

        # Buy signal: RSI not overbought, MACD bullish, price above short MA, short MA above long MA
        buy_conditions = (
            (rsi < 70) &
            (macd > signal_line) &
            (data['Close'] > sma_fast) &
            (sma_fast > sma_slow)
        )

        # Sell signal: RSI not oversold, MACD bearish, price below short MA, short MA below long MA
        sell_conditions = (
            (rsi > 30) &
            (macd < signal_line) &
            (data['Close'] < sma_fast) &
            (sma_fast < sma_slow)
        )

        signals[buy_conditions] = 1
        signals[sell_conditions] = -1

        return signals


class PriceActionStrategy(TradingStrategy):
    """Price Action Strategy based on candlestick patterns"""

    def __init__(self, lookback: int = 20):
        super().__init__("Price Action")
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate price action signals"""
        # Calculate support and resistance levels
        high_max = data['High'].rolling(window=self.lookback).max()
        low_min = data['Low'].rolling(window=self.lookback).min()

        signals = pd.Series(0, index=data.index)

        # Buy signal: Price bounces from support
        signals[
            (data['Low'] <= low_min * 1.01) &
            (data['Close'] > data['Open'])
        ] = 1

        # Sell signal: Price rejects from resistance
        signals[
            (data['High'] >= high_max * 0.99) &
            (data['Close'] < data['Open'])
        ] = -1

        return signals


class BacktestEngine:
    """Backtesting engine for forex strategies"""

    def __init__(self, initial_capital: float = 10000, commission: float = 0.0001):
        """
        Initialize backtest engine

        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as percentage, 0.0001 = 0.01%)
        """
        self.initial_capital = initial_capital
        self.commission = commission

    def backtest_strategy(self, strategy: TradingStrategy, data: pd.DataFrame) -> Dict:
        """
        Backtest a single strategy

        Args:
            strategy: TradingStrategy instance
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with backtest results
        """
        # Generate signals
        signals = strategy.generate_signals(data)

        # Execute trades
        trades_df = strategy.execute_trades(data, signals, self.initial_capital)

        if trades_df.empty:
            return {
                'strategy': strategy.name,
                'total_trades': 0,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
            }

        # Calculate metrics
        final_capital = trades_df['capital'].iloc[-1]
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Calculate win/loss statistics
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']

        if len(sell_trades) > 0:
            # Calculate P&L for each round trip
            pnl_list = []
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_price = buy_trades.iloc[i]['price']
                sell_price = sell_trades.iloc[i]['price']
                buy_cost = buy_price * buy_trades.iloc[i]['quantity']
                sell_revenue = sell_price * sell_trades.iloc[i]['quantity']
                commission_cost = buy_cost * self.commission + sell_revenue * self.commission
                pnl = sell_revenue - buy_cost - commission_cost
                pnl_pct = (pnl / buy_cost) * 100
                pnl_list.append(pnl_pct)

            if pnl_list:
                wins = [p for p in pnl_list if p > 0]
                losses = [p for p in pnl_list if p < 0]

                win_rate = (len(wins) / len(pnl_list)) * 100 if pnl_list else 0
                avg_win = np.mean(wins) if wins else 0
                avg_loss = np.mean(losses) if losses else 0

                # Calculate maximum drawdown
                cumulative_returns = pd.Series([self.initial_capital] + list(trades_df['capital']))
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max * 100
                max_drawdown = drawdown.min()

                # Calculate Sharpe Ratio (simplified)
                if len(pnl_list) > 1:
                    sharpe_ratio = np.mean(pnl_list) / np.std(pnl_list) if np.std(pnl_list) > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                max_drawdown = 0
                sharpe_ratio = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            max_drawdown = 0
            sharpe_ratio = 0

        return {
            'strategy': strategy.name,
            'total_trades': len(sell_trades),
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades_df': trades_df,
            'pnl_list': pnl_list if pnl_list else [],
        }

    def compare_strategies(self, data: pd.DataFrame, strategies: List[TradingStrategy]) -> pd.DataFrame:
        """
        Compare multiple strategies

        Args:
            data: DataFrame with OHLCV data
            strategies: List of TradingStrategy instances

        Returns:
            DataFrame with comparison results
        """
        results = []

        for strategy in strategies:
            print(f"Backtesting {strategy.name}...")
            result = self.backtest_strategy(strategy, data)
            results.append(result)

        results_df = pd.DataFrame(results)

        # Sort by total return
        results_df = results_df.sort_values('total_return_pct', ascending=False)

        return results_df


def run_comprehensive_backtest(pair: str, start_date: str, end_date: str,
                                initial_capital: float = 10000) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run comprehensive backtest on a forex pair

    Args:
        pair: Forex pair symbol (e.g., 'EURUSD')
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        initial_capital: Starting capital

    Returns:
        Tuple of (results DataFrame, dictionary with all trades)
    """
    # Fetch data
    print(f"Fetching data for {pair}...")
    data = ForexDataFetcher.get_forex_data(pair, start_date, end_date)

    if data is None or data.empty:
        raise ValueError(f"Could not fetch data for {pair}")

    # Calculate indicators
    data = TechnicalIndicators.calculate_all_indicators(data)

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

    # Run backtests
    engine = BacktestEngine(initial_capital=initial_capital)
    results_df = engine.compare_strategies(data, strategies)

    # Get all trades
    all_trades = {}
    for strategy in strategies:
        result = engine.backtest_strategy(strategy, data)
        if 'trades_df' in result and not result['trades_df'].empty:
            all_trades[strategy.name] = result['trades_df']

    return results_df, all_trades


if __name__ == "__main__":
    # Example usage
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print("Running comprehensive backtest for EUR/USD...")
    results, trades = run_comprehensive_backtest('EURUSD', start_date.strftime('%Y-%m-%d'),
                                                  end_date.strftime('%Y-%m-%d'))

    print("\n" + "="*80)
    print("STRATEGY COMPARISON RESULTS")
    print("="*80)
    print(results[['strategy', 'total_return_pct', 'win_rate', 'sharpe_ratio', 'max_drawdown']].to_string())
