"""
Advanced Backtesting System with Backtrader
===========================================

This module provides comprehensive backtesting capabilities for trading strategies.
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class DataFetcher:
    """Fetch stock data for backtesting"""

    @staticmethod
    def get_yahoo_data(ticker, start_date, end_date):
        """Fetch data from Yahoo Finance"""
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None


class SMACrossover(bt.Strategy):
    """Simple Moving Average Crossover Strategy"""

    params = (
        ("ma_short", 20),
        ("ma_long", 50),
        ("printlog", True),
    )

    def __init__(self):
        # Calculate moving averages
        self.ma_short = bt.indicators.SMA(self.data.close, period=self.params.ma_short)
        self.ma_long = bt.indicators.SMA(self.data.close, period=self.params.ma_long)

        # Crossover indicator
        self.crossover = bt.indicators.CrossOver(self.ma_short, self.ma_long)

        # Track orders
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        """Order notification"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        """Trade notification"""
        if not trade.isclosed:
            return

        self.log(f"TRADE PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")

    def log(self, txt, dt=None):
        """Logging function"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")

    def next(self):
        """Main strategy logic"""
        if self.order:
            return

        if not self.position:
            # Buy if short MA crosses above long MA
            if self.crossover > 0:
                size = int(
                    self.broker.cash / self.data.close[0] * 0.95
                )  # Use 95% of cash
                self.order = self.buy(size=size)
        else:
            # Sell if short MA crosses below long MA
            if self.crossover < 0:
                self.order = self.sell(size=self.position.size)


class RSIMeanReversion(bt.Strategy):
    """RSI Mean Reversion Strategy"""

    params = (
        ("rsi_period", 14),
        ("rsi_upper", 70),
        ("rsi_lower", 30),
        ("printlog", True),
    )

    def __init__(self):
        # Calculate RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # Track orders
        self.order = None

    def notify_order(self, order):
        """Order notification"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}")

        self.order = None

    def log(self, txt, dt=None):
        """Logging function"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")

    def next(self):
        """Main strategy logic"""
        if self.order:
            return

        if not self.position:
            # Buy if RSI is oversold
            if self.rsi < self.params.rsi_lower:
                size = int(self.broker.cash / self.data.close[0] * 0.95)
                self.order = self.buy(size=size)
        else:
            # Sell if RSI is overbought
            if self.rsi > self.params.rsi_upper:
                self.order = self.sell(size=self.position.size)


class BollingerBands(bt.Strategy):
    """Bollinger Bands Strategy"""

    params = (
        ("bb_period", 20),
        ("bb_dev", 2),
        ("printlog", True),
    )

    def __init__(self):
        # Calculate Bollinger Bands
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.params.bb_period, devfactor=self.params.bb_dev
        )

        # Track orders
        self.order = None

    def notify_order(self, order):
        """Order notification"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}")

        self.order = None

    def log(self, txt, dt=None):
        """Logging function"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")

    def next(self):
        """Main strategy logic"""
        if self.order:
            return

        if not self.position:
            # Buy if price touches lower band
            if self.data.close[0] <= self.bb.lines.bot[0]:
                size = int(self.broker.cash / self.data.close[0] * 0.95)
                self.order = self.buy(size=size)
        else:
            # Sell if price touches upper band
            if self.data.close[0] >= self.bb.lines.top[0]:
                self.order = self.sell(size=self.position.size)


class BacktestEngine:
    """Backtesting engine with multiple strategies"""

    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash
        self.cerebro = bt.Cerebro()

    def add_data(self, ticker, start_date, end_date):
        """Add data to cerebro"""
        data = DataFetcher.get_yahoo_data(ticker, start_date, end_date)
        if data is not None:
            data_feed = bt.feeds.PandasData(dataname=data)
            self.cerebro.adddata(data_feed)
            return True
        return False

    def add_strategy(self, strategy_name, **kwargs):
        """Add strategy to cerebro"""
        strategies = {
            "sma_crossover": SMACrossover,
            "rsi_mean_reversion": RSIMeanReversion,
            "bollinger_bands": BollingerBands,
        }

        if strategy_name in strategies:
            self.cerebro.addstrategy(strategies[strategy_name], **kwargs)
            return True
        return False

    def set_broker(self, cash=None, commission=0.001):
        """Set broker parameters"""
        if cash is None:
            cash = self.initial_cash
        self.cerebro.broker.setcash(cash)
        self.cerebro.broker.setcommission(commission=commission)

    def add_analyzers(self):
        """Add analyzers for performance metrics"""
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    def run_backtest(self):
        """Run the backtest"""
        self.add_analyzers()
        results = self.cerebro.run()
        return results

    def plot_results(self):
        """Plot backtest results"""
        fig = self.cerebro.plot(style="candlestick", barup="green", bardown="red")
        return fig

    def get_performance_metrics(self, results):
        """Extract performance metrics"""
        strategy = results[0]

        # Get analyzer results
        sharpe_ratio = strategy.analyzers.sharpe.get_analysis()
        returns = strategy.analyzers.returns.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()

        # Calculate additional metrics
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100

        metrics = {
            "initial_cash": self.initial_cash,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio.get("sharperatio", 0),
            "max_drawdown": drawdown.get("max", {}).get("drawdown", 0),
            "max_drawdown_len": drawdown.get("max", {}).get("len", 0),
            "total_trades": trades.get("total", {}).get("closed", 0),
            "won_trades": trades.get("won", {}).get("total", 0),
            "lost_trades": trades.get("lost", {}).get("total", 0),
        }

        if metrics["total_trades"] > 0:
            metrics["win_rate"] = metrics["won_trades"] / metrics["total_trades"] * 100
        else:
            metrics["win_rate"] = 0

        return metrics


def run_comprehensive_backtest(ticker, start_date, end_date, strategies=None):
    """Run comprehensive backtest with multiple strategies"""
    if strategies is None:
        strategies = ["sma_crossover", "rsi_mean_reversion", "bollinger_bands"]

    results = {}

    for strategy in strategies:
        print(f"\n=== Backtesting {strategy} for {ticker} ===")

        engine = BacktestEngine(initial_cash=10000)

        if engine.add_data(ticker, start_date, end_date):
            engine.add_strategy(strategy)
            engine.set_broker()

            backtest_results = engine.run_backtest()
            metrics = engine.get_performance_metrics(backtest_results)

            results[strategy] = metrics

            # Print results
            print(f"Final Value: ${metrics['final_value']:.2f}")
            print(f"Total Return: {metrics['total_return']:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Total Trades: {metrics['total_trades']}")

    return results


def compare_strategies(results):
    """Compare strategy performance"""
    if not results:
        return None

    comparison = pd.DataFrame(results).T

    # Sort by total return
    comparison = comparison.sort_values("total_return", ascending=False)

    print("\n=== Strategy Comparison ===")
    print(
        comparison[["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]].round(
            2
        )
    )

    return comparison


if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print("Starting comprehensive backtest...")
    results = run_comprehensive_backtest(ticker, start_date, end_date)
    comparison = compare_strategies(results)

    print("\nBacktest completed!")
