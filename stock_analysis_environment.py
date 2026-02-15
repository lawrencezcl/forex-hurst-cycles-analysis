"""
Stock Analysis Environment Setup
=================================

This script demonstrates the essential stock analysis capabilities available
with the installed libraries.

Core Libraries Installed:
- yfinance: Yahoo Finance data retrieval
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib/seaborn: Data visualization
- plotly: Interactive charts
- stockstats: Technical analysis indicators
- backtrader: Backtesting framework
- pyportfolioopt: Portfolio optimization
- cvxpy: Convex optimization
- quantlib: Quantitative finance
- bt: Backtesting
- streamlit: Web app framework
- dash: Interactive web applications
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stockstats import StockDataFrame
import backtrader as bt
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import HRPOpt, black_litterman, BlackLittermanModel


def get_stock_data(ticker, period="1y"):
    """Fetch stock data using yfinance"""
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data


def calculate_technical_indicators(data):
    """Calculate technical indicators using stockstats"""
    stock_df = StockDataFrame.retype(data)
    indicators = {
        "RSI": stock_df["rsi_14"],
        "MACD": stock_df["macd"],
        "MA_20": stock_df["close_20_sma"],
        "MA_50": stock_df["close_50_sma"],
        "BB_upper": stock_df["boll_ub"],
        "BB_lower": stock_df["boll_lb"],
        "Volume_MA": stock_df["volume_20_sma"],
    }
    return indicators


def create_interactive_chart(data, ticker):
    """Create interactive candlestick chart with technical indicators"""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{ticker} Price", "Volume", "RSI"),
        row_width=[0.2, 0.2, 0.7],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Volume
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"), row=2, col=1)

    # RSI
    stock_df = StockDataFrame.retype(data)
    rsi = stock_df["rsi_14"]
    fig.add_trace(
        go.Scatter(x=data.index, y=rsi, name="RSI", line=dict(color="purple")),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker} Technical Analysis",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=800,
    )

    return fig


def optimize_portfolio(tickers):
    """Modern portfolio theory optimization"""
    # Fetch data for all tickers
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = get_stock_data(ticker)["Close"]
        except:
            print(f"Could not fetch data for {ticker}")
            return None

    df = pd.DataFrame(data)

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimize for maximum Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    return cleaned_weights, ef.portfolio_performance(verbose=True)


def basic_backtest(ticker, strategy="sma_cross"):
    """Simple backtesting example using backtrader"""

    class SMACrossStrategy(bt.Strategy):
        params = (("ma_period", 20),)

        def __init__(self):
            self.ma = bt.indicators.SimpleMovingAverage(
                self.data.close, period=self.params.ma_period
            )

        def next(self):
            if not self.position:
                if self.data.close > self.ma:
                    self.buy()
            else:
                if self.data.close < self.ma:
                    self.sell()

    # Create cerebro instance
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(SMACrossStrategy)

    # Add data
    data = bt.feeds.PandasData(dataname=get_stock_data(ticker))
    cerebro.adddata(data)

    # Set initial cash
    cerebro.broker.setcash(10000)

    # Run backtest
    results = cerebro.run()

    return cerebro.broker.getvalue()


def generate_signals(data):
    """Generate trading signals based on technical indicators"""
    stock_df = StockDataFrame.retype(data)

    signals = pd.DataFrame(index=data.index)

    # RSI signals
    signals["RSI"] = stock_df["rsi_14"]
    signals["RSI_Signal"] = np.where(
        signals["RSI"] < 30, "BUY", np.where(signals["RSI"] > 70, "SELL", "HOLD")
    )

    # MACD signals
    signals["MACD"] = stock_df["macd"]
    signals["MACD_Signal"] = stock_df["macds"]
    signals["MACD_Histogram"] = stock_df["macdh"]

    # Moving average crossover
    signals["MA_Short"] = stock_df["close_20_sma"]
    signals["MA_Long"] = stock_df["close_50_sma"]
    signals["MA_Signal"] = np.where(
        signals["MA_Short"] > signals["MA_Long"], "BUY", "SELL"
    )

    return signals


if __name__ == "__main__":
    print("Stock Analysis Environment - Ready to Use!")
    print("\nAvailable functions:")
    print("- get_stock_data(ticker, period)")
    print("- calculate_technical_indicators(data)")
    print("- create_interactive_chart(data, ticker)")
    print("- optimize_portfolio(tickers)")
    print("- basic_backtest(ticker, strategy)")
    print("- generate_signals(data)")
    print("\nExample usage:")
    print("data = get_stock_data('AAPL', '1y')")
    print("indicators = calculate_technical_indicators(data)")
    print("chart = create_interactive_chart(data, 'AAPL')")
    print("weights, performance = optimize_portfolio(['AAPL', 'GOOGL', 'MSFT'])")
    print("final_value = basic_backtest('AAPL')")
    print("signals = generate_signals(data)")
