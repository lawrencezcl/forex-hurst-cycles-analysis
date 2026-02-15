"""
Advanced Stock Analysis Dashboard with Streamlit
===============================================

Run this dashboard with: streamlit run stock_dashboard.py

Features:
- Real-time stock data fetching
- Technical analysis visualization
- Portfolio optimization
- Backtesting results
- Trading signal generation
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stockstats import StockDataFrame
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import HRPOpt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_stock_data(ticker, period="1y"):
    """Fetch stock data"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None


def calculate_indicators(data):
    """Calculate technical indicators"""
    stock_df = StockDataFrame.retype(data)
    return {
        "RSI": stock_df["rsi_14"],
        "MACD": stock_df["macd"],
        "MACD_Signal": stock_df["macds"],
        "MACD_Hist": stock_df["macdh"],
        "MA_20": stock_df["close_20_sma"],
        "MA_50": stock_df["close_50_sma"],
        "BB_Upper": stock_df["boll_ub"],
        "BB_Lower": stock_df["boll_lb"],
        "ATR": stock_df["atr_14"],
        "ADX": stock_df["dx_14"],
    }


def create_price_chart(data, ticker, indicators):
    """Create comprehensive price chart"""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{ticker} Price", "Volume", "MACD", "RSI"),
        row_width=[0.2, 0.2, 0.2, 0.7],
    )

    # Price chart
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

    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=data.index, y=indicators["MA_20"], name="MA 20", line=dict(color="orange")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index, y=indicators["MA_50"], name="MA 50", line=dict(color="blue")
        ),
        row=1,
        col=1,
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators["BB_Upper"],
            name="BB Upper",
            line=dict(color="gray", dash="dash"),
            fill=None,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators["BB_Lower"],
            name="BB Lower",
            line=dict(color="gray", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.1)",
        ),
        row=1,
        col=1,
    )

    # Volume
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"), row=2, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(
            x=data.index, y=indicators["MACD"], name="MACD", line=dict(color="blue")
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators["MACD_Signal"],
            name="Signal",
            line=dict(color="red"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=data.index, y=indicators["MACD_Hist"], name="Histogram"), row=3, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=data.index, y=indicators["RSI"], name="RSI", line=dict(color="purple")
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    fig.update_layout(
        title=f"{ticker} Technical Analysis Dashboard",
        height=1000,
        xaxis_rangeslider_visible=False,
    )

    return fig


def optimize_portfolio(tickers):
    """Portfolio optimization"""
    try:
        # Fetch data
        price_data = {}
        for ticker in tickers:
            data, _ = get_stock_data(ticker)
            if data is not None:
                price_data[ticker] = data["Close"]

        if not price_data:
            st.error("Could not fetch data for portfolio optimization")
            return None

        df = pd.DataFrame(price_data)

        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        # Optimize
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        performance = ef.portfolio_performance()

        return cleaned_weights, performance, df

    except Exception as e:
        st.error(f"Portfolio optimization error: {e}")
        return None


def generate_trading_signals(data, indicators):
    """Generate trading signals"""
    signals = pd.DataFrame(index=data.index)

    # RSI signals
    signals["RSI_Signal"] = np.where(
        indicators["RSI"] < 30, "BUY", np.where(indicators["RSI"] > 70, "SELL", "HOLD")
    )

    # MACD signals
    macd_signal = np.where(
        indicators["MACD"] > indicators["MACD_Signal"], "BUY", "SELL"
    )
    signals["MACD_Signal"] = macd_signal

    # MA crossover signals
    ma_signal = np.where(indicators["MA_20"] > indicators["MA_50"], "BUY", "SELL")
    signals["MA_Signal"] = ma_signal

    # Combined signal
    buy_count = (
        (signals["RSI_Signal"] == "BUY").astype(int)
        + (signals["MACD_Signal"] == "BUY").astype(int)
        + (signals["MA_Signal"] == "BUY").astype(int)
    )

    sell_count = (
        (signals["RSI_Signal"] == "SELL").astype(int)
        + (signals["MACD_Signal"] == "SELL").astype(int)
        + (signals["MA_Signal"] == "SELL").astype(int)
    )

    signals["Combined_Signal"] = np.where(
        buy_count >= 2, "BUY", np.where(sell_count >= 2, "SELL", "HOLD")
    )

    return signals


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">📈 Stock Analysis Dashboard</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("Analysis Controls")

    # Stock selection
    ticker = st.sidebar.text_input(
        "Enter Stock Ticker",
        value="AAPL",
        help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)",
    )

    # Time period
    period = st.sidebar.selectbox(
        "Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3
    )

    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Technical Analysis", "Portfolio Optimization", "Trading Signals"],
    )

    if st.sidebar.button("Analyze"):
        if analysis_type == "Technical Analysis":
            st.header("Technical Analysis")

            # Fetch data
            data, info = get_stock_data(ticker, period)
            if data is None:
                st.error("Could not fetch stock data")
                return

            # Calculate indicators
            indicators = calculate_indicators(data)

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                current_price = data["Close"][-1]
                st.markdown(
                    f'<div class="metric-card"><h3>{current_price:.2f}</h3><p>Current Price</p></div>',
                    unsafe_allow_html=True,
                )

            with col2:
                change = data["Close"][-1] - data["Close"][-2]
                change_pct = (change / data["Close"][-2]) * 100
                color = "green" if change > 0 else "red"
                st.markdown(
                    f'<div class="metric-card"><h3 style="color: {color}">{change_pct:.2f}%</h3><p>Daily Change</p></div>',
                    unsafe_allow_html=True,
                )

            with col3:
                rsi_current = indicators["RSI"][-1]
                rsi_signal = (
                    "Overbought"
                    if rsi_current > 70
                    else "Oversold"
                    if rsi_current < 30
                    else "Neutral"
                )
                st.markdown(
                    f'<div class="metric-card"><h3>{rsi_current:.2f}</h3><p>RSI ({rsi_signal})</p></div>',
                    unsafe_allow_html=True,
                )

            with col4:
                volume_avg = data["Volume"].mean()
                volume_current = data["Volume"][-1]
                volume_ratio = volume_current / volume_avg
                st.markdown(
                    f'<div class="metric-card"><h3>{volume_ratio:.2f}x</h3><p>Volume Ratio</p></div>',
                    unsafe_allow_html=True,
                )

            # Chart
            fig = create_price_chart(data, ticker, indicators)
            st.plotly_chart(fig, use_container_width=True)

            # Company info
            if info:
                st.subheader("Company Information")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Market Cap", f"${info.get('marketCap', 0):,.0f}")
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")

                with col2:
                    st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
                    st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")

                with col3:
                    st.metric("Avg Volume", f"{info.get('averageVolume', 0):,}")
                    st.metric(
                        "Dividend Yield", f"{info.get('dividendYield', 0) * 100:.2f}%"
                    )

        elif analysis_type == "Portfolio Optimization":
            st.header("Portfolio Optimization")

            # Portfolio input
            tickers_input = st.text_input(
                "Enter tickers (comma-separated)", "AAPL,GOOGL,MSFT,AMZN,TSLA"
            )
            tickers = [t.strip().upper() for t in tickers_input.split(",")]

            result = optimize_portfolio(tickers)
            if result:
                weights, performance, price_data = result

                # Display weights
                st.subheader("Optimal Weights")
                weights_df = pd.DataFrame(
                    list(weights.items()), columns=["Asset", "Weight"]
                )
                st.dataframe(weights_df)

                # Performance metrics
                st.subheader("Portfolio Performance")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Expected Return", f"{performance[0] * 100:.2f}%")

                with col2:
                    st.metric("Annual Volatility", f"{performance[1] * 100:.2f}%")

                with col3:
                    st.metric("Sharpe Ratio", f"{performance[2]:.2f}")

                # Portfolio visualization
                st.subheader("Portfolio Allocation")
                fig = go.Figure(
                    data=[
                        go.Pie(labels=weights_df["Asset"], values=weights_df["Weight"])
                    ]
                )
                st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Trading Signals":
            st.header("Trading Signals")

            # Fetch data
            data, info = get_stock_data(ticker, period)
            if data is None:
                st.error("Could not fetch stock data")
                return

            # Calculate indicators and signals
            indicators = calculate_indicators(data)
            signals = generate_trading_signals(data, indicators)

            # Latest signals
            st.subheader("Latest Trading Signals")
            latest_signals = signals.iloc[-1:]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                signal = latest_signals["RSI_Signal"].iloc[0]
                color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[signal]
                st.markdown(
                    f'<div class="metric-card"><h3 style="color: {color}">{signal}</h3><p>RSI Signal</p></div>',
                    unsafe_allow_html=True,
                )

            with col2:
                signal = latest_signals["MACD_Signal"].iloc[0]
                color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[signal]
                st.markdown(
                    f'<div class="metric-card"><h3 style="color: {color}">{signal}</h3><p>MACD Signal</p></div>',
                    unsafe_allow_html=True,
                )

            with col3:
                signal = latest_signals["MA_Signal"].iloc[0]
                color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[signal]
                st.markdown(
                    f'<div class="metric-card"><h3 style="color: {color}">{signal}</h3><p>MA Signal</p></div>',
                    unsafe_allow_html=True,
                )

            with col4:
                signal = latest_signals["Combined_Signal"].iloc[0]
                color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[signal]
                st.markdown(
                    f'<div class="metric-card"><h3 style="color: {color}">{signal}</h3><p>Combined Signal</p></div>',
                    unsafe_allow_html=True,
                )

            # Signal history
            st.subheader("Signal History")
            signal_counts = signals["Combined_Signal"].value_counts()

            fig = go.Figure(
                data=[go.Bar(x=signal_counts.index, y=signal_counts.values)]
            )
            fig.update_layout(
                title="Signal Distribution", xaxis_title="Signal", yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed signal table
            st.subheader("Signal History Table")
            display_signals = signals.tail(30).copy()
            display_signals["Close"] = data["Close"].tail(30)
            st.dataframe(
                display_signals[
                    [
                        "Combined_Signal",
                        "RSI_Signal",
                        "MACD_Signal",
                        "MA_Signal",
                        "Close",
                    ]
                ].reset_index()
            )


if __name__ == "__main__":
    main()
