"""
Generate Sample Forex Data for Testing
======================================

Since Yahoo Finance has rate limits, this script generates realistic
forex data for testing the trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_forex_data(pair: str, start_date: str, end_date: str, seed: int = None) -> pd.DataFrame:
    """
    Generate realistic forex price data

    Args:
        pair: Forex pair symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    # Remove weekends (forex markets are closed on weekends)
    dates = dates[dates.dayofweek < 5]

    # Initial price based on pair
    initial_prices = {
        'EURUSD': 1.1000, 'GBPUSD': 1.3000, 'USDJPY': 150.00,
        'AUDUSD': 0.6500, 'USDCAD': 1.3500, 'EURGBP': 0.8600,
        'EURJPY': 165.00, 'GBPJPY': 195.00, 'EURCAD': 1.4850,
        'GBPAUD': 1.9500, 'AUDJPY': 97.50, 'CADJPY': 111.00,
    }

    # Get initial price or use default
    initial_price = initial_prices.get(pair, 1.0000)

    # Generate price movements using geometric Brownian motion
    n_days = len(dates)

    # Parameters for GBM
    dt = 1/252  # Daily
    mu = 0.05   # Annual drift (5%)
    sigma = 0.12  # Annual volatility (12%)

    # Generate random price changes
    random_shocks = np.random.standard_normal(n_days)

    # Calculate prices
    prices = [initial_price]
    for i in range(1, n_days):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * random_shocks[i]
        new_price = prices[-1] * np.exp(drift + diffusion)
        prices.append(new_price)

    prices = np.array(prices)

    # Generate OHLC from close prices
    high = prices * (1 + np.abs(np.random.normal(0, 0.002, n_days)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.002, n_days)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]

    # Generate volume (random but with some patterns)
    base_volume = 100000
    volume = base_volume * (1 + np.random.normal(0, 0.3, n_days))
    volume = np.abs(volume).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_prices,
        'High': np.maximum(high, np.maximum(open_prices, prices)),
        'Low': np.minimum(low, np.minimum(open_prices, prices)),
        'Close': prices,
        'Volume': volume,
    }, index=dates)

    return df


def generate_all_forex_data(pairs: list, start_date: str, end_date: str) -> dict:
    """Generate data for multiple forex pairs"""
    data_dict = {}

    for i, pair in enumerate(pairs):
        print(f"Generating data for {pair}...")
        data_dict[pair] = generate_forex_data(pair, start_date, end_date, seed=i)

    return data_dict


if __name__ == "__main__":
    # Generate sample data
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    start_date = '2024-01-01'
    end_date = '2025-01-01'

    data = generate_all_forex_data(pairs, start_date, end_date)

    for pair, df in data.items():
        print(f"\n{pair}:")
        print(df.head())
        print(f"Final price: {df['Close'].iloc[-1]:.5f}")
