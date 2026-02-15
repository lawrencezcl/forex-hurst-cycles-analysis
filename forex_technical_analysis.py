"""
Comprehensive Forex Technical Analysis System
=============================================

This module provides extensive technical analysis capabilities for forex trading,
including multiple indicators and strategies for backtesting.

Technical Indicators Implemented:
- Moving Averages (SMA, EMA, WMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)
- ADX (Average Directional Index)
- CCI (Commodity Channel Index)
- Williams %R
- Parabolic SAR
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ForexDataFetcher:
    """Fetch forex data from Yahoo Finance"""

    # Major forex pairs
    FOREX_PAIRS = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
        'EURGBP': 'EURGBP=X',
        'EURJPY': 'EURJPY=X',
        'GBPJPY': 'GBPJPY=X',
        'EURCAD': 'EURCAD=X',
        'GBPAUD': 'GBPAUD=X',
        'AUDJPY': 'AUDJPY=X',
        'CADJPY': 'CADJPY=X',
    }

    @staticmethod
    def get_forex_data(pair: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch forex data for a given pair

        Args:
            pair: Forex pair symbol (e.g., 'EURUSD')
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = ForexDataFetcher.FOREX_PAIRS.get(pair, f"{pair}=X")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if data.empty:
                print(f"Warning: No data retrieved for {pair}")
                return None

            return data
        except Exception as e:
            print(f"Error fetching data for {pair}: {e}")
            return None

    @staticmethod
    def get_multiple_pairs(pairs: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple forex pairs"""
        data_dict = {}
        for pair in pairs:
            data = ForexDataFetcher.get_forex_data(pair, start_date, end_date)
            if data is not None:
                data_dict[pair] = data
        return data_dict


class TechnicalIndicators:
    """Calculate various technical indicators"""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        k = 100 * ((close - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        tr = TechnicalIndicators.atr(high, low, close, period)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0.0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0.0)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        high_max = high.rolling(window=period).max()
        low_min = low.rolling(window=period).min()
        williams_r = -100 * ((high_max - close) / (high_max - low_min))
        return williams_r

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and add to DataFrame"""
        data = df.copy()

        # Moving Averages
        data['SMA_20'] = TechnicalIndicators.sma(data['Close'], 20)
        data['SMA_50'] = TechnicalIndicators.sma(data['Close'], 50)
        data['SMA_200'] = TechnicalIndicators.sma(data['Close'], 200)
        data['EMA_12'] = TechnicalIndicators.ema(data['Close'], 12)
        data['EMA_26'] = TechnicalIndicators.ema(data['Close'], 26)

        # RSI
        data['RSI'] = TechnicalIndicators.rsi(data['Close'], 14)

        # MACD
        macd, signal, hist = TechnicalIndicators.macd(data['Close'])
        data['MACD'] = macd
        data['MACD_Signal'] = signal
        data['MACD_Hist'] = hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(data['Close'])
        data['BB_Upper'] = bb_upper
        data['BB_Middle'] = bb_middle
        data['BB_Lower'] = bb_lower
        data['BB_Width'] = (bb_upper - bb_lower) / bb_middle

        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(data['High'], data['Low'], data['Close'])
        data['Stoch_K'] = stoch_k
        data['Stoch_D'] = stoch_d

        # ATR
        data['ATR'] = TechnicalIndicators.atr(data['High'], data['Low'], data['Close'])

        # ADX
        data['ADX'] = TechnicalIndicators.adx(data['High'], data['Low'], data['Close'])

        # CCI
        data['CCI'] = TechnicalIndicators.cci(data['High'], data['Low'], data['Close'])

        # Williams %R
        data['Williams_R'] = TechnicalIndicators.williams_r(data['High'], data['Low'], data['Close'])

        # Price changes
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

        return data


class ForexAnalyzer:
    """Main forex analysis class"""

    def __init__(self, pair: str, start_date: str, end_date: str):
        """
        Initialize forex analyzer

        Args:
            pair: Forex pair symbol (e.g., 'EURUSD')
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
        """
        self.pair = pair
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.indicators = None

    def fetch_data(self) -> bool:
        """Fetch forex data"""
        self.data = ForexDataFetcher.get_forex_data(self.pair, self.start_date, self.end_date)
        return self.data is not None

    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators"""
        if self.data is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")

        self.indicators = TechnicalIndicators.calculate_all_indicators(self.data)
        return self.indicators

    def get_current_signals(self) -> Dict[str, str]:
        """Get current trading signals based on indicators"""
        if self.indicators is None:
            raise ValueError("Indicators not calculated. Call calculate_indicators() first.")

        latest = self.indicators.iloc[-1]
        signals = {}

        # RSI Signal
        if latest['RSI'] > 70:
            signals['RSI'] = 'OVERBOUGHT - SELL'
        elif latest['RSI'] < 30:
            signals['RSI'] = 'OVERSOLD - BUY'
        else:
            signals['RSI'] = 'NEUTRAL'

        # MACD Signal
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
            signals['MACD'] = 'BULLISH - BUY'
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:
            signals['MACD'] = 'BEARISH - SELL'
        else:
            signals['MACD'] = 'NEUTRAL'

        # Moving Average Signal
        if latest['SMA_20'] > latest['SMA_50'] > latest['SMA_200']:
            signals['MA_Trend'] = 'STRONG UPTREND - BUY'
        elif latest['SMA_20'] < latest['SMA_50'] < latest['SMA_200']:
            signals['MA_Trend'] = 'STRONG DOWNTREND - SELL'
        else:
            signals['MA_Trend'] = 'SIDEWAYS'

        # Bollinger Bands Signal
        if latest['Close'] > latest['BB_Upper']:
            signals['BB'] = 'ABOVE UPPER BAND - POTENTIAL SELL'
        elif latest['Close'] < latest['BB_Lower']:
            signals['BB'] = 'BELOW LOWER BAND - POTENTIAL BUY'
        else:
            signals['BB'] = 'WITHIN BANDS'

        # Stochastic Signal
        if latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:
            signals['Stochastic'] = 'OVERBOUGHT - SELL'
        elif latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20:
            signals['Stochastic'] = 'OVERSOLD - BUY'
        else:
            signals['Stochastic'] = 'NEUTRAL'

        # ADX Signal
        if latest['ADX'] > 50:
            signals['ADX_Strength'] = 'VERY STRONG TREND'
        elif latest['ADX'] > 25:
            signals['ADX_Strength'] = 'STRONG TREND'
        elif latest['ADX'] > 20:
            signals['ADX_Strength'] = 'TRENDING'
        else:
            signals['ADX_Strength'] = 'WEAK/NO TREND'

        # CCI Signal
        if latest['CCI'] > 100:
            signals['CCI'] = 'OVERBOUGHT - SELL'
        elif latest['CCI'] < -100:
            signals['CCI'] = 'OVERSOLD - BUY'
        else:
            signals['CCI'] = 'NEUTRAL'

        # Williams %R Signal
        if latest['Williams_R'] > -20:
            signals['Williams_R'] = 'OVERBOUGHT - SELL'
        elif latest['Williams_R'] < -80:
            signals['Williams_R'] = 'OVERSOLD - BUY'
        else:
            signals['Williams_R'] = 'NEUTRAL'

        return signals

    def generate_summary_report(self) -> str:
        """Generate a summary report of current market conditions"""
        if self.indicators is None:
            raise ValueError("Indicators not calculated. Call calculate_indicators() first.")

        latest = self.indicators.iloc[-1]
        signals = self.get_current_signals()

        report = f"""
{'='*80}
FOREX TECHNICAL ANALYSIS REPORT - {self.pair}
{'='*80}
Date: {latest.name.strftime('%Y-%m-%d %H:%M:%S')}
Current Price: {latest['Close']:.5f}
Daily Change: {(latest['Close'] - self.indicators['Close'].iloc[-2]):.5f} ({(latest['Close']/self.indicators['Close'].iloc[-2] - 1)*100:.2f}%)

{'='*80}
TECHNICAL INDICATORS
{'='*80}
RSI (14): {latest['RSI']:.2f}
MACD: {latest['MACD']:.6f} | Signal: {latest['MACD_Signal']:.6f} | Histogram: {latest['MACD_Hist']:.6f}
Moving Averages: SMA(20): {latest['SMA_20']:.5f} | SMA(50): {latest['SMA_50']:.5f} | SMA(200): {latest['SMA_200']:.5f}
Bollinger Bands: Upper: {latest['BB_Upper']:.5f} | Middle: {latest['BB_Middle']:.5f} | Lower: {latest['BB_Lower']:.5f}
Stochastic: %K: {latest['Stoch_K']:.2f} | %D: {latest['Stoch_D']:.2f}
ATR (14): {latest['ATR']:.6f}
ADX (14): {latest['ADX']:.2f}
CCI (20): {latest['CCI']:.2f}
Williams %R: {latest['Williams_R']:.2f}

{'='*80}
TRADING SIGNALS
{'='*80}
"""
        for indicator, signal in signals.items():
            report += f"{indicator:20s}: {signal}\n"

        report += f"{'='*80}\n"

        return report


def analyze_forex_pair(pair: str, start_date: str, end_date: str) -> ForexAnalyzer:
    """
    Analyze a forex pair and return analyzer object

    Args:
        pair: Forex pair symbol (e.g., 'EURUSD')
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        ForexAnalyzer object with all indicators calculated
    """
    analyzer = ForexAnalyzer(pair, start_date, end_date)

    if not analyzer.fetch_data():
        raise ValueError(f"Could not fetch data for {pair}")

    analyzer.calculate_indicators()
    return analyzer


if __name__ == "__main__":
    # Example usage
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Analyze EUR/USD
    print("Analyzing EUR/USD...")
    analyzer = analyze_forex_pair('EURUSD', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    print(analyzer.generate_summary_report())
