# Financial Markets MCP Configuration

This directory contains the installed MCP servers for professional technical analysis on financial markets.

## Installed MCP Servers

### 1. Yahoo Finance (yfinance) - ENABLED
**Status:** Ready to use (no API key required)
**Features:**
- Stock data, company info, financials, trading metrics
- Recent news articles for stocks
- Historical price data with charts (candlestick, VWAP, volume profile)
- Search stocks, ETFs, mutual funds

### 2. CCXT Crypto Trading - ENABLED
**Status:** Ready to use for public data (API keys optional for trading)
**Features:**
- 20+ cryptocurrency exchanges (Binance, Coinbase, Kraken, etc.)
- Spot, futures, swap markets
- Real-time ticker, order book, OHLCV data
- Trading operations (with API keys)

### 3. Crypto Indicators - ENABLED
**Status:** Ready to use
**Features:**
- Technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)
- Trading signals and strategies

### 4. DexScreener - ENABLED
**Status:** Ready to use
**Features:**
- Real-time on-chain market prices
- DEX data for DeFi tokens
- Liquidity pool information

### 5. AlphaVantage - DISABLED (needs API key)
**Get API key:** https://www.alphavantage.co/support/#api-key
**Features:**
- Stock quotes, company info, time series data
- Options chains with Greeks
- ETF profiles with holdings
- Earnings calendar
- Crypto exchange rates

### 6. Twelve Data - DISABLED (needs API key)
**Get API key:** https://twelvedata.com/register
**Features:**
- Real-time quotes for stocks, forex, crypto
- 100+ technical indicators
- Fundamental data
- U-tool: Natural language API router

### 7. Polygon.io - DISABLED (needs API key)
**Get API key:** https://polygon.io
**Features:**
- Stocks, indices, forex, options data
- Real-time and historical data
- WebSocket streaming

### 8. CoinMarketCap - DISABLED (needs API key)
**Get API key:** https://pro.coinmarketcap.com/account
**Features:**
- Cryptocurrency listings and quotes
- Market data and rankings

### 9. Alpaca Trading - DISABLED (needs API key)
**Get API key:** https://app.alpaca.markets/signup
**Features:**
- Paper and live trading
- Stock and crypto trading
- Portfolio management
- Real-time quotes

## Configuration

The MCP servers are configured in `~/.config/opencode/opencode.json`

### To enable a disabled server:

1. Get the API key from the provider
2. Edit the config file:
   ```bash
   nano ~/.config/opencode/opencode.json
   ```
3. Replace `YOUR_XXX_API_KEY` with your actual key
4. Change `"enabled": false` to `"enabled": true`
5. Restart opencode

## Quick Start

1. **Basic stock analysis (no API key needed):**
   - Use yfinance: "Get AAPL stock info and show recent news"

2. **Crypto analysis (no API key needed):**
   - Use CCXT: "What's the current Bitcoin price on Binance?"
   - Use crypto-indicators: "Calculate RSI for Bitcoin"
   - Use DexScreener: "Show ETH/USDT pool data"

3. **Advanced analysis (requires API keys):**
   - Enable AlphaVantage for options chains, earnings data
   - Enable Twelve Data for 100+ technical indicators
   - Enable Alpaca for paper/live trading

## MCP Server Locations

All MCP servers are cloned to:
```
/root/ideas/caishen/mcp-servers/
├── alpha-vantage-mcp/
├── yfinance-mcp/
├── alpaca-mcp/
├── twelvedata-mcp/
├── polygon-mcp/
├── ccxt-mcp/
├── crypto-indicators-mcp/
├── coinmarket-mcp-server/
└── mcp-dexscreener/
```
