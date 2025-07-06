# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the "kabu" repository (Japanese for "stock"), a Python-based automated stock trading system that integrates with Rakuten Securities API. The system now includes advanced random walk modeling for enhanced trading decisions.

## Current State

- **Language/Framework**: Python 3.9+ with Poetry for dependency management
- **Project Structure**: Modular architecture with separate components
- **Build System**: Poetry-based dependency management
- **Testing**: pytest framework implemented
- **Key Features**: 
  - Random walk model for price prediction
  - Dual-signal trading system (Moving Average + Random Walk)
  - Real-time visualization and performance tracking
  - Automated stock selection based on volume and volatility

## Architecture

```
src/
├── main.py                   # Main trading system with integrated analysis
├── random_walk.py            # Random walk model implementation
├── trend_indicators.py       # Technical indicators and trend analysis
├── visualizer.py             # Trading visualization and performance metrics
├── test_random_walk.py       # Random walk model test suite
└── test_trend_indicators.py  # Trend analysis test suite
```

## Development Commands

### Setup
```bash
poetry install
```

### Testing (Docker)
```bash
# テストをDockerコンテナで実行
./scripts/run-tests.sh

# または手動でテスト実行
docker-compose --profile test build kabu-test
docker-compose --profile test run --rm kabu-test pytest src/test_random_walk.py -v
```

### Running the System
```bash
# メインシステムの実行
docker-compose up kabu-trading

# デモの実行
./scripts/run-demo.sh
```

### Code Quality
```bash
poetry run black src/
poetry run flake8 src/
poetry run mypy src/
```

## Advanced Trading System Integration

The system now combines multiple sophisticated analysis methods:

1. **Random Walk Model**: Estimates drift and volatility from price history
2. **Monte Carlo Simulation**: Generates probability distributions for future prices
3. **Trend Analysis**: Comprehensive technical indicators (MA, MACD, RSI, Bollinger Bands)
4. **Multi-Signal System**: Combines Random Walk, Moving Average, and Trend signals
5. **Smart Stock Selection**: Prioritizes stocks with strong buy signals from trend indicators
6. **Risk Management**: Uses confidence intervals and probability thresholds

## Key Features

- **Adaptive Parameters**: Model parameters update based on recent price data
- **Comprehensive Technical Analysis**: SMA, EMA, MACD, RSI, Bollinger Bands, Stochastic, Williams %R
- **Intelligent Stock Selection**: Prioritizes stocks with strong bullish trend signals
- **Multi-Layer Signal Validation**: Requires agreement between trend indicators and random walk model
- **Visualization**: Real-time charts showing predictions, confidence intervals, and performance
- **Performance Tracking**: Comprehensive metrics including win rate, Sharpe ratio, and drawdown
- **Robustness**: Handles extreme market conditions and edge cases

## Configuration

Key parameters can be adjusted in `main.py`:
- `window_size`: Historical data window for random walk model (default: 50)
- `buy_threshold/sell_threshold`: Probability thresholds for trading signals (default: 0.55)
- `confidence_level`: Statistical confidence level for predictions (default: 0.95)
- `trend_signal_weight`: Weight of trend signals in stock scoring (default: 0.4)
- `min_trend_strength`: Minimum trend strength for stock selection (default: 30)

## Dependencies

- `numpy`: Numerical computations and random number generation
- `pandas`: Data manipulation and analysis
- `matplotlib`: Visualization and charting
- `scipy`: Statistical functions and distributions
- `requests`: API communication
- `python-dotenv`: Environment variable management