# NIFTY 50 Quantitative Trading System

End-to-end quantitative trading pipeline combining regime detection, options Greeks, and machine learning for Indian equity markets.

## Overview

This project implements a complete quantitative trading system designed for ML Engineer / Quantitative Researcher roles. It demonstrates:

- **Financial Data Engineering**: Multi-asset data pipeline (Spot, Futures, Options)
- **Advanced Feature Engineering**: 50+ features including Black-Scholes Greeks
- **Statistical Modeling**: Hidden Markov Models for regime detection
- **Quantitative Strategy**: EMA crossover with regime filtering and risk management
- **Machine Learning**: XGBoost and LSTM for trade quality enhancement
- **Performance Analysis**: Comprehensive metrics and outlier pattern discovery

## Quick Start

```bash
# Setup environment
python -m venv venv
venv\Scripts\activate  # Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py
```

**Expected Runtime**: 5-15 minutes

**Output**: Visualizations in `results/`, trained models in `models/`, detailed data in CSV files.

## Project Structure

```
├── data/
│   ├── raw/              # Original market data (spot, futures, options)
│   ├── processed/        # Cleaned and merged data
│   └── features/         # Engineered features with Greeks
├── src/
│   ├── data_acquisition/ # Data fetching and cleaning
│   ├── feature_engineering/  # Technical indicators and Greeks
│   ├── regime_detection/ # HMM implementation
│   ├── strategy/         # Trading strategy and backtest
│   ├── ml_models/        # XGBoost and LSTM training
│   ├── analysis/         # Performance and outlier analysis
│   ├── config.py         # Configuration parameters
│   └── utils.py          # Helper functions
├── models/               # Trained models (HMM, XGBoost, LSTM)
├── results/              # Visualizations and reports
├── notebooks/            # Jupyter notebooks for exploration
└── tests/                # Unit tests (structure)
```

## Key Features

### 1. Data Pipeline
- **NIFTY 50 Spot**: 5-minute OHLCV bars
- **NIFTY Futures**: Current month contract with automatic rollover
- **Options Chain**: ATM ± 2 strikes (Calls + Puts) with IV, OI, Volume
- **Data Quality**: Missing data handling, outlier removal, timestamp alignment
- **Dynamic ATM**: Real-time strike calculation based on spot price

### 2. Feature Engineering (50+ Features)

**Technical Indicators**
- EMA (5, 15), RSI (14), Bollinger Bands (20, 2)
- Volume indicators and momentum features

**Options Greeks** (Black-Scholes Model)
- Delta, Gamma, Theta, Vega, Rho
- Calculated for ATM ± 2 strikes (both Calls and Puts)
- Risk-free rate: 6.5% (RBI repo rate)

**Derived Features**
- Implied Volatility metrics (average, skew)
- Put-Call Ratios (OI-based, Volume-based)
- Futures basis and mispricing indicators
- Delta neutrality measures
- Gamma exposure indicators

### 3. Regime Detection (HMM)
- **3 States**: Uptrend, Sideways, Downtrend
- **Input Features**: Options-based only (IV, PCR, basis, Greeks)
- **Algorithm**: Baum-Welch training, Viterbi inference
- **Output**: Probabilistic regime classification for each timestamp

### 4. Trading Strategy
- **Signal**: EMA(5) × EMA(15) crossover
- **Filter**: Trade only in favorable regimes (Long in Uptrend, Short in Downtrend)
- **Risk Management**: 2% stop loss, 4% take profit, 2% position sizing
- **Backtest**: Walk-forward methodology with comprehensive metrics

### 5. Machine Learning
- **XGBoost**: Gradient boosting for tabular features (200 trees, depth 6)
- **LSTM**: Sequential pattern learning (64→32 units, 20-step sequences)
- **Task**: Binary classification - predict trade profitability
- **Enhancement**: ML confidence filtering improves trade quality

### 6. Performance Analysis
- **Metrics**: Sharpe, Sortino, Calmar ratios, Maximum Drawdown, Win Rate
- **Outlier Detection**: 3-sigma method for exceptional trades
- **Pattern Recognition**: Regime, time-of-day, IV environment analysis
- **Statistical Testing**: T-tests, Chi-square for significance

## Results

### Generated Outputs

**Visualizations** (`results/`)
- `regime_visualization.png` - Price with regime colors, timeline, distribution
- `ema_strategy_results.png` - Entry/exit signals, equity curve, drawdown
- `feature_importance.png` - Top 20 features from XGBoost
- `outlier_analysis.png` - Return distribution, outlier patterns

**Data Files** (`results/`)
- `ema_strategy_backtest.csv` - Complete backtest results
- `ema_strategy_trades.csv` - Individual trade records
- `outlier_trades.csv` - Exceptional trade analysis

**Models** (`models/`)
- `hmm_regime_model.pkl` - Regime detector
- `xgboost_model.pkl` - Trade classifier
- `lstm_model.h5` - Sequential learner
- `feature_scaler.pkl` - Feature normalizer

### Key Findings
- Regime filtering significantly reduces false signals
- Options Greeks more predictive than technical indicators
- ML enhancement improves trade quality over quantity
- Outlier trades cluster in specific regimes and time periods

## Tech Stack

**Core**
- Python 3.9+
- NumPy, Pandas - Data manipulation
- Jupyter - Interactive exploration

**Machine Learning**
- scikit-learn - Preprocessing, metrics, utilities
- XGBoost - Gradient boosting
- TensorFlow/Keras - Deep learning
- hmmlearn - Hidden Markov Models

**Financial & Visualization**
- yfinance - Market data
- scipy - Statistical functions
- matplotlib, seaborn - Visualization

## Configuration

Edit `src/config.py` to customize:

```python
# Data
START_DATE = datetime.now() - timedelta(days=365)
TIMEFRAME = '5min'

# Trading
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs
POSITION_SIZE = 0.02       # 2% per trade
STOP_LOSS_PCT = 0.02       # 2%
TAKE_PROFIT_PCT = 0.04     # 4%

# ML
XGBOOST_PARAMS = {'n_estimators': 200, 'max_depth': 6, ...}
LSTM_PARAMS = {'lstm_units': [64, 32], 'epochs': 50, ...}
```

## Documentation

- **[INSTALLATION.md](INSTALLATION.md)** - Detailed setup instructions and troubleshooting
- **[METHODOLOGY.md](METHODOLOGY.md)** - Technical methodology and algorithms
- **[RESULTS.md](RESULTS.md)** - Comprehensive results and analysis
- **[notebooks/](notebooks/)** - Jupyter notebooks for interactive exploration

## Skills Demonstrated

- Financial data engineering and ETL pipelines
- Options pricing theory (Black-Scholes Greeks)
- Statistical modeling (Hidden Markov Models)
- Machine learning (XGBoost, LSTM)
- Quantitative strategy development
- Risk management and backtesting
- Performance analysis and insights
- Professional software engineering practices

## Limitations

- Synthetic data used for demonstration purposes
- No transaction costs or slippage modeled
- Simplified execution assumptions (mid-price fills)
- Single-asset focus (no portfolio optimization)
- Assumes historical patterns persist

## Future Enhancements

- Real-time data integration
- Transaction cost and slippage modeling
- Multi-asset portfolio optimization
- Ensemble ML models
- Live trading infrastructure
- Advanced risk metrics (VaR, CVaR)

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Author

Built for ML Engineer / Quantitative Researcher role demonstration.

Demonstrates: Data Engineering, Feature Engineering, Statistical Modeling, Machine Learning, Risk Management, Performance Analysis.
