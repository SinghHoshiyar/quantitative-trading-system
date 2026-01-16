# 🚀 Quantitative Trading System - NIFTY 50

> **Professional-grade end-to-end quantitative trading pipeline for ML Engineer / Quantitative Researcher role**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start-5-minutes)
- [Features](#-key-features)
- [Project Structure](#-project-structure)
- [Results](#-what-youll-get)
- [Documentation](#-documentation)
- [Technologies](#-technologies-used)

---

## 🎯 Overview

A **complete quantitative trading system** that demonstrates:

✅ **Financial Data Engineering** - Multi-asset data pipeline (Spot, Futures, Options)  
✅ **Advanced Feature Engineering** - 50+ features including Options Greeks  
✅ **Statistical Modeling** - Hidden Markov Models for regime detection  
✅ **Machine Learning** - XGBoost + LSTM for trade prediction  
✅ **Risk Management** - Professional position sizing and stop losses  
✅ **Performance Analysis** - Comprehensive metrics and outlier detection  

**Built for**: Demonstrating real-world quant research capabilities  
**Time Period**: 1 year of 5-minute data  
**Market**: NIFTY 50 (Indian equity index)  

---

## ⚡ Quick Start (5 Minutes)

### 1. Setup Environment

```bash
# Clone or download this repository
cd quantitative-trading-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python run_pipeline.py
```

**That's it!** The system will automatically:
- Fetch and clean market data
- Engineer 50+ features
- Detect market regimes using HMM
- Backtest trading strategy
- Train ML models (XGBoost + LSTM)
- Analyze high-performance trades
- Generate professional visualizations

**Expected Runtime**: 5-10 minutes

### 3. View Results

Check the `results/` directory for:
- 📊 `regime_visualization.png` - Market regime analysis
- 📈 `ema_strategy_results.png` - Trading performance
- 🎯 `feature_importance.png` - ML feature analysis
- 💎 `outlier_analysis.png` - Exceptional trade patterns

---

## 🌟 Key Features

### 1. Data Pipeline
- **Multi-Asset Integration**: Spot, Futures, Options
- **Data Quality**: Missing data handling, outlier removal
- **Timestamp Alignment**: Synchronized across instruments
- **Futures Rollover**: Automatic expiry handling
- **Dynamic ATM**: Real-time strike calculation

### 2. Feature Engineering (50+ Features)

**Technical Indicators**
- EMA (5, 15), RSI, Bollinger Bands
- Volume indicators, Momentum features

**Options Greeks** (Black-Scholes)
- Delta, Gamma, Theta, Vega, Rho
- For ATM ± 2 strikes (Calls + Puts)

**Derived Features**
- Implied Volatility metrics
- Put-Call Ratios (OI, Volume)
- Futures basis and mispricing
- Delta neutrality measures
- Gamma exposure indicators

### 3. Regime Detection (HMM)
- **3 States**: Uptrend, Sideways, Downtrend
- **Options-Based**: Uses derivatives for classification
- **Probabilistic**: Transition matrix analysis
- **Validated**: Regime-specific performance metrics

### 4. Trading Strategy
- **Signal**: EMA(5) × EMA(15) crossover
- **Filter**: Trade only in favorable regimes
- **Risk Management**: 2% stop loss, 4% take profit
- **Position Sizing**: 2% capital per trade
- **Metrics**: Sharpe, Sortino, Calmar, Max Drawdown

### 5. Machine Learning
- **XGBoost**: Gradient boosting for tabular features
- **LSTM**: Deep learning for sequential patterns
- **Binary Classification**: Trade profitability prediction
- **Confidence Filtering**: Only high-confidence trades
- **Feature Importance**: Identifies key drivers

### 6. Performance Analysis
- **Outlier Detection**: 3-sigma exceptional trades
- **Pattern Recognition**: What makes trades exceptional?
- **Statistical Testing**: Hypothesis validation
- **Insights**: Regime, time, IV correlations

---

## 📁 Project Structure

```
quantitative-trading-system/
│
├── 📄 START_HERE.md              ⭐ Begin here!
├── 📄 run_pipeline.py            🚀 Main execution script
├── 📄 requirements.txt           📦 Dependencies
│
├── 📁 src/                       💻 Source Code
│   ├── config.py                 ⚙️  Configuration
│   ├── utils.py                  🛠️  Helper functions
│   ├── data_acquisition/         📥 Data fetching & cleaning
│   ├── feature_engineering/      🔧 Feature creation
│   ├── regime_detection/         🎯 HMM implementation
│   ├── strategy/                 📈 Trading strategy
│   ├── ml_models/                🤖 ML training
│   └── analysis/                 📊 Performance analysis
│
├── 📁 data/                      💾 Market Data
│   ├── raw/                      Original data
│   ├── processed/                Cleaned data
│   └── features/                 Engineered features
│
├── 📁 models/                    🧠 Trained Models
│   ├── hmm_regime_model.pkl      Regime detector
│   ├── xgboost_model.pkl         Trade classifier
│   ├── lstm_model.h5             Sequential learner
│   └── feature_scaler.pkl        Feature normalizer
│
├── 📁 results/                   📊 Outputs
│   ├── *.png                     Visualizations
│   └── *.csv                     Detailed results
│
├── 📁 notebooks/                 📓 Jupyter Notebooks
│   └── 01_exploratory_analysis.ipynb
│
└── 📚 Documentation/              📖 Comprehensive Guides
    ├── QUICKSTART.md             5-minute setup
    ├── METHODOLOGY.md            Technical details
    ├── SETUP_GUIDE.md            Installation help
    ├── PRESENTATION_OUTLINE.md   PPT structure
    └── ... (10+ documentation files)
```

---

## 📊 What You'll Get

### Visualizations (PNG)
1. **Regime Visualization** - Price with regime colors, timeline, distribution
2. **Strategy Results** - Entry/exit signals, equity curve, drawdown
3. **Feature Importance** - Top 20 features from XGBoost
4. **Outlier Analysis** - Exceptional trade patterns and insights

### Data Files (CSV)
1. **Backtest Results** - Complete trade-by-trade data
2. **Individual Trades** - Entry, exit, PnL, duration
3. **Outlier Trades** - High-performance trades with features

### Models (PKL/H5)
1. **HMM Model** - Regime detector
2. **XGBoost Model** - Trade classifier
3. **LSTM Model** - Sequential learner
4. **Feature Scaler** - Normalization

### Metrics
- Total Return, Sharpe Ratio, Sortino Ratio
- Calmar Ratio, Maximum Drawdown
- Win Rate, Average Trade Duration
- ML Accuracy, AUC-ROC

---

## 📚 Documentation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **START_HERE.md** | First steps | Starting out |
| **QUICKSTART.md** | 5-min setup | Want to run immediately |
| **WHAT_WE_BUILT.md** | Project explanation | Understanding scope |
| **METHODOLOGY.md** | Technical details | Deep dive into algorithms |
| **SETUP_GUIDE.md** | Installation help | Troubleshooting |
| **COMMANDS.md** | Command reference | Need specific commands |
| **PRESENTATION_OUTLINE.md** | PPT structure | Creating presentation |
| **COMPLETION_CHECKLIST.md** | Pre-submission | Before interview |
| **PROJECT_SUMMARY.md** | Complete overview | Final review |
| **PROJECT_MAP.md** | Navigation guide | Finding information |

---

## 🛠️ Technologies Used

### Core
- **Python 3.9+** - Programming language
- **NumPy, Pandas** - Data manipulation
- **Jupyter** - Interactive exploration

### Data & Markets
- **yfinance** - Market data fetching
- **nsepy** - NSE data (optional)

### Machine Learning
- **scikit-learn** - ML utilities
- **XGBoost** - Gradient boosting
- **TensorFlow/Keras** - Deep learning
- **hmmlearn** - Hidden Markov Models

### Visualization
- **Matplotlib** - Plotting
- **Seaborn** - Statistical visualization

### Financial Math
- **scipy** - Statistical functions
- **py_vollib** - Options pricing

---

## 🎯 Use Cases

### For Job Applications
✅ Demonstrates end-to-end quant workflow  
✅ Shows advanced technical skills  
✅ Proves domain knowledge  
✅ Professional code quality  

### For Learning
✅ Complete quantitative finance pipeline  
✅ Options pricing and Greeks  
✅ Statistical modeling (HMM)  
✅ ML for trading  

### For Portfolio
✅ GitHub-ready project  
✅ Comprehensive documentation  
✅ Professional visualizations  
✅ Interview talking points  

---

## 📈 Performance Highlights

*Results generated after running the pipeline*

- **Strategy**: EMA crossover with regime filtering
- **Risk Management**: 2% stop loss, 4% take profit
- **ML Enhancement**: Confidence-based trade filtering
- **Analysis**: 3-sigma outlier detection

---

## 🚀 Next Steps

1. **Run the System**: `python run_pipeline.py`
2. **Explore Results**: Check `results/` directory
3. **Understand Code**: Read `METHODOLOGY.md`
4. **Prepare Presentation**: Use `PRESENTATION_OUTLINE.md`
5. **Deploy to GitHub**: Follow `COMMANDS.md`

---

## 🤝 Contributing

This is a demonstration project for job applications. Feel free to:
- Fork and customize for your needs
- Use as learning material
- Extend with new features
- Share with attribution

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 👤 Author

**Quantitative Trading System**  
Built for ML Engineer / Quantitative Researcher role  
Demonstrates: Data Engineering, Feature Engineering, Statistical Modeling, Machine Learning, Risk Management

---

## 🌟 Star This Project

If you find this useful, please star the repository!

---

## 📞 Support

- **Documentation**: See `docs/` folder
- **Issues**: Check `SETUP_GUIDE.md` troubleshooting
- **Questions**: Review `METHODOLOGY.md`

---

**Ready to start?** → Open `START_HERE.md` for your first steps!

**Need help?** → Check `PROJECT_MAP.md` for navigation!

**Want to understand?** → Read `WHAT_WE_BUILT.md` for complete overview!

---

*Built with ❤️ for quantitative finance and machine learning*
