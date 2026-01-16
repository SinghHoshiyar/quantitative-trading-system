# Project Navigation Map

## 🗺️ Quick Navigation Guide

### 🚀 Getting Started (Start Here!)

1. **First Time?** → Read `WHAT_WE_BUILT.md`
2. **Want to Run?** → Follow `QUICKSTART.md`
3. **Need Details?** → Check `SETUP_GUIDE.md`
4. **Understanding Code?** → Review `METHODOLOGY.md`

---

## 📂 File Organization

### 📚 Documentation Files (Root Directory)

| File | Purpose | When to Use |
|------|---------|-------------|
| `README.md` | Project overview | First introduction |
| `QUICKSTART.md` | 5-minute setup | Want to run immediately |
| `SETUP_GUIDE.md` | Detailed installation | Troubleshooting setup |
| `METHODOLOGY.md` | Technical details | Understanding algorithms |
| `COMMANDS.md` | Command reference | Need specific commands |
| `PROJECT_SUMMARY.md` | Complete summary | Preparing presentation |
| `COMPLETION_CHECKLIST.md` | Pre-submission tasks | Before interview |
| `PRESENTATION_OUTLINE.md` | PPT structure | Creating slides |
| `WHAT_WE_BUILT.md` | Project explanation | Understanding scope |
| `PROJECT_MAP.md` | This file | Navigation help |
| `requirements.txt` | Dependencies | Installation |
| `LICENSE` | Legal | Open source info |
| `.gitignore` | Git rules | Version control |

### 💻 Source Code (`src/`)

```
src/
├── config.py                      # ⚙️ All configuration settings
├── utils.py                       # 🛠️ Helper functions
├── __init__.py                    # 📦 Package initialization
│
├── data_acquisition/              # 📥 Data fetching & cleaning
│   ├── __init__.py
│   ├── fetch_data.py             # Fetch spot, futures, options
│   └── clean_data.py             # Clean and merge data
│
├── feature_engineering/           # 🔧 Feature creation
│   ├── __init__.py
│   └── create_features.py        # Technical, Greeks, derived
│
├── regime_detection/              # 🎯 Market state detection
│   ├── __init__.py
│   └── hmm_regimes.py            # HMM implementation
│
├── strategy/                      # 📈 Trading strategy
│   ├── __init__.py
│   └── ema_strategy.py           # EMA + regime filtering
│
├── ml_models/                     # 🤖 Machine learning
│   ├── __init__.py
│   └── train_models.py           # XGBoost + LSTM
│
└── analysis/                      # 📊 Performance analysis
    ├── __init__.py
    └── outlier_analysis.py       # High-performance trades
```

### 📊 Data Directories

```
data/
├── raw/                          # Original market data
│   ├── nifty_spot_5min.csv
│   ├── nifty_futures_5min.csv
│   └── nifty_options_5min.csv
│
├── processed/                    # Cleaned & merged
│   └── nifty_merged_5min.csv
│
└── features/                     # Engineered features
    ├── nifty_features_5min.csv
    └── nifty_with_regimes.csv
```

### 🤖 Models Directory

```
models/
├── hmm_regime_model.pkl         # Regime detector
├── xgboost_model.pkl            # Trade classifier
├── lstm_model.h5                # Sequential learner
└── feature_scaler.pkl           # Feature normalizer
```

### 📈 Results Directory

```
results/
├── regime_visualization.png      # Regime analysis
├── ema_strategy_results.png     # Strategy performance
├── feature_importance.png       # ML feature analysis
├── outlier_analysis.png         # Exceptional trades
├── ema_strategy_backtest.csv    # Full backtest data
├── ema_strategy_trades.csv      # Individual trades
└── outlier_trades.csv           # High-performance trades
```

### 📓 Notebooks Directory

```
notebooks/
└── 01_exploratory_analysis.ipynb  # Interactive exploration
```

---

## 🔄 Workflow Map

### Complete Pipeline Flow

```
1. Data Acquisition
   ↓
   fetch_data.py → Fetches spot, futures, options
   ↓
   clean_data.py → Cleans and merges
   ↓
2. Feature Engineering
   ↓
   create_features.py → Creates 50+ features
   ↓
3. Regime Detection
   ↓
   hmm_regimes.py → Classifies market states
   ↓
4. Trading Strategy
   ↓
   ema_strategy.py → Backtests strategy
   ↓
5. Machine Learning
   ↓
   train_models.py → Trains XGBoost + LSTM
   ↓
6. Analysis
   ↓
   outlier_analysis.py → Analyzes exceptional trades
   ↓
7. Results
   ↓
   Visualizations + Reports + Insights
```

### Run Everything

```bash
python run_pipeline.py
```

This executes all steps automatically!

---

## 🎯 Use Case Navigation

### "I want to..."

#### Run the System
→ `QUICKSTART.md` → `python run_pipeline.py`

#### Understand the Code
→ `METHODOLOGY.md` → Review `src/` modules

#### Customize Settings
→ `src/config.py` → Edit parameters

#### Explore Data
→ `notebooks/01_exploratory_analysis.ipynb`

#### View Results
→ `results/` directory → Open PNG files

#### Prepare Presentation
→ `PRESENTATION_OUTLINE.md` → Create PowerPoint

#### Troubleshoot Issues
→ `SETUP_GUIDE.md` → Troubleshooting section

#### Learn Commands
→ `COMMANDS.md` → Find specific command

#### Check Before Submission
→ `COMPLETION_CHECKLIST.md` → Verify everything

#### Understand Project Scope
→ `PROJECT_SUMMARY.md` → Complete overview

#### Deploy to GitHub
→ `COMMANDS.md` → Git commands section

---

## 📖 Reading Order

### For First-Time Users

1. `WHAT_WE_BUILT.md` - Understand what you have
2. `QUICKSTART.md` - Get it running
3. `README.md` - Project overview
4. Explore `results/` - See outputs
5. `METHODOLOGY.md` - Understand how it works

### For Interview Preparation

1. `PROJECT_SUMMARY.md` - Complete overview
2. `METHODOLOGY.md` - Technical deep dive
3. `PRESENTATION_OUTLINE.md` - Structure slides
4. `COMPLETION_CHECKLIST.md` - Verify readiness
5. Review all code in `src/`

### For Customization

1. `src/config.py` - Change settings
2. `METHODOLOGY.md` - Understand algorithms
3. Relevant module in `src/` - Modify code
4. `COMMANDS.md` - Run specific parts
5. Test changes

---

## 🔍 Finding Specific Information

### Configuration
**Where?** `src/config.py`
**What?** All settings (dates, capital, risk, ML params)

### Data Fetching
**Where?** `src/data_acquisition/fetch_data.py`
**What?** How data is acquired

### Feature Engineering
**Where?** `src/feature_engineering/create_features.py`
**What?** How features are created

### Greeks Calculation
**Where?** `src/feature_engineering/create_features.py`
**Method?** `_calculate_greeks()`

### Regime Detection
**Where?** `src/regime_detection/hmm_regimes.py`
**What?** HMM implementation

### Trading Logic
**Where?** `src/strategy/ema_strategy.py`
**What?** Entry/exit rules, risk management

### ML Models
**Where?** `src/ml_models/train_models.py`
**What?** XGBoost and LSTM training

### Performance Metrics
**Where?** `src/utils.py`
**What?** Sharpe, Sortino, Calmar calculations

### Outlier Analysis
**Where?** `src/analysis/outlier_analysis.py`
**What?** 3-sigma detection and pattern analysis

---

## 🎨 Visualization Map

### Generated Visualizations

1. **Regime Visualization**
   - File: `results/regime_visualization.png`
   - Shows: Price with regime colors, timeline, distribution
   - Created by: `src/regime_detection/hmm_regimes.py`

2. **Strategy Results**
   - File: `results/ema_strategy_results.png`
   - Shows: Signals, equity curve, drawdown
   - Created by: `src/strategy/ema_strategy.py`

3. **Feature Importance**
   - File: `results/feature_importance.png`
   - Shows: Top 20 features from XGBoost
   - Created by: `src/ml_models/train_models.py`

4. **Outlier Analysis**
   - File: `results/outlier_analysis.png`
   - Shows: Return distribution, outliers, patterns
   - Created by: `src/analysis/outlier_analysis.py`

---

## 🛠️ Modification Guide

### Want to Change...

#### Date Range
→ `src/config.py` → `START_DATE`, `END_DATE`

#### Capital
→ `src/config.py` → `INITIAL_CAPITAL`

#### Risk Parameters
→ `src/config.py` → `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`

#### EMA Periods
→ `src/config.py` → `EMA_SHORT`, `EMA_LONG`

#### ML Hyperparameters
→ `src/config.py` → `XGBOOST_PARAMS`, `LSTM_PARAMS`

#### Add New Feature
→ `src/feature_engineering/create_features.py` → `create_derived_features()`

#### Change Strategy Logic
→ `src/strategy/ema_strategy.py` → `generate_signals()`

#### Add New Model
→ `src/ml_models/train_models.py` → Add new training method

---

## 🚨 Troubleshooting Map

### Issue: Installation fails
→ `SETUP_GUIDE.md` → Troubleshooting section

### Issue: Import errors
→ Check virtual environment is activated
→ `pip install -r requirements.txt`

### Issue: No data fetched
→ Normal! System uses synthetic data
→ See `src/data_acquisition/fetch_data.py`

### Issue: Out of memory
→ `src/config.py` → Reduce date range

### Issue: Pipeline fails
→ Check logs in console
→ Run modules individually (see `COMMANDS.md`)

### Issue: Results look wrong
→ Review `METHODOLOGY.md`
→ Check configuration in `src/config.py`

---

## 📞 Quick Reference

### Most Important Files

1. **To Run**: `run_pipeline.py`
2. **To Configure**: `src/config.py`
3. **To Understand**: `METHODOLOGY.md`
4. **To Present**: `PRESENTATION_OUTLINE.md`
5. **To Troubleshoot**: `SETUP_GUIDE.md`

### Most Important Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run
python run_pipeline.py

# Explore
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### Most Important Concepts

1. **Regime Detection**: Market state classification
2. **Options Greeks**: Delta, Gamma, Theta, Vega, Rho
3. **Risk Management**: Stop loss, take profit, position sizing
4. **ML Enhancement**: Trade quality improvement
5. **Outlier Analysis**: Understanding exceptional trades

---

## 🎓 Learning Path

### Beginner Level
1. Run the pipeline
2. View results
3. Explore notebook
4. Read README

### Intermediate Level
1. Understand each module
2. Modify configuration
3. Customize features
4. Review methodology

### Advanced Level
1. Modify algorithms
2. Add new strategies
3. Implement new models
4. Extend analysis

---

## ✅ Success Checklist

- [ ] Read `WHAT_WE_BUILT.md`
- [ ] Run `python run_pipeline.py`
- [ ] Check `results/` directory
- [ ] Open Jupyter notebook
- [ ] Review all visualizations
- [ ] Understand `METHODOLOGY.md`
- [ ] Prepare presentation
- [ ] Complete `COMPLETION_CHECKLIST.md`
- [ ] Ready for interview!

---

**Remember**: This map is your guide. Bookmark it and refer back whenever you need direction!

**Pro Tip**: Keep this file open in a separate window while working on the project for quick reference.
