# Quick Reference Card

## Essential Commands

### Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac
pip install -r requirements.txt
```

### Run
```bash
python run_pipeline.py         # Complete pipeline
jupyter notebook notebooks/    # Explore data
```

### Individual Modules
```bash
python -m src.data_acquisition.fetch_data
python -m src.data_acquisition.clean_data
python -m src.feature_engineering.create_features
python -m src.regime_detection.hmm_regimes
python -m src.strategy.ema_strategy
python -m src.ml_models.train_models
python -m src.analysis.outlier_analysis
```

## File Locations

### Documentation
- `README.md` - Start here
- `INSTALLATION.md` - Setup guide
- `METHODOLOGY.md` - Technical details
- `RESULTS.md` - Analysis
- `PRESENTATION_GUIDE.md` - Presentation structure
- `TECHNICAL_APPENDIX.md` - Formulas
- `PROJECT_CHECKLIST.md` - Verification

### Configuration
- `src/config.py` - All parameters

### Results
- `results/*.png` - Visualizations
- `results/*.csv` - Data files
- `models/*.pkl` - Trained models

## Key Parameters (src/config.py)

```python
# Data
START_DATE = datetime.now() - timedelta(days=365)
TIMEFRAME = '5min'

# Trading
INITIAL_CAPITAL = 1000000      # ₹10 Lakhs
POSITION_SIZE = 0.02           # 2% per trade
STOP_LOSS_PCT = 0.02           # 2%
TAKE_PROFIT_PCT = 0.04         # 4%

# Regime
N_REGIMES = 3                  # Uptrend, Sideways, Downtrend

# ML
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05
}

LSTM_PARAMS = {
    'sequence_length': 20,
    'lstm_units': [64, 32],
    'epochs': 50
}
```

## Expected Outputs

### Data Files
```
data/raw/nifty_spot_5min.csv
data/raw/nifty_futures_5min.csv
data/raw/nifty_options_5min.csv
data/processed/nifty_merged_5min.csv
data/features/nifty_features_5min.csv
data/features/nifty_with_regimes.csv
```

### Models
```
models/hmm_regime_model.pkl
models/xgboost_model.pkl
models/lstm_model.h5
models/feature_scaler.pkl
```

### Results
```
results/regime_visualization.png
results/ema_strategy_results.png
results/feature_importance.png
results/outlier_analysis.png
results/ema_strategy_backtest.csv
results/ema_strategy_trades.csv
results/outlier_trades.csv
```

## Pipeline Steps

1. **Data Acquisition** - Fetch spot, futures, options
2. **Data Cleaning** - Clean and merge data
3. **Feature Engineering** - Create 50+ features
4. **Regime Detection** - Train HMM, classify states
5. **Strategy Backtest** - EMA + regime filter
6. **ML Training** - XGBoost + LSTM
7. **Outlier Analysis** - Pattern discovery

**Runtime**: 5-15 minutes

## Key Metrics

### Strategy Performance
- Total Trades: ~2,000
- Win Rate: ~47%
- Sharpe Ratio: Calculated
- Max Drawdown: Tracked

### ML Performance
- XGBoost Accuracy: ~53%
- LSTM Accuracy: ~52%
- Feature Importance: Extracted

### Outlier Analysis
- Outlier Trades: ~3% of total
- Pattern Recognition: By regime, time, IV

## Troubleshooting

### Import Errors
```bash
# Activate venv
venv\Scripts\activate
# Reinstall
pip install -r requirements.txt
```

### Memory Issues
```python
# In src/config.py
START_DATE = datetime.now() - timedelta(days=180)  # Reduce to 6 months
```

### TensorFlow Warnings
```bash
set TF_CPP_MIN_LOG_LEVEL=2     # Windows
export TF_CPP_MIN_LOG_LEVEL=2  # Linux/Mac
```

## Documentation Quick Links

| Need | Document |
|------|----------|
| Setup | INSTALLATION.md |
| Technical | METHODOLOGY.md |
| Results | RESULTS.md |
| Formulas | TECHNICAL_APPENDIX.md |
| Presentation | PRESENTATION_GUIDE.md |
| Checklist | PROJECT_CHECKLIST.md |
| Navigation | DOCUMENTATION_INDEX.md |

## Presentation Structure

1. Introduction (3 slides)
2. Data Engineering (4 slides)
3. Feature Engineering (4 slides)
4. Regime Detection (4 slides)
5. Trading Strategy (5 slides)
6. Machine Learning (5 slides)
7. Outlier Analysis (3 slides)
8. Conclusion (3 slides)

**Total**: 25-30 slides  
**Duration**: 20-25 minutes

## Key Talking Points

### Technical
- Black-Scholes Greeks implementation
- HMM for regime detection
- XGBoost + LSTM for ML
- 3-sigma outlier detection

### Business
- Regime awareness reduces false signals
- Options data provides forward-looking intelligence
- ML improves trade quality
- Risk management is critical

### Limitations
- Synthetic data for demonstration
- No transaction costs modeled
- Single-asset focus
- Assumes pattern persistence

## Interview Questions

### Technical
- Why HMM? → Captures hidden states, handles non-stationarity
- Prevent overfitting? → Time-series split, regularization, early stopping
- Why ~53% accuracy? → Markets are noisy, focus on quality improvement

### Business
- Profitable live? → Needs real data, execution, costs
- Capital needed? → ₹10 lakhs minimum
- Retrain frequency? → Monthly HMM, weekly ML

## Success Checklist

- [ ] Pipeline runs successfully
- [ ] All outputs generated
- [ ] Documentation reviewed
- [ ] Presentation created
- [ ] Practiced 3+ times
- [ ] GitHub updated
- [ ] Ready for interview



## Final Steps

1. Run: `python run_pipeline.py`
2. Verify: Check `results/` and `models/`
3. Review: Read all documentation
4. Create: PowerPoint presentation
5. Practice: 3+ times
6. Push: To GitHub
7. Prepare: For interview

---

**Quick Start**: README.md → INSTALLATION.md → Run Pipeline → RESULTS.md

**Deep Dive**: METHODOLOGY.md → TECHNICAL_APPENDIX.md

---

**You're ready! 🚀**
