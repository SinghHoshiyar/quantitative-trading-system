# Quick Start Guide

## Get Running in 5 Minutes

### Step 1: Install Dependencies (2 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Run the Complete Pipeline (3 minutes)

```bash
python run_pipeline.py
```

That's it! The system will:
1. ✅ Fetch market data
2. ✅ Clean and merge data
3. ✅ Engineer features
4. ✅ Detect market regimes
5. ✅ Run trading strategy
6. ✅ Train ML models
7. ✅ Analyze high-performance trades

### Step 3: View Results

Check the `results/` directory for:
- 📊 `regime_visualization.png` - Market regime analysis
- 📈 `ema_strategy_results.png` - Trading performance
- 🎯 `feature_importance.png` - ML feature analysis
- 💎 `outlier_analysis.png` - Exceptional trade patterns

### Step 4: Explore Data

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## What You Get

### Performance Metrics
- Total return percentage
- Sharpe ratio (risk-adjusted returns)
- Win rate and trade statistics
- Maximum drawdown
- ML model accuracy

### Visualizations
- Price charts with regime colors
- Equity curve
- Feature importance
- Trade distribution
- Outlier analysis

### Models
- HMM regime detector
- XGBoost classifier
- LSTM neural network

## Next Steps

1. **Review Results**: Check `results/` folder
2. **Analyze Trades**: Open Jupyter notebook
3. **Tune Parameters**: Edit `src/config.py`
4. **Prepare Presentation**: Use `PRESENTATION_OUTLINE.md`

## Troubleshooting

**Problem**: Package installation fails
**Solution**: Try `pip install --upgrade pip` first

**Problem**: Out of memory
**Solution**: Reduce date range in `src/config.py`

**Problem**: No data fetched
**Solution**: System uses synthetic data automatically (this is normal for demo)

## File Structure

```
├── data/              # Market data
├── models/            # Trained models
├── results/           # Visualizations and reports
├── src/               # Source code
│   ├── data_acquisition/
│   ├── feature_engineering/
│   ├── regime_detection/
│   ├── strategy/
│   ├── ml_models/
│   └── analysis/
├── notebooks/         # Jupyter notebooks
└── run_pipeline.py    # Main entry point
```

## Key Configuration

Edit `src/config.py` to customize:

```python
# Date range
START_DATE = datetime.now() - timedelta(days=365)
END_DATE = datetime.now()

# Capital
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs

# Risk management
STOP_LOSS_PCT = 0.02  # 2%
TAKE_PROFIT_PCT = 0.04  # 4%

# ML parameters
ML_CONFIDENCE_THRESHOLD = 0.5
```

## Support

For detailed documentation:
- `README.md` - Project overview
- `SETUP_GUIDE.md` - Detailed installation
- `METHODOLOGY.md` - Technical details
- `PRESENTATION_OUTLINE.md` - Presentation guide

## Success Checklist

- [ ] Virtual environment activated
- [ ] All packages installed
- [ ] Pipeline runs without errors
- [ ] Results generated in `results/`
- [ ] Models saved in `models/`
- [ ] Jupyter notebook opens
- [ ] Ready to analyze and present!

---

**Estimated Total Time**: 5-15 minutes (depending on system speed)

**Expected Output**: Complete quantitative trading system with backtested results, trained ML models, and professional visualizations ready for presentation.
