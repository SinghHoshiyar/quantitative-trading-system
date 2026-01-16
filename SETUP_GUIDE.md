# Setup Guide - Quantitative Trading System

## Prerequisites

### System Requirements
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space
- Internet connection for data fetching

### Required Software
- Python 3.9+
- pip (Python package manager)
- Git (for version control)
- Jupyter Notebook (for exploration)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd quantitative-trading-system
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install TA-Lib (Optional but Recommended)

TA-Lib requires special installation:

**Windows:**
1. Download TA-Lib wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Install: `pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl`

**Linux:**
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

**Mac:**
```bash
brew install ta-lib
pip install TA-Lib
```

If TA-Lib installation fails, the system will work without it (using pandas-ta as fallback).

### 5. Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, tensorflow; print('All packages installed successfully!')"
```

## Directory Structure Setup

The required directories are created automatically when you run the pipeline. However, you can create them manually:

```bash
mkdir -p data/raw data/processed data/features
mkdir -p models results notebooks tests
```

## Configuration

### 1. Review Configuration

Open `src/config.py` and review the settings:

- **Date Range**: Adjust `START_DATE` and `END_DATE` if needed
- **Capital**: Modify `INITIAL_CAPITAL` (default: ₹10,00,000)
- **Risk Parameters**: Adjust `STOP_LOSS_PCT` and `TAKE_PROFIT_PCT`
- **ML Parameters**: Tune model hyperparameters if needed

### 2. Data Sources

The system uses:
- **yfinance** for NIFTY spot data
- **Synthetic data generation** for futures and options (for demonstration)

For production use, replace with:
- NSE API
- Commercial data providers (Bloomberg, Reuters)
- Broker APIs

## Running the System

### Option 1: Complete Pipeline (Recommended for First Run)

```bash
python run_pipeline.py
```

This runs all steps sequentially:
1. Data fetching
2. Data cleaning
3. Feature engineering
4. Regime detection
5. Strategy backtesting
6. ML model training
7. Outlier analysis

**Expected Runtime**: 5-15 minutes (depending on system)

### Option 2: Individual Modules

Run each step separately:

```bash
# Step 1: Fetch data
python src/data_acquisition/fetch_data.py

# Step 2: Clean data
python src/data_acquisition/clean_data.py

# Step 3: Create features
python src/feature_engineering/create_features.py

# Step 4: Detect regimes
python src/regime_detection/hmm_regimes.py

# Step 5: Run strategy
python src/strategy/ema_strategy.py

# Step 6: Train ML models
python src/ml_models/train_models.py

# Step 7: Analyze outliers
python src/analysis/outlier_analysis.py
```

### Option 3: Jupyter Notebook Exploration

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## Output Files

After running the pipeline, you'll find:

### Data Files
- `data/raw/` - Raw market data
- `data/processed/` - Cleaned and merged data
- `data/features/` - Engineered features

### Models
- `models/hmm_regime_model.pkl` - HMM regime detector
- `models/xgboost_model.pkl` - XGBoost classifier
- `models/lstm_model.h5` - LSTM model
- `models/feature_scaler.pkl` - Feature scaler

### Results
- `results/regime_visualization.png` - Regime analysis
- `results/ema_strategy_results.png` - Strategy performance
- `results/feature_importance.png` - Feature importance
- `results/outlier_analysis.png` - Outlier trade analysis
- `results/ema_strategy_backtest.csv` - Detailed backtest results
- `results/ema_strategy_trades.csv` - Individual trade records

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'xxx'
```
**Solution**: Ensure virtual environment is activated and all packages are installed:
```bash
pip install -r requirements.txt
```

**2. Memory Errors**
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce data size in `config.py` or increase system RAM

**3. TensorFlow/Keras Warnings**
```
WARNING: TensorFlow...
```
**Solution**: These are usually harmless. To suppress:
```bash
export TF_CPP_MIN_LOG_LEVEL=2  # Linux/Mac
set TF_CPP_MIN_LOG_LEVEL=2     # Windows
```

**4. Data Fetching Issues**
```
Error fetching data from yfinance
```
**Solution**: The system automatically falls back to synthetic data generation. For real data, check internet connection or use alternative data sources.

**5. HMM Convergence Warnings**
```
ConvergenceWarning: HMM did not converge
```
**Solution**: Increase `HMM_N_ITER` in `config.py` or adjust features used for regime detection.

### Getting Help

1. Check the logs in console output
2. Review error messages carefully
3. Ensure all prerequisites are installed
4. Verify Python version: `python --version`
5. Check package versions: `pip list`

## Performance Optimization

### For Faster Execution

1. **Reduce Data Size**: In `config.py`, reduce date range
2. **Use Fewer Features**: Comment out complex features in feature engineering
3. **Reduce ML Epochs**: Lower `LSTM_PARAMS['epochs']` in config
4. **Use Fewer Trees**: Reduce `XGBOOST_PARAMS['n_estimators']`

### For Better Results

1. **More Data**: Increase date range (2-3 years)
2. **More Features**: Add technical indicators
3. **Hyperparameter Tuning**: Use GridSearchCV for ML models
4. **Ensemble Methods**: Combine multiple models

## Next Steps

After successful setup:

1. **Review Results**: Check `results/` directory for visualizations
2. **Analyze Performance**: Open Jupyter notebook for detailed analysis
3. **Tune Parameters**: Adjust configuration based on results
4. **Extend System**: Add new features or strategies
5. **Prepare Presentation**: Use results to create PowerPoint

## Production Deployment Considerations

For live trading (NOT included in this project):

1. **Real-time Data**: Integrate with live data feeds
2. **Order Execution**: Connect to broker APIs
3. **Risk Management**: Implement position limits and circuit breakers
4. **Monitoring**: Add logging and alerting
5. **Backtesting**: Extensive historical testing
6. **Paper Trading**: Test with simulated orders first
7. **Compliance**: Ensure regulatory compliance

## Support

For issues or questions:
1. Review documentation
2. Check code comments
3. Examine log outputs
4. Debug step by step

## License

MIT License - See LICENSE file for details
