# Installation Guide

## Prerequisites

### System Requirements
- Python 3.9 or higher
- 8GB RAM (16GB recommended)
- 5GB free disk space
- Windows/Linux/macOS

### Required Software
- Python 3.9+
- pip (Python package manager)
- Git (optional, for cloning)

## Installation Steps

### 1. Clone or Download Repository

```bash
git clone https://github.com/SinghHoshiyar/quantitative-trading-system
cd quantitative-trading-system
```

Or download and extract the ZIP file.

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, tensorflow; print('Installation successful')"
```

## Running the System

### Complete Pipeline

```bash
python run_pipeline.py
```

This executes all steps:
1. Data acquisition
2. Data cleaning
3. Feature engineering
4. Regime detection
5. Strategy backtest
6. ML model training
7. Outlier analysis

**Expected Runtime**: 5-15 minutes

### Individual Modules

Run specific steps:

```bash
# Data acquisition
python -m src.data_acquisition.fetch_data

# Data cleaning
python -m src.data_acquisition.clean_data

# Feature engineering
python -m src.feature_engineering.create_features

# Regime detection
python -m src.regime_detection.hmm_regimes

# Strategy backtest
python -m src.strategy.ema_strategy

# ML training
python -m src.ml_models.train_models

# Outlier analysis
python -m src.analysis.outlier_analysis
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## Configuration

Edit `src/config.py` to customize:

### Data Configuration
```python
START_DATE = datetime.now() - timedelta(days=365)
END_DATE = datetime.now()
TIMEFRAME = '5min'
```

### Trading Parameters
```python
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs
POSITION_SIZE = 0.02       # 2% per trade
STOP_LOSS_PCT = 0.02       # 2%
TAKE_PROFIT_PCT = 0.04     # 4%
```

### ML Parameters
```python
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    ...
}

LSTM_PARAMS = {
    'sequence_length': 20,
    'lstm_units': [64, 32],
    'epochs': 50,
    ...
}
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Ensure virtual environment is activated
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Memory Errors

**Problem**: `MemoryError` or system slowdown

**Solution**: Reduce data size in `src/config.py`:
```python
START_DATE = datetime.now() - timedelta(days=180)  # 6 months instead of 1 year
```

### TensorFlow Warnings

**Problem**: TensorFlow GPU warnings

**Solution**: These are usually harmless. To suppress:
```bash
# Windows
set TF_CPP_MIN_LOG_LEVEL=2

# Linux/macOS
export TF_CPP_MIN_LOG_LEVEL=2
```

### Data Fetching Issues

**Problem**: yfinance data fetch fails

**Solution**: The system automatically generates synthetic data for demonstration. This is intentional and does not affect the methodology demonstration.

## Output Files

After running the pipeline, check:

### Data Directory
```
data/
├── raw/              # Original market data
├── processed/        # Cleaned data
└── features/         # Engineered features
```

### Models Directory
```
models/
├── hmm_regime_model.pkl
├── xgboost_model.pkl
├── lstm_model.h5
└── feature_scaler.pkl
```

### Results Directory
```
results/
├── regime_visualization.png
├── ema_strategy_results.png
├── feature_importance.png
├── outlier_analysis.png
├── ema_strategy_backtest.csv
├── ema_strategy_trades.csv
└── outlier_trades.csv
```

## Performance Optimization

### Faster Execution
- Reduce date range (6 months instead of 1 year)
- Reduce LSTM epochs (25 instead of 50)
- Reduce XGBoost trees (100 instead of 200)

### Better Results
- Increase data size (2-3 years)
- Add more features
- Tune hyperparameters
- Use ensemble methods

## Next Steps

1. Review generated results in `results/` directory
2. Explore data using Jupyter notebook
3. Customize configuration for experiments
4. Read `METHODOLOGY.md` for technical details
5. Check `RESULTS.md` for analysis

## Support

For issues:
1. Check error messages carefully
2. Verify Python version: `python --version`
3. Verify package versions: `pip list`
4. Ensure virtual environment is activated
5. Check available disk space and memory
