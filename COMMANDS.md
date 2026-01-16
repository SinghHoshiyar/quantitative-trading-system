# Command Reference Guide

## Quick Commands

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
python run_pipeline.py
```

### Run Individual Modules
```bash
# Data acquisition
python src/data_acquisition/fetch_data.py

# Data cleaning
python src/data_acquisition/clean_data.py

# Feature engineering
python src/feature_engineering/create_features.py

# Regime detection
python src/regime_detection/hmm_regimes.py

# Strategy backtest
python src/strategy/ema_strategy.py

# ML training
python src/ml_models/train_models.py

# Outlier analysis
python src/analysis/outlier_analysis.py
```

### Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook

# Open specific notebook
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## Development Commands

### Code Quality
```bash
# Format code
black src/

# Check style
flake8 src/

# Type checking
mypy src/
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_feature_engineering.py
```

### Git Commands
```bash
# Initialize repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Complete quantitative trading system"

# Add remote
git remote add origin <your-repo-url>

# Push to GitHub
git push -u origin main
```

## Data Management

### View Data
```bash
# View first few rows
python -c "import pandas as pd; print(pd.read_csv('data/features/nifty_features_5min.csv').head())"

# Check data shape
python -c "import pandas as pd; print(pd.read_csv('data/features/nifty_features_5min.csv').shape)"

# View columns
python -c "import pandas as pd; print(pd.read_csv('data/features/nifty_features_5min.csv').columns.tolist())"
```

### Clean Data
```bash
# Remove all generated data
rmdir /s /q data\raw data\processed data\features  # Windows
rm -rf data/raw data/processed data/features  # Linux/Mac

# Remove models
rmdir /s /q models  # Windows
rm -rf models  # Linux/Mac

# Remove results
rmdir /s /q results  # Windows
rm -rf results  # Linux/Mac
```

## Python Interactive Commands

### Load and Explore
```python
# Start Python
python

# Load data
import pandas as pd
df = pd.read_csv('data/features/nifty_features_5min.csv')

# Basic info
print(df.info())
print(df.describe())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# View specific columns
print(df[['close', 'regime', 'iv_atm_call']].head())
```

### Load Models
```python
# Load XGBoost model
import joblib
xgb_model = joblib.load('models/xgboost_model.pkl')

# Load LSTM model
from tensorflow import keras
lstm_model = keras.models.load_model('models/lstm_model.h5')

# Load HMM model
hmm_model = joblib.load('models/hmm_regime_model.pkl')
```

### Quick Analysis
```python
# Load results
trades = pd.read_csv('results/ema_strategy_trades.csv')

# Calculate metrics
print(f"Total trades: {len(trades)}")
print(f"Win rate: {(trades['return'] > 0).mean():.2%}")
print(f"Average return: {trades['return'].mean():.4f}")
print(f"Best trade: {trades['return'].max():.4f}")
print(f"Worst trade: {trades['return'].min():.4f}")
```

## Configuration Commands

### Edit Configuration
```bash
# Open config file
notepad src\config.py  # Windows
nano src/config.py  # Linux/Mac
vim src/config.py  # Vim users
```

### Common Configuration Changes
```python
# In src/config.py

# Change date range
START_DATE = datetime.now() - timedelta(days=180)  # 6 months

# Change capital
INITIAL_CAPITAL = 500000  # 5 Lakhs

# Change risk parameters
STOP_LOSS_PCT = 0.015  # 1.5%
TAKE_PROFIT_PCT = 0.03  # 3%

# Change ML threshold
ML_CONFIDENCE_THRESHOLD = 0.6  # More conservative
```

## Troubleshooting Commands

### Check Installation
```bash
# Check Python version
python --version

# Check pip version
pip --version

# List installed packages
pip list

# Check specific package
pip show pandas
```

### Fix Common Issues
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Reinstall package
pip uninstall pandas
pip install pandas

# Clear pip cache
pip cache purge

# Install from requirements again
pip install -r requirements.txt --force-reinstall
```

### System Information
```bash
# Check Python path
python -c "import sys; print(sys.executable)"

# Check installed packages location
python -c "import site; print(site.getsitepackages())"

# Check memory usage
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"
```

## Performance Commands

### Profiling
```python
# Profile code
python -m cProfile -o profile.stats run_pipeline.py

# View profile
python -m pstats profile.stats
# Then: sort cumtime, stats 20

# Memory profiling
python -m memory_profiler run_pipeline.py
```

### Timing
```bash
# Time complete pipeline
# Windows
powershell "Measure-Command {python run_pipeline.py}"

# Linux/Mac
time python run_pipeline.py
```

## Visualization Commands

### Generate Plots
```python
# In Python
import matplotlib.pyplot as plt
import pandas as pd

# Load and plot
df = pd.read_csv('results/ema_strategy_backtest.csv')
plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'], df['capital'])
plt.title('Equity Curve')
plt.savefig('results/custom_equity_curve.png', dpi=300)
plt.close()
```

### View Images
```bash
# Windows
start results\regime_visualization.png

# Linux
xdg-open results/regime_visualization.png

# Mac
open results/regime_visualization.png
```

## Export Commands

### Export Results
```python
# Export to Excel
import pandas as pd
trades = pd.read_csv('results/ema_strategy_trades.csv')
trades.to_excel('results/trades_report.xlsx', index=False)

# Export summary
summary = {
    'Total Trades': len(trades),
    'Win Rate': (trades['return'] > 0).mean(),
    'Avg Return': trades['return'].mean()
}
pd.DataFrame([summary]).to_csv('results/summary.csv', index=False)
```

### Create Archive
```bash
# Create zip of results
# Windows
powershell "Compress-Archive -Path results\* -DestinationPath results_backup.zip"

# Linux/Mac
zip -r results_backup.zip results/

# Create project archive (excluding data)
tar -czf project_backup.tar.gz --exclude='data' --exclude='*.pyc' .
```

## Documentation Commands

### Generate Documentation
```bash
# Install sphinx
pip install sphinx

# Initialize docs
sphinx-quickstart docs

# Build HTML docs
cd docs
make html
```

### View Documentation
```bash
# Start local server
python -m http.server 8000

# Open browser to http://localhost:8000
```

## Deployment Commands (Future)

### Docker (Optional)
```bash
# Build image
docker build -t quant-trading-system .

# Run container
docker run -v $(pwd)/data:/app/data quant-trading-system
```

### Requirements Export
```bash
# Export current environment
pip freeze > requirements_exact.txt

# Export minimal requirements
pipreqs . --force
```

## Useful Aliases (Optional)

### Add to .bashrc or .zshrc (Linux/Mac)
```bash
alias qts-run='python run_pipeline.py'
alias qts-fetch='python src/data_acquisition/fetch_data.py'
alias qts-strategy='python src/strategy/ema_strategy.py'
alias qts-ml='python src/ml_models/train_models.py'
alias qts-notebook='jupyter notebook notebooks/01_exploratory_analysis.ipynb'
```

### Add to PowerShell Profile (Windows)
```powershell
function qts-run { python run_pipeline.py }
function qts-fetch { python src/data_acquisition/fetch_data.py }
function qts-strategy { python src/strategy/ema_strategy.py }
function qts-ml { python src/ml_models/train_models.py }
```

## Emergency Commands

### Kill Stuck Process
```bash
# Windows
taskkill /F /IM python.exe

# Linux/Mac
pkill -9 python
```

### Reset Environment
```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rmdir /s /q venv  # Windows
rm -rf venv  # Linux/Mac

# Recreate
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Cheat Sheet

### Most Used Commands
```bash
# 1. Setup
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# 2. Run
python run_pipeline.py

# 3. Explore
jupyter notebook notebooks/01_exploratory_analysis.ipynb

# 4. Check results
dir results  # Windows
ls results  # Linux/Mac

# 5. Git
git add . && git commit -m "Update" && git push
```

### One-Liner Pipeline
```bash
# Complete setup and run
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt && python run_pipeline.py
```

---

**Pro Tip**: Create a `run.bat` (Windows) or `run.sh` (Linux/Mac) script with your most common commands for quick access!

**Example run.bat:**
```batch
@echo off
call venv\Scripts\activate
python run_pipeline.py
pause
```

**Example run.sh:**
```bash
#!/bin/bash
source venv/bin/activate
python run_pipeline.py
```
