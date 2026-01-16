"""
Configuration file for the quantitative trading system.
"""
from datetime import datetime, timedelta

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Date range for data fetching (1 year of data)
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365)

# Data frequency
TIMEFRAME = '5min'

# Symbols
SPOT_SYMBOL = '^NSEI'  # NIFTY 50 Index
FUTURES_SYMBOL = 'NIFTY'
OPTIONS_SYMBOL = 'NIFTY'

# Options configuration
ATM_RANGE = 2  # ATM ± 2 strikes
STRIKE_INTERVAL = 50  # NIFTY strike interval

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# EMA periods
EMA_SHORT = 5
EMA_LONG = 15

# Risk-free rate for options pricing
RISK_FREE_RATE = 0.065  # 6.5%

# Options Greeks to calculate
GREEKS = ['delta', 'gamma', 'theta', 'vega', 'rho']

# ============================================================================
# REGIME DETECTION CONFIGURATION
# ============================================================================

# HMM parameters
N_REGIMES = 3  # Uptrend, Sideways, Downtrend
HMM_COVARIANCE_TYPE = 'full'
HMM_N_ITER = 100
RANDOM_STATE = 42

# Regime labels
REGIME_LABELS = {
    0: 'Downtrend',
    1: 'Sideways',
    2: 'Uptrend'
}

# Features for regime detection (options-based only)
REGIME_FEATURES = [
    'iv_atm_call',
    'iv_atm_put',
    'pcr_oi',
    'pcr_volume',
    'futures_basis',
    'delta_skew',
    'gamma_exposure'
]

# ============================================================================
# TRADING STRATEGY CONFIGURATION
# ============================================================================

# Strategy parameters
INITIAL_CAPITAL = 1000000  # 10 Lakhs
POSITION_SIZE = 0.02  # 2% per trade
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.04  # 4% take profit

# Regime filter
ALLOWED_LONG_REGIMES = [2]  # Only uptrend
ALLOWED_SHORT_REGIMES = [0]  # Only downtrend
NO_TRADE_REGIMES = [1]  # Sideways

# ============================================================================
# MACHINE LEARNING CONFIGURATION
# ============================================================================

# Train-test split
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# LSTM parameters
LSTM_PARAMS = {
    'sequence_length': 20,
    'lstm_units': [64, 32],
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32
}

# ML confidence threshold
ML_CONFIDENCE_THRESHOLD = 0.5

# Features for ML (excluding target and identifiers)
ML_FEATURE_COLUMNS = [
    # Price features
    'close', 'volume', 'ema_5', 'ema_15',
    # Options features
    'iv_atm_call', 'iv_atm_put', 'pcr_oi', 'pcr_volume',
    # Greeks
    'delta_atm_call', 'gamma_atm_call', 'theta_atm_call', 'vega_atm_call',
    # Derived features
    'futures_basis', 'delta_skew', 'gamma_exposure',
    # Regime
    'regime'
]

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Outlier detection
OUTLIER_THRESHOLD = 3  # 3-sigma

# Performance metrics
METRICS = [
    'total_return',
    'sharpe_ratio',
    'sortino_ratio',
    'calmar_ratio',
    'max_drawdown',
    'win_rate',
    'avg_trade_duration'
]

# ============================================================================
# FILE PATHS
# ============================================================================

# Data paths
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
FEATURES_DATA_DIR = 'data/features'

# Model paths
MODELS_DIR = 'models'

# Results paths
RESULTS_DIR = 'results'

# File names
SPOT_DATA_FILE = f'{RAW_DATA_DIR}/nifty_spot_5min.csv'
FUTURES_DATA_FILE = f'{RAW_DATA_DIR}/nifty_futures_5min.csv'
OPTIONS_DATA_FILE = f'{RAW_DATA_DIR}/nifty_options_5min.csv'
MERGED_DATA_FILE = f'{PROCESSED_DATA_DIR}/nifty_merged_5min.csv'
FEATURES_FILE = f'{FEATURES_DATA_DIR}/nifty_features_5min.csv'
REGIME_FILE = f'{FEATURES_DATA_DIR}/nifty_with_regimes.csv'

# Model files
HMM_MODEL_FILE = f'{MODELS_DIR}/hmm_regime_model.pkl'
XGBOOST_MODEL_FILE = f'{MODELS_DIR}/xgboost_model.pkl'
LSTM_MODEL_FILE = f'{MODELS_DIR}/lstm_model.h5'
SCALER_FILE = f'{MODELS_DIR}/feature_scaler.pkl'

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
