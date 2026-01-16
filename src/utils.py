"""
Utility functions for the quantitative trading system.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import joblib

from src.config import LOG_LEVEL, LOG_FORMAT


def setup_logger(name: str) -> logging.Logger:
    """Setup logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    
    return logger


def save_dataframe(df: pd.DataFrame, filepath: str, logger: Optional[logging.Logger] = None):
    """Save DataFrame to CSV with logging."""
    df.to_csv(filepath, index=False)
    if logger:
        logger.info(f"Saved data to {filepath} - Shape: {df.shape}")


def load_dataframe(filepath: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load DataFrame from CSV with logging."""
    df = pd.read_csv(filepath)
    if logger:
        logger.info(f"Loaded data from {filepath} - Shape: {df.shape}")
    return df


def save_model(model, filepath: str, logger: Optional[logging.Logger] = None):
    """Save model using joblib."""
    joblib.dump(model, filepath)
    if logger:
        logger.info(f"Saved model to {filepath}")


def load_model(filepath: str, logger: Optional[logging.Logger] = None):
    """Load model using joblib."""
    model = joblib.load(filepath)
    if logger:
        logger.info(f"Loaded model from {filepath}")
    return model


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate percentage returns."""
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns."""
    return np.log(prices / prices.shift(1))


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.065, periods: int = 252*78) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods: Number of periods per year (252 days * 78 5-min periods per day)
    """
    excess_returns = returns - (risk_free_rate / periods)
    if returns.std() == 0:
        return 0.0
    return np.sqrt(periods) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.065, periods: int = 252*78) -> float:
    """Calculate annualized Sortino ratio (downside deviation)."""
    excess_returns = returns - (risk_free_rate / periods)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    return np.sqrt(periods) * excess_returns.mean() / downside_returns.std()


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + equity_curve).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_calmar_ratio(returns: pd.Series, periods: int = 252*78) -> float:
    """Calculate Calmar ratio (return / max drawdown)."""
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (periods / len(returns)) - 1
    max_dd = abs(calculate_max_drawdown(returns))
    
    if max_dd == 0:
        return 0.0
    
    return annual_return / max_dd


def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate win rate (percentage of profitable trades)."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def detect_outliers(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using z-score method.
    
    Args:
        data: Series of values
        threshold: Z-score threshold (default 3 sigma)
    
    Returns:
        Boolean series indicating outliers
    """
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold


def normalize_features(df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
    """
    Normalize features.
    
    Args:
        df: DataFrame with features
        columns: Columns to normalize
        method: 'standard' (z-score) or 'minmax'
    """
    df_normalized = df.copy()
    
    for col in columns:
        if method == 'standard':
            df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()
        elif method == 'minmax':
            df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    return df_normalized


def create_lagged_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """Create lagged features for time series."""
    df_lagged = df.copy()
    
    for col in columns:
        for lag in lags:
            df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df_lagged


def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """Create rolling window features."""
    df_rolling = df.copy()
    
    for col in columns:
        for window in windows:
            df_rolling[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            df_rolling[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
    
    return df_rolling


def align_timestamps(dfs: List[pd.DataFrame], timestamp_col: str = 'timestamp') -> List[pd.DataFrame]:
    """Align multiple DataFrames on common timestamps."""
    # Find common timestamps
    common_timestamps = set(dfs[0][timestamp_col])
    for df in dfs[1:]:
        common_timestamps &= set(df[timestamp_col])
    
    # Filter each DataFrame
    aligned_dfs = []
    for df in dfs:
        aligned_df = df[df[timestamp_col].isin(common_timestamps)].copy()
        aligned_dfs.append(aligned_df)
    
    return aligned_dfs


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, symbol: str = '₹') -> str:
    """Format value as currency."""
    return f"{symbol}{value:,.2f}"


def print_performance_summary(metrics: Dict[str, float], logger: Optional[logging.Logger] = None):
    """Print formatted performance summary."""
    summary = "\n" + "="*60 + "\n"
    summary += "PERFORMANCE SUMMARY\n"
    summary += "="*60 + "\n"
    
    for key, value in metrics.items():
        if 'rate' in key.lower() or 'ratio' in key.lower():
            summary += f"{key:30s}: {value:.4f}\n"
        elif 'return' in key.lower() or 'drawdown' in key.lower():
            summary += f"{key:30s}: {format_percentage(value)}\n"
        else:
            summary += f"{key:30s}: {value}\n"
    
    summary += "="*60 + "\n"
    
    if logger:
        logger.info(summary)
    else:
        print(summary)
