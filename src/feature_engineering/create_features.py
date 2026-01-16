"""
Feature engineering module for creating technical indicators, Greeks, and derived features.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional

import sys
sys.path.append('.')

from src.config import *
from src.utils import setup_logger, load_dataframe, save_dataframe


logger = setup_logger(__name__)


class FeatureEngineer:
    """Create advanced features for trading."""
    
    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features."""
        logger.info("Creating all features")
        
        df = df.copy()
        
        # 1. Technical indicators
        df = self.create_ema_features(df)
        
        # 2. Options Greeks
        df = self.create_greeks_features(df)
        
        # 3. Derived features
        df = self.create_derived_features(df)
        
        # Drop rows with NaN (from rolling calculations)
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values")
        
        logger.info(f"Created features - Final shape: {df.shape}")
        return df
    
    def create_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create EMA indicators."""
        logger.info("Creating EMA features")
        
        df[f'ema_{EMA_SHORT}'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
        df[f'ema_{EMA_LONG}'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
        
        return df
    
    def create_greeks_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate options Greeks using Black-Scholes."""
        logger.info("Creating Greeks features")
        
        # Calculate time to expiry (assuming monthly expiry, ~25 days)
        df['days_to_expiry'] = 25
        df['time_to_expiry'] = df['days_to_expiry'] / 365.0
        
        # Calculate Greeks for ATM options
        for offset in [0]:  # Focus on ATM for main features
            suffix = f'_atm{offset:+d}' if offset != 0 else '_atm+0'
            
            # Call Greeks
            if f'call_iv{suffix}' in df.columns:
                greeks = self._calculate_greeks(
                    S=df['close'],
                    K=df['atm_strike'],
                    T=df['time_to_expiry'],
                    r=self.risk_free_rate,
                    sigma=df[f'call_iv{suffix}'],
                    option_type='call'
                )
                
                for greek, values in greeks.items():
                    df[f'{greek}_atm_call'] = values
            
            # Put Greeks
            if f'put_iv{suffix}' in df.columns:
                greeks = self._calculate_greeks(
                    S=df['close'],
                    K=df['atm_strike'],
                    T=df['time_to_expiry'],
                    r=self.risk_free_rate,
                    sigma=df[f'put_iv{suffix}'],
                    option_type='put'
                )
                
                for greek, values in greeks.items():
                    df[f'{greek}_atm_put'] = values
        
        return df
    
    def _calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate Black-Scholes Greeks."""
        # Avoid division by zero
        T = np.maximum(T, 1/365)
        sigma = np.maximum(sigma, 0.01)
        
        # d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        greeks = {}
        
        if option_type == 'call':
            # Delta
            greeks['delta'] = norm.cdf(d1)
            
            # Theta (per day)
            theta_annual = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                           - r * K * np.exp(-r * T) * norm.cdf(d2))
            greeks['theta'] = theta_annual / 365
            
        else:  # put
            # Delta
            greeks['delta'] = norm.cdf(d1) - 1
            
            # Theta (per day)
            theta_annual = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                           + r * K * np.exp(-r * T) * norm.cdf(-d2))
            greeks['theta'] = theta_annual / 365
        
        # Gamma (same for call and put)
        greeks['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega (same for call and put, per 1% change in IV)
        greeks['vega'] = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho (per 1% change in interest rate)
        if option_type == 'call':
            greeks['rho'] = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            greeks['rho'] = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return greeks
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from options and futures."""
        logger.info("Creating derived features")
        
        # 1. IV features
        if 'call_iv_atm+0' in df.columns and 'put_iv_atm+0' in df.columns:
            df['iv_atm_call'] = df['call_iv_atm+0']
            df['iv_atm_put'] = df['put_iv_atm+0']
            df['iv_skew'] = df['iv_atm_put'] - df['iv_atm_call']
            df['iv_avg'] = (df['iv_atm_call'] + df['iv_atm_put']) / 2
        
        # 2. Put-Call Ratios
        if 'call_oi_atm+0' in df.columns and 'put_oi_atm+0' in df.columns:
            df['pcr_oi'] = df['put_oi_atm+0'] / (df['call_oi_atm+0'] + 1)
        
        if 'call_volume_atm+0' in df.columns and 'put_volume_atm+0' in df.columns:
            df['pcr_volume'] = df['put_volume_atm+0'] / (df['call_volume_atm+0'] + 1)
        
        # 3. Futures basis (futures premium over spot)
        if 'futures_close' in df.columns:
            df['futures_basis'] = (df['futures_close'] - df['close']) / df['close']
        
        # 4. Delta neutrality
        if 'delta_atm_call' in df.columns and 'delta_atm_put' in df.columns:
            df['delta_skew'] = df['delta_atm_call'] + df['delta_atm_put']
        
        # 5. Gamma exposure
        if 'gamma_atm_call' in df.columns and 'gamma_atm_put' in df.columns:
            df['gamma_exposure'] = df['gamma_atm_call'] + df['gamma_atm_put']
        
        # 6. Volatility features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol'] = df['returns'].rolling(window=20).std() * np.sqrt(252*78)
        
        # 7. Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1)
        
        # 8. Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # 9. Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # 10. RSI
        df['rsi'] = self._calculate_rsi(df['close'], period=14)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi


def main():
    """Main function to create features."""
    logger.info("Starting feature engineering process")
    
    # Load merged data
    df = load_dataframe(MERGED_DATA_FILE, logger)
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    
    # Save features
    save_dataframe(features_df, FEATURES_FILE, logger)
    
    logger.info("Feature engineering completed successfully")
    logger.info(f"Total features created: {len(features_df.columns)}")


if __name__ == "__main__":
    main()
