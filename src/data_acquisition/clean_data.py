"""
Data cleaning module for handling missing data, bad ticks, and alignment.
"""
import pandas as pd
import numpy as np
from typing import Optional

import sys
sys.path.append('.')

from src.config import *
from src.utils import setup_logger, load_dataframe, save_dataframe


logger = setup_logger(__name__)


class DataCleaner:
    """Clean and prepare market data."""
    
    def __init__(self):
        pass
    
    def clean_spot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean spot data."""
        logger.info(f"Cleaning spot data - Initial shape: {df.shape}")
        
        df = df.copy()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Handle missing values
        df = self._handle_missing_candles(df)
        
        # Remove bad ticks (outliers)
        df = self._remove_bad_ticks(df)
        
        # Ensure positive prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = df[col].clip(lower=0)
        
        # Ensure high >= low
        df['high'] = df[['high', 'low']].max(axis=1)
        df['low'] = df[['high', 'low']].min(axis=1)
        
        logger.info(f"Cleaned spot data - Final shape: {df.shape}")
        return df
    
    def clean_futures_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean futures data and handle rollover."""
        logger.info(f"Cleaning futures data - Initial shape: {df.shape}")
        
        df = df.copy()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Handle missing values
        df = self._handle_missing_candles(df, price_col='futures_close')
        
        # Remove bad ticks
        df = self._remove_bad_ticks(df, price_col='futures_close')
        
        # Ensure positive values
        price_cols = ['futures_open', 'futures_high', 'futures_low', 'futures_close']
        for col in price_cols:
            df[col] = df[col].clip(lower=0)
        
        logger.info(f"Cleaned futures data - Final shape: {df.shape}")
        return df
    
    def clean_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean options data."""
        logger.info(f"Cleaning options data - Initial shape: {df.shape}")
        
        df = df.copy()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['timestamp', 'strike', 'option_type']).reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp', 'strike', 'option_type'], keep='first')
        
        # Handle missing values - updated method
        df['ltp'] = df.groupby(['strike', 'option_type'])['ltp'].ffill()
        df['iv'] = df.groupby(['strike', 'option_type'])['iv'].ffill()
        
        # Remove bad IVs (negative or too high)
        df = df[(df['iv'] > 0) & (df['iv'] < 2)]
        
        # Ensure positive prices
        df['ltp'] = df['ltp'].clip(lower=0)
        
        logger.info(f"Cleaned options data - Final shape: {df.shape}")
        return df
    
    def _handle_missing_candles(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Fill missing candles using forward fill."""
        # Create complete timestamp range
        full_range = pd.date_range(
            start=df['timestamp'].min(),
            end=df['timestamp'].max(),
            freq='5min'
        )
        
        # Filter trading hours
        full_range = full_range[
            (full_range.hour >= 9) & (full_range.hour < 16) &
            ~((full_range.hour == 9) & (full_range.minute < 15)) &
            ~((full_range.hour == 15) & (full_range.minute > 30))
        ]
        
        # Reindex and forward fill
        df = df.set_index('timestamp').reindex(full_range)
        df = df.ffill().bfill()  # Updated method
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        
        return df
    
    def _remove_bad_ticks(self, df: pd.DataFrame, price_col: str = 'close', threshold: float = 5.0) -> pd.DataFrame:
        """Remove outlier ticks using z-score method."""
        # Calculate returns
        returns = df[price_col].pct_change()
        
        # Calculate z-scores
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        
        # Mark outliers
        outliers = z_scores > threshold
        
        # Replace outliers with interpolated values
        df.loc[outliers, price_col] = np.nan
        df[price_col] = df[price_col].interpolate(method='linear')
        
        logger.info(f"Removed {outliers.sum()} bad ticks")
        return df
    
    def merge_data(self, spot_df: pd.DataFrame, futures_df: pd.DataFrame, 
                   options_df: pd.DataFrame) -> pd.DataFrame:
        """Merge spot, futures, and options data on timestamp."""
        logger.info("Merging spot, futures, and options data")
        
        # Merge spot and futures
        merged = pd.merge(spot_df, futures_df, on='timestamp', how='inner')
        
        # Calculate ATM strike dynamically
        merged['atm_strike'] = (merged['close'] // STRIKE_INTERVAL) * STRIKE_INTERVAL
        
        # Pivot options data to wide format
        options_pivot = self._pivot_options_data(options_df)
        
        # Merge with options
        merged = pd.merge(merged, options_pivot, on='timestamp', how='left')
        
        # Forward fill missing options data - updated method
        options_cols = [col for col in merged.columns if col.startswith(('call_', 'put_'))]
        merged[options_cols] = merged[options_cols].ffill()
        
        logger.info(f"Merged data shape: {merged.shape}")
        return merged
    
    def _pivot_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot options data to wide format (ATM ± 2)."""
        # Filter for ATM ± 2 strikes relative to spot
        # For simplicity, we'll take the middle 5 strikes
        strikes = sorted(df['strike'].unique())
        mid_idx = len(strikes) // 2
        selected_strikes = strikes[max(0, mid_idx-2):min(len(strikes), mid_idx+3)]
        
        df_filtered = df[df['strike'].isin(selected_strikes)].copy()
        
        # Create pivot for calls and puts
        pivoted_data = []
        
        for timestamp in df_filtered['timestamp'].unique():
            row_data = {'timestamp': timestamp}
            
            df_ts = df_filtered[df_filtered['timestamp'] == timestamp]
            
            # Get ATM strike (middle strike)
            atm_strike = selected_strikes[len(selected_strikes) // 2]
            
            for i, strike in enumerate(selected_strikes):
                offset = i - 2  # -2, -1, 0, 1, 2
                
                # Call data
                call_data = df_ts[(df_ts['strike'] == strike) & (df_ts['option_type'] == 'CE')]
                if not call_data.empty:
                    row_data[f'call_ltp_atm{offset:+d}'] = call_data['ltp'].values[0]
                    row_data[f'call_iv_atm{offset:+d}'] = call_data['iv'].values[0]
                    row_data[f'call_volume_atm{offset:+d}'] = call_data['volume'].values[0]
                    row_data[f'call_oi_atm{offset:+d}'] = call_data['oi'].values[0]
                
                # Put data
                put_data = df_ts[(df_ts['strike'] == strike) & (df_ts['option_type'] == 'PE')]
                if not put_data.empty:
                    row_data[f'put_ltp_atm{offset:+d}'] = put_data['ltp'].values[0]
                    row_data[f'put_iv_atm{offset:+d}'] = put_data['iv'].values[0]
                    row_data[f'put_volume_atm{offset:+d}'] = put_data['volume'].values[0]
                    row_data[f'put_oi_atm{offset:+d}'] = put_data['oi'].values[0]
            
            pivoted_data.append(row_data)
        
        return pd.DataFrame(pivoted_data)


def main():
    """Main function to clean and merge data."""
    logger.info("Starting data cleaning process")
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Load raw data
    spot_df = load_dataframe(SPOT_DATA_FILE, logger)
    futures_df = load_dataframe(FUTURES_DATA_FILE, logger)
    options_df = load_dataframe(OPTIONS_DATA_FILE, logger)
    
    # Clean data
    spot_clean = cleaner.clean_spot_data(spot_df)
    futures_clean = cleaner.clean_futures_data(futures_df)
    options_clean = cleaner.clean_options_data(options_df)
    
    # Merge data
    merged_df = cleaner.merge_data(spot_clean, futures_clean, options_clean)
    
    # Save merged data
    save_dataframe(merged_df, MERGED_DATA_FILE, logger)
    
    logger.info("Data cleaning completed successfully")


if __name__ == "__main__":
    main()
