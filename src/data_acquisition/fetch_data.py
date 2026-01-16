"""
Data fetching module for NIFTY spot, futures, and options data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import yfinance as yf
from nsepy import get_history
from nsepy.derivatives import get_expiry_date

import sys
sys.path.append('.')

from src.config import *
from src.utils import setup_logger, save_dataframe


logger = setup_logger(__name__)


class DataFetcher:
    """Fetch market data for NIFTY 50."""
    
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_spot_data(self) -> pd.DataFrame:
        """
        Fetch NIFTY 50 spot data (OHLCV).
        Using yfinance for intraday data.
        """
        logger.info(f"Fetching NIFTY spot data from {self.start_date} to {self.end_date}")
        
        try:
            # Fetch data using yfinance
            ticker = yf.Ticker(SPOT_SYMBOL)
            df = ticker.history(start=self.start_date, end=self.end_date, interval='5m')
            
            if df.empty:
                logger.warning("No spot data fetched, generating synthetic data")
                return self._generate_synthetic_spot_data()
            
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            df = df.reset_index()
            df = df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'})
            
            # Select required columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Fetched {len(df)} spot data records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching spot data: {e}")
            logger.info("Generating synthetic spot data as fallback")
            return self._generate_synthetic_spot_data()
    
    def fetch_futures_data(self) -> pd.DataFrame:
        """
        Fetch NIFTY futures data with rollover handling.
        """
        logger.info(f"Fetching NIFTY futures data from {self.start_date} to {self.end_date}")
        
        try:
            # For demonstration, we'll generate synthetic futures data
            # In production, use NSE API or data provider
            logger.warning("Generating synthetic futures data (use NSE API in production)")
            return self._generate_synthetic_futures_data()
            
        except Exception as e:
            logger.error(f"Error fetching futures data: {e}")
            return self._generate_synthetic_futures_data()
    
    def fetch_options_data(self) -> pd.DataFrame:
        """
        Fetch NIFTY options chain data (ATM ± 2 strikes).
        """
        logger.info(f"Fetching NIFTY options data from {self.start_date} to {self.end_date}")
        
        try:
            # For demonstration, we'll generate synthetic options data
            # In production, use NSE API or data provider
            logger.warning("Generating synthetic options data (use NSE API in production)")
            return self._generate_synthetic_options_data()
            
        except Exception as e:
            logger.error(f"Error fetching options data: {e}")
            return self._generate_synthetic_options_data()
    
    def _generate_synthetic_spot_data(self) -> pd.DataFrame:
        """Generate synthetic spot data for testing."""
        logger.info("Generating synthetic spot data")
        
        # Generate 5-minute timestamps for 1 year
        timestamps = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='5min'
        )
        
        # Filter trading hours (9:15 AM to 3:30 PM IST)
        timestamps = timestamps[
            (timestamps.hour >= 9) & (timestamps.hour < 16) &
            ~((timestamps.hour == 9) & (timestamps.minute < 15)) &
            ~((timestamps.hour == 15) & (timestamps.minute > 30))
        ]
        
        n = len(timestamps)
        
        # Generate realistic price movement
        np.random.seed(42)
        base_price = 18000
        returns = np.random.normal(0.0001, 0.01, n)
        prices = base_price * (1 + returns).cumprod()
        
        # Generate OHLCV
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n))),
            'close': prices * (1 + np.random.normal(0, 0.001, n)),
            'volume': np.random.randint(100000, 1000000, n)
        })
        
        logger.info(f"Generated {len(df)} synthetic spot records")
        return df
    
    def _generate_synthetic_futures_data(self) -> pd.DataFrame:
        """Generate synthetic futures data."""
        logger.info("Generating synthetic futures data")
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='5min'
        )
        
        timestamps = timestamps[
            (timestamps.hour >= 9) & (timestamps.hour < 16) &
            ~((timestamps.hour == 9) & (timestamps.minute < 15)) &
            ~((timestamps.hour == 15) & (timestamps.minute > 30))
        ]
        
        n = len(timestamps)
        
        # Generate futures prices (slightly above spot)
        np.random.seed(43)
        base_price = 18050  # Futures premium
        returns = np.random.normal(0.0001, 0.01, n)
        prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'futures_open': prices,
            'futures_high': prices * (1 + np.abs(np.random.normal(0, 0.002, n))),
            'futures_low': prices * (1 - np.abs(np.random.normal(0, 0.002, n))),
            'futures_close': prices * (1 + np.random.normal(0, 0.001, n)),
            'futures_volume': np.random.randint(50000, 500000, n),
            'open_interest': np.random.randint(1000000, 5000000, n)
        })
        
        logger.info(f"Generated {len(df)} synthetic futures records")
        return df
    
    def _generate_synthetic_options_data(self) -> pd.DataFrame:
        """Generate synthetic options data (ATM ± 2 strikes)."""
        logger.info("Generating synthetic options data")
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='5min'
        )
        
        timestamps = timestamps[
            (timestamps.hour >= 9) & (timestamps.hour < 16) &
            ~((timestamps.hour == 9) & (timestamps.minute < 15)) &
            ~((timestamps.hour == 15) & (timestamps.minute > 30))
        ]
        
        n = len(timestamps)
        
        # Generate ATM strike (round to nearest 50)
        base_price = 18000
        atm_strike = (base_price // STRIKE_INTERVAL) * STRIKE_INTERVAL
        
        # Generate strikes (ATM ± 2)
        strikes = [atm_strike + i * STRIKE_INTERVAL for i in range(-ATM_RANGE, ATM_RANGE + 1)]
        
        np.random.seed(44)
        
        # Generate options data for each strike
        options_data = []
        
        for strike in strikes:
            # Call options
            call_df = pd.DataFrame({
                'timestamp': timestamps,
                'strike': strike,
                'option_type': 'CE',
                'ltp': np.maximum(0, (base_price - strike) + np.random.normal(50, 20, n)),
                'iv': np.random.normal(0.15, 0.03, n),
                'volume': np.random.randint(1000, 50000, n),
                'oi': np.random.randint(100000, 1000000, n)
            })
            
            # Put options
            put_df = pd.DataFrame({
                'timestamp': timestamps,
                'strike': strike,
                'option_type': 'PE',
                'ltp': np.maximum(0, (strike - base_price) + np.random.normal(50, 20, n)),
                'iv': np.random.normal(0.16, 0.03, n),
                'volume': np.random.randint(1000, 50000, n),
                'oi': np.random.randint(100000, 1000000, n)
            })
            
            options_data.append(call_df)
            options_data.append(put_df)
        
        df = pd.concat(options_data, ignore_index=True)
        
        logger.info(f"Generated {len(df)} synthetic options records")
        return df


def main():
    """Main function to fetch all data."""
    logger.info("Starting data fetching process")
    
    # Initialize fetcher
    fetcher = DataFetcher(START_DATE, END_DATE)
    
    # Fetch spot data
    spot_df = fetcher.fetch_spot_data()
    save_dataframe(spot_df, SPOT_DATA_FILE, logger)
    
    # Fetch futures data
    futures_df = fetcher.fetch_futures_data()
    save_dataframe(futures_df, FUTURES_DATA_FILE, logger)
    
    # Fetch options data
    options_df = fetcher.fetch_options_data()
    save_dataframe(options_df, OPTIONS_DATA_FILE, logger)
    
    logger.info("Data fetching completed successfully")


if __name__ == "__main__":
    main()
