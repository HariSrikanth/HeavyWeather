"""
Data loading and preprocessing module for the HeavyWeather portfolio allocation system.
Handles downloading, cleaning, and feature engineering of financial data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
import time
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from config import (
    TRADITIONAL_ASSETS,
    FUTURES_ASSETS,
    HIGH_VOL_ASSETS,
    REGIMES,
    DATA_DIR,
    env_config
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles downloading and preprocessing of financial data."""
    
    def __init__(
        self,
        asset_class: str = 'traditional',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        regime: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        """
        Initialize the data loader.
        
        Args:
            asset_class: One of 'traditional', 'futures', or 'high_vol'
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            regime: One of the predefined regimes in config.REGIMES
            max_retries: Maximum number of download retries
            retry_delay: Delay between retries in seconds
        """
        self.asset_class = asset_class.lower()
        self.asset_dict = self._get_asset_dict()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set date range
        if regime:
            if regime not in REGIMES:
                raise ValueError(f"Unknown regime: {regime}")
            self.start_date = REGIMES[regime]['start']
            self.end_date = REGIMES[regime]['end']
        else:
            self.start_date = start_date or '2003-01-01'
            self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Initialize data storage
        self.price_data = None
        self.returns_data = None
        self.features_data = None
        
    def _get_asset_dict(self) -> Dict[str, str]:
        """Get the appropriate asset dictionary based on asset class."""
        asset_dicts = {
            'traditional': TRADITIONAL_ASSETS,
            'futures': FUTURES_ASSETS,
            'high_vol': HIGH_VOL_ASSETS
        }
        if self.asset_class not in asset_dicts:
            raise ValueError(f"Unknown asset class: {self.asset_class}")
        return asset_dicts[self.asset_class]
    
    def _download_with_retry(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download data for a single ticker with retry logic.
        
        Args:
            ticker: Asset ticker symbol
            
        Returns:
            DataFrame with price data or None if download fails
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading {ticker} (attempt {attempt + 1}/{self.max_retries})")
                
                # Add a small delay between attempts
                if attempt > 0:
                    time.sleep(self.retry_delay)
                
                # Create a Ticker object
                ticker_obj = yf.Ticker(ticker)
                
                # Get historical data using history() method
                data = ticker_obj.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval="1d",
                    auto_adjust=True,
                    prepost=False,
                    actions=False
                )
                
                if data.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue
                
                # Convert index to timezone-naive datetime
                data.index = data.index.tz_localize(None)
                
                # Rename columns to multi-index
                data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
                logger.info(f"Successfully downloaded {ticker}")
                return data
                
            except Exception as e:
                logger.error(f"Error downloading {ticker} (attempt {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to download {ticker} after {self.max_retries} attempts")
                    return None
                continue
        
        return None
    
    def download_data(self) -> pd.DataFrame:
        """
        Download price data for all assets in the selected class.
        
        Returns:
            DataFrame with OHLCV data for all assets
        """
        logger.info(f"Downloading data for {self.asset_class} assets from {self.start_date} to {self.end_date}")
        
        # Download data for each asset
        dfs = []
        for ticker, description in self.asset_dict.items():
            if ticker == 'CASH':
                # Create synthetic cash returns (0% return)
                dates = pd.date_range(self.start_date, self.end_date, freq='B', tz=None)  # Explicitly timezone-naive
                cash_df = pd.DataFrame(
                    index=dates,
                    data={
                        'Open': 1.0,
                        'High': 1.0,
                        'Low': 1.0,
                        'Close': 1.0,
                        'Volume': 0
                    }
                )
                cash_df.columns = pd.MultiIndex.from_product([[ticker], cash_df.columns])
                dfs.append(cash_df)
            else:
                data = self._download_with_retry(ticker)
                if data is not None:
                    dfs.append(data)
        
        if not dfs:
            raise ValueError("No data downloaded for any assets")
        
        # Combine all dataframes
        self.price_data = pd.concat(dfs, axis=1)
        
        # Forward fill missing values (up to 5 days)
        self.price_data = self.price_data.ffill(limit=5)
        
        # Drop any remaining rows with NaN values
        self.price_data = self.price_data.dropna()
        
        logger.info(f"Downloaded data shape: {self.price_data.shape}")
        return self.price_data
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate returns for all assets.
        
        Returns:
            DataFrame with returns for all assets
        """
        if self.price_data is None:
            raise ValueError("Must download data before calculating returns")
        
        # Calculate returns for each asset
        returns = pd.DataFrame(index=self.price_data.index)
        
        for ticker in self.asset_dict.keys():
            if ticker == 'CASH':
                returns[ticker] = 0.0
            else:
                # Calculate log returns
                returns[ticker] = np.log(
                    self.price_data[ticker]['Close'] / 
                    self.price_data[ticker]['Close'].shift(1)
                )
        
        # Drop first row (NaN due to return calculation)
        returns = returns.iloc[1:]
        self.returns_data = returns
        
        logger.info(f"Calculated returns shape: {self.returns_data.shape}")
        return self.returns_data
    
    def calculate_features(self) -> pd.DataFrame:
        """
        Calculate technical indicators and features for each asset.
        
        Returns:
            DataFrame with features for all assets
        """
        if self.price_data is None:
            raise ValueError("Must download data before calculating features")
        
        features = pd.DataFrame(index=self.price_data.index)
        
        for ticker in self.asset_dict.keys():
            if ticker == 'CASH':
                continue
                
            close_prices = self.price_data[ticker]['Close']
            
            # Calculate technical indicators
            if env_config.use_rsi:
                rsi = RSIIndicator(close_prices, window=14)
                features[f'{ticker}_rsi'] = rsi.rsi()
            
            if env_config.use_macd:
                macd = MACD(close_prices)
                features[f'{ticker}_macd'] = macd.macd()
                features[f'{ticker}_macd_signal'] = macd.macd_signal()
                features[f'{ticker}_macd_diff'] = macd.macd_diff()
            
            if env_config.use_volatility:
                # Calculate rolling volatility (20-day)
                returns = np.log(close_prices / close_prices.shift(1))
                features[f'{ticker}_volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
                
                # Add Bollinger Bands
                bb = BollingerBands(close_prices, window=20, window_dev=2)
                features[f'{ticker}_bb_high'] = bb.bollinger_hband()
                features[f'{ticker}_bb_low'] = bb.bollinger_lband()
                features[f'{ticker}_bb_mid'] = bb.bollinger_mavg()
        
        # Add VIX if available and enabled
        if env_config.use_vix and 'VIXY' in self.asset_dict:
            vix_data = self.price_data['VIXY']['Close']
            features['vix'] = vix_data
        
        # Forward fill missing values (up to 5 days)
        features = features.fillna(method='ffill', limit=5)
        
        # Drop any remaining rows with NaN values
        features = features.dropna()
        
        self.features_data = features
        logger.info(f"Calculated features shape: {self.features_data.shape}")
        return self.features_data
    
    def save_data(self, prefix: str = '') -> None:
        """Save processed data to disk."""
        if self.price_data is None or self.returns_data is None or self.features_data is None:
            raise ValueError("Must process all data before saving")
        
        # Validate data alignment before saving
        if not (self.price_data.index.equals(self.returns_data.index) and 
                self.returns_data.index.equals(self.features_data.index)):
            logger.warning("Data indices are not aligned. Aligning data before saving...")
            # Find common index
            common_index = self.price_data.index.intersection(
                self.returns_data.index.intersection(self.features_data.index)
            )
            # Align all data to common index
            self.price_data = self.price_data.loc[common_index]
            self.returns_data = self.returns_data.loc[common_index]
            self.features_data = self.features_data.loc[common_index]
        
        # Create filename prefix
        prefix = f"{prefix}_{self.asset_class}" if prefix else self.asset_class
        
        # Ensure all indices are timezone-naive and sorted
        self.price_data.index = pd.to_datetime(self.price_data.index).tz_localize(None)
        self.returns_data.index = pd.to_datetime(self.returns_data.index).tz_localize(None)
        self.features_data.index = pd.to_datetime(self.features_data.index).tz_localize(None)
        
        # Sort all dataframes by date
        self.price_data = self.price_data.sort_index()
        self.returns_data = self.returns_data.sort_index()
        self.features_data = self.features_data.sort_index()
        
        # Save price data with MultiIndex columns
        # First, ensure the columns are properly named
        if isinstance(self.price_data.columns, pd.MultiIndex):
            # If already a MultiIndex, ensure it has the right names
            self.price_data.columns.names = ['asset', 'feature']
        else:
            # If not a MultiIndex, create one
            # Assuming columns are in order: Open, High, Low, Close, Volume for each asset
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            assets = list(self.asset_dict.keys())
            new_cols = pd.MultiIndex.from_product([assets, features], names=['asset', 'feature'])
            self.price_data.columns = new_cols
        
        # Save to CSV with explicit column names and index
        # For price data, flatten the MultiIndex columns
        price_data_to_save = self.price_data.copy()
        price_data_to_save.columns = [f"{asset}_{feature}" for asset, feature in price_data_to_save.columns]
        price_data_to_save.to_csv(
            os.path.join(DATA_DIR, f'{prefix}_prices.csv'),
            date_format='%Y-%m-%d'
        )
        
        # Save returns and features data
        self.returns_data.to_csv(
            os.path.join(DATA_DIR, f'{prefix}_returns.csv'),
            date_format='%Y-%m-%d'
        )
        self.features_data.to_csv(
            os.path.join(DATA_DIR, f'{prefix}_features.csv'),
            date_format='%Y-%m-%d'
        )
        
        logger.info(f"Saved data to {DATA_DIR} with prefix {prefix}")
        logger.info(f"Data shapes - Prices: {self.price_data.shape}, Returns: {self.returns_data.shape}, Features: {self.features_data.shape}")
        logger.info(f"Date ranges - Prices: {self.price_data.index[0]} to {self.price_data.index[-1]}")
        logger.info(f"Date ranges - Returns: {self.returns_data.index[0]} to {self.returns_data.index[-1]}")
        logger.info(f"Date ranges - Features: {self.features_data.index[0]} to {self.features_data.index[-1]}")
    
    def load_data(self, prefix: str = '') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed data from disk."""
        prefix = f"{prefix}_{self.asset_class}" if prefix else self.asset_class
        
        try:
            # Load price data
            logger.info(f"Loading price data from {os.path.join(DATA_DIR, f'{prefix}_prices.csv')}")
            price_data = pd.read_csv(
                os.path.join(DATA_DIR, f'{prefix}_prices.csv'),
                index_col=0,
                parse_dates=True,
                date_format='%Y-%m-%d'
            )
            logger.info(f"Raw price data columns: {price_data.columns.tolist()}")
            
            # Reconstruct MultiIndex columns from flattened names
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            assets = list(self.asset_dict.keys())
            
            # Create a mapping from flattened names to MultiIndex
            col_mapping = {}
            for asset in assets:
                for feature in features:
                    col_mapping[f"{asset}_{feature}"] = (asset, feature)
            
            # Rename columns to MultiIndex
            price_data.columns = pd.MultiIndex.from_tuples(
                [col_mapping[col] for col in price_data.columns],
                names=['asset', 'feature']
            )
            
            # Ensure index is timezone-naive and sorted
            price_data.index = pd.to_datetime(price_data.index, format='%Y-%m-%d').tz_localize(None)
            price_data = price_data.sort_index()
            
            logger.info(f"Processed price data columns: {price_data.columns.tolist()}")
            logger.info(f"Price data shape: {price_data.shape}")
            logger.info(f"Price data head:\n{price_data.head()}")
            self.price_data = price_data
            
            # Load returns data
            logger.info(f"Loading returns data from {os.path.join(DATA_DIR, f'{prefix}_returns.csv')}")
            self.returns_data = pd.read_csv(
                os.path.join(DATA_DIR, f'{prefix}_returns.csv'),
                index_col=0,
                parse_dates=True,
                date_format='%Y-%m-%d'
            )
            self.returns_data.index = pd.to_datetime(self.returns_data.index, format='%Y-%m-%d').tz_localize(None)
            self.returns_data = self.returns_data.sort_index()
            logger.info(f"Returns data shape: {self.returns_data.shape}")
            logger.info(f"Returns data columns: {self.returns_data.columns.tolist()}")
            logger.info(f"Returns data head:\n{self.returns_data.head()}")
            
            # Load features data
            logger.info(f"Loading features data from {os.path.join(DATA_DIR, f'{prefix}_features.csv')}")
            self.features_data = pd.read_csv(
                os.path.join(DATA_DIR, f'{prefix}_features.csv'),
                index_col=0,
                parse_dates=True,
                date_format='%Y-%m-%d'
            )
            self.features_data.index = pd.to_datetime(self.features_data.index, format='%Y-%m-%d').tz_localize(None)
            self.features_data = self.features_data.sort_index()
            logger.info(f"Features data shape: {self.features_data.shape}")
            logger.info(f"Features data columns: {self.features_data.columns.tolist()}")
            logger.info(f"Features data head:\n{self.features_data.head()}")
            
            # Log index information
            logger.info(f"Price data index: {price_data.index[0]} to {price_data.index[-1]}")
            logger.info(f"Returns data index: {self.returns_data.index[0]} to {self.returns_data.index[-1]}")
            logger.info(f"Features data index: {self.features_data.index[0]} to {self.features_data.index[-1]}")
            
            # Check index alignment and length
            logger.info(f"Index alignment: Prices == Returns: {self.price_data.index.equals(self.returns_data.index)}, Returns == Features: {self.returns_data.index.equals(self.features_data.index)}")
            logger.info(f"Lengths: Prices: {len(self.price_data)}, Returns: {len(self.returns_data)}, Features: {len(self.features_data)}")
            
            # Validate data alignment after loading
            if not (self.price_data.index.equals(self.returns_data.index) and 
                    self.returns_data.index.equals(self.features_data.index)):
                logger.warning("Loaded data indices are not aligned. Aligning data...")
                # Find common index
                common_index = self.price_data.index.intersection(
                    self.returns_data.index.intersection(self.features_data.index)
                )
                logger.info(f"Common index length: {len(common_index)}")
                if len(common_index) == 0:
                    raise ValueError("No common dates found between price, returns, and features data")
                
                # Align all data to common index
                self.price_data = self.price_data.loc[common_index]
                self.returns_data = self.returns_data.loc[common_index]
                self.features_data = self.features_data.loc[common_index]
            
            logger.info(f"Final data shapes after alignment - Prices: {self.price_data.shape}, Returns: {self.returns_data.shape}, Features: {self.features_data.shape}")
            return self.price_data, self.returns_data, self.features_data
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            logger.warning(f"No saved data found for prefix {prefix}")
            return None, None, None
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

def main():
    """Example usage of the DataLoader class."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and process financial data')
    parser.add_argument('--asset-class', type=str, default='traditional',
                      choices=['traditional', 'futures', 'high_vol'],
                      help='Asset class to download')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--regime', type=str, choices=list(REGIMES.keys()),
                      help='Market regime to download')
    parser.add_argument('--save', action='store_true',
                      help='Save processed data to disk')
    
    args = parser.parse_args()
    
    # Initialize data loader
    loader = DataLoader(
        asset_class=args.asset_class,
        start_date=args.start_date,
        end_date=args.end_date,
        regime=args.regime
    )
    
    # Download and process data
    loader.download_data()
    loader.calculate_returns()
    loader.calculate_features()
    
    if args.save:
        loader.save_data()

if __name__ == '__main__':
    main() 