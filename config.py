"""
Configuration parameters for the HeavyWeather portfolio allocation system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union
from datetime import datetime

# Asset class definitions
TRADITIONAL_ASSETS = {
    'SPY': 'S&P 500 ETF',
    'TLT': '20+ Year Treasury Bond ETF',
    'GLD': 'Gold ETF',
    'SHY': 'Short Treasury Bond ETF',
    'CASH': 'Cash Position'
}

FUTURES_ASSETS = {
    'USO': 'Crude Oil ETF',
    'VIXY': 'Volatility ETF',
    'UUP': 'USD Index ETF',
    'SPY': 'S&P 500 ETF',
    'GLD': 'Gold ETF',
    'CASH': 'Cash Position'
}

HIGH_VOL_ASSETS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'ARKK': 'ARK Innovation ETF',
    'TSLA': 'Tesla',
    'COIN': 'Coinbase',
    'CASH': 'Cash Position'
}

# Market regime definitions
REGIMES = {
    'gfc': {
        'name': 'Global Financial Crisis',
        'start': '2007-01-01',
        'end': '2009-12-31',
        'description': 'Stress test during financial crisis'
    },
    'covid': {
        'name': 'COVID Crash',
        'start': '2020-01-01',
        'end': '2020-06-30',
        'description': 'Volatility and futures testing during pandemic'
    },
    'tech_crash': {
        'name': 'Tech/Crypto Crash',
        'start': '2021-01-01',
        'end': '2023-12-31',
        'description': 'Speculation unwind period'
    },
    'training': {
        'name': 'General Training',
        'start': '2003-01-01',
        'end': '2019-12-31',
        'description': 'Normal market regimes for training'
    },
    'testing': {
        'name': 'Final Testing',
        'start': '2020-01-01',
        'end': '2023-12-31',
        'description': 'Out-of-sample performance evaluation'
    }
}

# Trading environment parameters
@dataclass
class EnvConfig:
    # State space parameters
    lookback_window: int = 30  # Increased from 20
    n_assets: int = 5  # Number of assets (excluding cash)
    transaction_cost: float = 0.0005  # Reduced from 0.001
    initial_balance: float = 100000.0  # Starting portfolio value
    
    # Technical indicators
    use_rsi: bool = True
    use_macd: bool = True
    use_volatility: bool = True
    use_vix: bool = True
    use_momentum: bool = True  # Added momentum indicator
    use_correlation: bool = True  # Added correlation indicator
    
    # Reward parameters
    risk_penalty: float = 0.3  # Reduced from 0.5
    reward_scaling: float = 1.0  # Kept the same
    
    # Trading constraints
    max_position_size: float = 0.35  # Reduced from 0.4
    min_position_size: float = 0.05  # Kept the same
    rebalance_frequency: str = 'daily'  # Kept the same
    
    # Crisis detection
    crisis_window: int = 10  # Added crisis detection window
    crisis_threshold: float = -0.015  # Added crisis threshold
    diversification_bonus: float = 0.15  # Increased from 0.1

# PPO training parameters
@dataclass
class PPOConfig:
    # Model architecture
    policy: str = 'MlpPolicy'
    net_arch: List[int] = field(default_factory=lambda: [256, 256, 128])  # Deeper network
    
    # Training parameters
    total_timesteps: int = 1_000_000  # Increased from 500_000
    learning_rate: float = 3e-4  # Reduced from 1e-3
    n_steps: int = 2048  # Increased from 1024
    batch_size: int = 256  # Increased from 128
    n_epochs: int = 10  # Increased from 5
    gamma: float = 0.99  # Kept the same
    gae_lambda: float = 0.95  # Kept the same
    clip_range: float = 0.2  # Kept the same
    ent_coef: float = 0.01  # Increased from 0.005 for better exploration
    
    # Evaluation parameters
    eval_freq: int = 10000  # Increased from 5000
    n_eval_episodes: int = 5  # Increased from 3
    
    # Logging
    tensorboard_log: str = 'logs/tensorboard'
    verbose: int = 1

# Performance metrics parameters
@dataclass
class MetricsConfig:
    # Return metrics
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    trading_days_per_year: int = 252
    
    # Statistical test parameters
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Rolling window parameters
    rolling_window: int = 252  # 1 year of trading days

# Create instances of config classes
env_config = EnvConfig()
ppo_config = PPOConfig()
metrics_config = MetricsConfig()

# File paths
MODEL_DIR = 'models/'
LOG_DIR = 'logs/'
DATA_DIR = 'data/'
NOTEBOOK_DIR = 'notebooks/'

# Create directories if they don't exist
import os
for directory in [MODEL_DIR, LOG_DIR, DATA_DIR, NOTEBOOK_DIR]:
    os.makedirs(directory, exist_ok=True) 