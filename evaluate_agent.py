"""
Evaluation script for the trained PPO agent.
Analyzes performance and compares against baseline strategies.
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import empyrical as ep
from scipy import stats
import torch
import matplotlib.dates as mdates

from data_loader import DataLoader
from trading_env import PortfolioTradingEnv
from config import (
    REGIMES,
    MODEL_DIR,
    LOG_DIR,
    metrics_config,
    env_config
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioEvaluator:
    """Evaluates portfolio performance and compares against baselines."""
    
    # Define crisis periods with more specific dates
    CRISIS_PERIODS = {
        '2008_financial': {
            'name': 'Global Financial Crisis',
            'start': '2008-09-01',  # Lehman Brothers bankruptcy
            'end': '2009-03-31',    # Market bottom
            'description': 'Peak of the financial crisis'
        },
        '2020_covid': {
            'name': 'COVID-19 Crash',
            'start': '2020-02-15',  # Market peak before crash
            'end': '2020-04-30',    # Initial recovery
            'description': 'COVID-19 market crash and recovery'
        },
        'normal_period': {
            'name': 'Normal Market Period',
            'start': '2017-01-01',  # Recent normal period
            'end': '2019-12-31',    # Before COVID
            'description': 'Recent normal market conditions'
        }
    }
    
    def __init__(
        self,
        model_path: str,
        asset_class: str,
        regime: str = 'testing',
        no_normalize: bool = False
    ):
        """
        Initialize the evaluator with enhanced performance tracking.
        
        Args:
            model_path: Path to the trained model
            asset_class: Asset class being traded
            regime: Market regime to evaluate on
            no_normalize: Whether to disable observation/reward normalization
        """
        self.model_path = model_path
        self.asset_class = asset_class
        self.regime = regime
        
        # Check for best model
        best_model_path = os.path.join(model_path, 'best_model.zip')
        final_model_path = os.path.join(model_path, 'final_model.zip')
        
        if os.path.exists(best_model_path):
            logger.info("Using best model from training (best_model.zip)")
            model_to_load = best_model_path
        else:
            logger.info("Best model not found, using final model (final_model.zip)")
            model_to_load = final_model_path
        
        # Load model first to get configuration
        try:
            self.model = PPO.load(model_to_load)
            logger.info(f"Successfully loaded model from {model_to_load}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Get lookback window from model's observation space
        model_obs_shape = self.model.observation_space.shape
        if len(model_obs_shape) == 2:  # (lookback, features)
            self.lookback_window = model_obs_shape[0]
            logger.info(f"Using lookback window from model: {self.lookback_window}")
        else:
            raise ValueError(f"Unexpected model observation space shape: {model_obs_shape}")
        
        # Load data for environment creation
        loader = DataLoader(asset_class=self.asset_class, regime=self.regime)
        price_data, returns_data, features_data = loader.load_data()
        if price_data is None or returns_data is None or features_data is None:
            raise RuntimeError(
                "Data loading failed after download. Check if the data files are being saved and paths are correct.\n"
                f"price_data: {type(price_data)}, returns_data: {type(returns_data)}, features_data: {type(features_data)}"
            )
        
        # Get the assets used in training (from returns data)
        self.assets = returns_data.columns.tolist()
        logger.info(f"Assets used in training: {self.assets}")
        
        # Filter price data to match training assets
        if isinstance(price_data.columns, pd.MultiIndex):
            # Handle multi-index columns (OHLCV data)
            price_data = price_data.loc[:, (self.assets, 'Close')]
            price_data.columns = self.assets  # Flatten to single index
        else:
            # Handle single index columns
            price_data = price_data[self.assets]
        
        # Filter features data to match training assets
        feature_columns = []
        for asset in self.assets:
            asset_features = [col for col in features_data.columns if col.startswith(f"{asset}_")]
            feature_columns.extend(asset_features)
        
        # Add any global features (like VIX)
        global_features = [col for col in features_data.columns if not any(col.startswith(f"{asset}_") for asset in self.assets)]
        feature_columns.extend(global_features)
        
        features_data = features_data[feature_columns]
        
        logger.info(f"Filtered data shapes:")
        logger.info(f"Price data: {price_data.shape}")
        logger.info(f"Returns data: {returns_data.shape}")
        logger.info(f"Features data: {features_data.shape}")
        
        self.price_data = price_data
        self.returns_data = returns_data
        self.features_data = features_data
        
        # Add performance tracking
        self.performance_metrics = {
            'normal': {},
            'crisis': {},
            'overall': {}
        }
        self.regime_periods = []
        self.position_concentration = []
        self.correlation_metrics = []
        
        # Set up VecNormalize
        if not no_normalize:
            try:
                vec_normalize_path = os.path.join(model_path, 'vec_normalize.pkl')
                if os.path.exists(vec_normalize_path):
                    self.vec_normalize = VecNormalize.load(
                        vec_normalize_path,
                        venv=DummyVecEnv([self.make_env])
                    )
                    logger.info("Successfully loaded VecNormalize stats")
                else:
                    logger.warning("VecNormalize file not found, creating new instance")
                    self.vec_normalize = VecNormalize(
                        DummyVecEnv([self.make_env]),
                        norm_obs=True,
                        norm_reward=True,
                        clip_obs=10.0,
                        training=False
                    )
            except Exception as e:
                logger.error(f"Error setting up VecNormalize: {str(e)}")
                logger.info("Falling back to no normalization")
                self.vec_normalize = None
        else:
            self.vec_normalize = None
        
        # Initialize results storage
        self.results = {}
    
    def _setup_environment(self) -> None:
        """Set up the evaluation environment."""
        # Environment is already created in __init__
        pass
    
    def make_env(self):
        """Create and return a trading environment with proper configuration."""
        env = PortfolioTradingEnv(
            price_data=self.price_data,
            returns_data=self.returns_data,
            features_data=self.features_data,
            lookback_window=self.lookback_window,
            # Use env_config attributes with dot notation
            transaction_cost=env_config.transaction_cost,
            risk_penalty=env_config.risk_penalty,
            max_position_size=env_config.max_position_size,
            min_position_size=env_config.min_position_size,
            crisis_threshold=env_config.crisis_threshold,
            diversification_bonus=env_config.diversification_bonus,
            initial_balance=env_config.initial_balance,
            reward_scaling=env_config.reward_scaling
        )
        # Debug logging for environment configuration
        logger.info("Environment configuration:")
        logger.info(f"Transaction cost: {env.transaction_cost}")
        logger.info(f"Risk penalty: {env.risk_penalty}")
        logger.info(f"Max position size: {env.max_position_size}")
        logger.info(f"Min position size: {env.min_position_size}")
        logger.info(f"Crisis threshold: {env.crisis_threshold}")
        logger.info(f"Diversification bonus: {env.diversification_bonus}")
        logger.info(f"Current observation space shape: {env.observation_space.shape}")
        return env
    
    def evaluate_agent(self, n_episodes: int = 1, random_policy: bool = False) -> Dict:
        """
        Evaluate the trained agent with enhanced performance tracking.
        """
        logger.info(f"Evaluating agent on {self.regime} regime (random_policy={random_policy})")
        
        # Run episodes
        portfolio_values = []
        portfolio_weights = []
        actions = []
        regime_returns = {
            'normal': [],
            'crisis': []
        }
        
        # Track regime information
        regime_history = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_values = []
            episode_weights = []
            episode_actions = []
            episode_regimes = []
            step_idx = 0
            
            # Debug logging for initial state
            logger.info(f"Initial portfolio weights: {info.get('portfolio_weights', None)}")
            logger.info(f"Initial regime: {info.get('regime', 'normal')}")
            
            while not done:
                if random_policy:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                
                # Debug logging for actions
                if step_idx % 100 == 0:  # Log every 100 steps
                    logger.info(f"Step {step_idx} - Action: {action}")
                    logger.info(f"Action sum: {np.sum(action)}")
                    logger.info(f"Action min/max: {np.min(action)}/{np.max(action)}")
                
                episode_actions.append(action)
                obs, reward, done, _, info = self.env.step(action)
                
                # Track portfolio metrics
                episode_values.append(info['portfolio_value'])
                episode_weights.append(info['portfolio_weights'])
                current_regime = info.get('regime', 'normal')
                episode_regimes.append(current_regime)
                
                # Debug logging for regime changes
                if step_idx > 0 and episode_regimes[-1] != episode_regimes[-2]:
                    logger.info(f"Regime change at step {step_idx}: {episode_regimes[-2]} -> {episode_regimes[-1]}")
                
                # Calculate and track metrics
                if step_idx > 0:
                    daily_return = (info['portfolio_value'] / episode_values[-2]) - 1
                    regime_returns[current_regime].append(daily_return)
                    
                    # Track position concentration
                    hhi = np.sum(info['portfolio_weights'] ** 2)
                    self.position_concentration.append({
                        'step': step_idx,
                        'hhi': hhi,
                        'regime': current_regime,
                        'weights': info['portfolio_weights'].tolist()  # Store actual weights
                    })
                    
                    # Track correlations
                    if len(episode_values) >= 30:
                        portfolio_returns = pd.Series([(v2/v1 - 1) for v1, v2 in zip(episode_values[-30:-1], episode_values[-29:])])
                        asset_returns = self.returns_data.iloc[step_idx-29:step_idx+1]
                        correlations = asset_returns.apply(lambda x: x.corr(portfolio_returns))
                        self.correlation_metrics.append({
                            'step': step_idx,
                            'avg_correlation': correlations.abs().mean(),
                            'regime': current_regime
                        })
                
                step_idx += 1
            
            # Store episode results with more detailed logging
            episode_index = self.returns_data.index[self.env.lookback_window-1:self.env.lookback_window-1+len(episode_values)]
            portfolio_values.append(pd.Series(episode_values, index=episode_index))
            portfolio_weights.append(pd.DataFrame(episode_weights, index=episode_index, columns=self.env.assets))
            actions.append(np.array(episode_actions))
            
            # Log final episode statistics
            logger.info(f"\nEpisode {episode + 1} Summary:")
            logger.info(f"Final portfolio value: ${episode_values[-1]:,.2f}")
            logger.info(f"Final weights: {episode_weights[-1]}")
            logger.info(f"Number of regime changes: {len(np.where(np.diff([r == 'crisis' for r in episode_regimes]))[0])}")
            logger.info(f"Time in crisis regime: {sum(r == 'crisis' for r in episode_regimes) / len(episode_regimes):.1%}")
            
            # Store regime information
            regime_history.extend(list(zip(episode_index, episode_regimes)))
            
            # Track regime periods with more detail
            regime_changes = np.where(np.diff([r == 'crisis' for r in episode_regimes]))[0]
            if len(regime_changes) > 0:
                start_idx = 0
                for change_idx in regime_changes:
                    period_returns = pd.Series(episode_values[start_idx:change_idx+1]).pct_change().dropna()
                    self.regime_periods.append({
                        'regime': episode_regimes[start_idx],
                        'start': episode_index[start_idx],
                        'end': episode_index[change_idx],
                        'returns': period_returns,
                        'avg_weight': np.mean(episode_weights[start_idx:change_idx+1], axis=0),
                        'volatility': period_returns.std() * np.sqrt(252),
                        'sharpe': period_returns.mean() / (period_returns.std() + 1e-8) * np.sqrt(252)
                    })
                    start_idx = change_idx + 1
                # Add final period
                period_returns = pd.Series(episode_values[start_idx:]).pct_change().dropna()
                self.regime_periods.append({
                    'regime': episode_regimes[start_idx],
                    'start': episode_index[start_idx],
                    'end': episode_index[-1],
                    'returns': period_returns,
                    'avg_weight': np.mean(episode_weights[start_idx:], axis=0),
                    'volatility': period_returns.std() * np.sqrt(252),
                    'sharpe': period_returns.mean() / (period_returns.std() + 1e-8) * np.sqrt(252)
                })
        
        # Convert regime history to DataFrame for easier access
        self.regime_history = pd.DataFrame(regime_history, columns=['date', 'regime'])
        self.regime_history.set_index('date', inplace=True)
        
        # Log regime statistics
        regime_counts = self.regime_history['regime'].value_counts()
        logger.info("\nRegime Statistics:")
        logger.info(f"Total periods: {len(self.regime_history)}")
        for regime, count in regime_counts.items():
            logger.info(f"{regime}: {count} periods ({count/len(self.regime_history):.1%})")
        
        # Add after regime_history is created in evaluate_agent
        logger.info(f"Regime counts: {self.regime_history['regime'].value_counts().to_dict()}")
        if 'overall' in self.performance_metrics and 'returns' in self.performance_metrics['overall']:
            returns_idx = self.performance_metrics['overall']['returns'].index
            logger.info(f"Returns index range: {returns_idx.min()} to {returns_idx.max()}")
            for crisis_name, info in self.CRISIS_PERIODS.items():
                crisis_range = pd.date_range(info['start'], info['end'])
                overlap = returns_idx.intersection(crisis_range)
                logger.info(f"{crisis_name}: {info['start']} to {info['end']} - Overlap with returns: {len(overlap)} days")
        
        # Calculate regime-specific metrics with more detail
        for regime, returns in regime_returns.items():
            if returns:
                returns_series = pd.Series(returns)
                self.performance_metrics[regime] = {
                    'returns': returns_series,
                    'cagr': ep.cagr(returns_series),
                    'volatility': ep.annual_volatility(returns_series),
                    'sharpe_ratio': ep.sharpe_ratio(returns_series),
                    'sortino_ratio': ep.sortino_ratio(returns_series),
                    'max_drawdown': ep.max_drawdown(returns_series),
                    'calmar_ratio': ep.calmar_ratio(returns_series),
                    'omega_ratio': ep.omega_ratio(returns_series),
                    'tail_ratio': ep.tail_ratio(returns_series),
                    'avg_weight': np.mean([w for w, r in zip(portfolio_weights[0].values, self.regime_history['regime']) if r == regime], axis=0) if regime in self.regime_history['regime'].values else None
                }
        
        # Calculate overall metrics with more detail
        if portfolio_values and len(portfolio_values[0]) > 0:
            returns = portfolio_values[0].pct_change().dropna()
            self.performance_metrics['overall'] = {
                'portfolio_values': portfolio_values[0],
                'portfolio_weights': portfolio_weights[0],
                'actions': actions[0],
                'returns': returns,
                'cagr': ep.cagr(returns),
                'volatility': ep.annual_volatility(returns),
                'sharpe_ratio': ep.sharpe_ratio(returns),
                'sortino_ratio': ep.sortino_ratio(returns),
                'max_drawdown': ep.max_drawdown(returns),
                'calmar_ratio': ep.calmar_ratio(returns),
                'omega_ratio': ep.omega_ratio(returns),
                'tail_ratio': ep.tail_ratio(returns),
                'avg_weight': np.mean(portfolio_weights[0].values, axis=0),
                'weight_std': np.std(portfolio_weights[0].values, axis=0)
            }
        
        # Store results
        self.results['agent' if not random_policy else 'random'] = self.performance_metrics
        
        return self.performance_metrics
    
    def evaluate_baselines(self) -> Dict[str, Dict]:
        """
        Evaluate baseline strategies.
        
        Returns:
            Dictionary of performance metrics for each baseline
        """
        logger.info("Evaluating baseline strategies")
        
        # Debug logging
        logger.info(f"Returns data shape: {self.returns_data.shape}")
        logger.info(f"Returns data columns: {self.returns_data.columns.tolist()}")
        logger.info(f"Returns data sample:\n{self.returns_data.head()}")
        logger.info(f"Returns data info:\n{self.returns_data.info()}")
        
        # Equal-weight portfolio
        equal_weights = np.ones(self.env.n_assets) / self.env.n_assets
        equal_weight_returns = self.returns_data.mean(axis=1)
        
        # Debug logging for equal weight returns
        logger.info(f"Equal weight returns shape: {equal_weight_returns.shape}")
        logger.info(f"Equal weight returns sample:\n{equal_weight_returns.head()}")
        logger.info(f"Equal weight returns info:\n{equal_weight_returns.info()}")
        
        # 60/40 portfolio (if applicable)
        if 'SPY' in self.returns_data.columns and 'TLT' in self.returns_data.columns:
            sixty_forty_weights = np.array([0.6, 0.4] + [0.0] * (self.env.n_assets - 2))
            sixty_forty_returns = (
                0.6 * self.returns_data['SPY'] +
                0.4 * self.returns_data['TLT']
            )
            # Debug logging for 60/40 returns
            logger.info(f"60/40 returns shape: {sixty_forty_returns.shape}")
            logger.info(f"60/40 returns sample:\n{sixty_forty_returns.head()}")
            logger.info(f"60/40 returns info:\n{sixty_forty_returns.info()}")
        else:
            sixty_forty_returns = None
            logger.info("SPY and/or TLT not found in returns data, skipping 60/40 portfolio")
        
        # Calculate metrics for each baseline
        baselines = {
            'equal_weight': equal_weight_returns,
            'sixty_forty': sixty_forty_returns
        }
        
        for name, returns in baselines.items():
            if returns is not None:
                try:
                    metrics = {
                        'returns': returns,
                        'cagr': ep.cagr(returns),
                        'volatility': ep.annual_volatility(returns),
                        'sharpe_ratio': ep.sharpe_ratio(returns),
                        'sortino_ratio': ep.sortino_ratio(returns),
                        'max_drawdown': ep.max_drawdown(returns),
                        'calmar_ratio': ep.calmar_ratio(returns),
                        'omega_ratio': ep.omega_ratio(returns),
                        'tail_ratio': ep.tail_ratio(returns)
                    }
                    self.results[name] = metrics
                    logger.info(f"Successfully calculated metrics for {name} strategy")
                except Exception as e:
                    logger.error(f"Error calculating metrics for {name} strategy: {str(e)}")
                    logger.error(f"Returns data causing error:\n{returns.head()}")
                    raise
        
        return self.results
    
    def evaluate_crisis_periods(self) -> Dict[str, Dict]:
        """Evaluate performance during specific crisis periods."""
        logger.info("Evaluating crisis period performance")
        
        # Get the actual start index after lookback window
        start_idx = self.env.lookback_window - 1
        logger.info(f"Starting evaluation from index {start_idx} (after lookback window)")
        
        for crisis_name, info in self.CRISIS_PERIODS.items():
            # Get crisis period data, adjusted for lookback window
            mask = (self.returns_data.index >= info['start']) & (self.returns_data.index <= info['end'])
            crisis_returns = self.returns_data[mask]
            
            if len(crisis_returns) == 0:
                logger.warning(f"No data available for crisis period: {crisis_name}")
                continue
            
            # Calculate metrics for each strategy
            crisis_metrics = {}
            
            # Agent performance
            if 'agent' in self.results:
                # Get the agent's returns from performance_metrics
                agent_returns = self.performance_metrics['overall']['returns']
                agent_index = agent_returns.index
                
                # Create mask for agent returns
                agent_mask = (agent_index >= info['start']) & (agent_index <= info['end'])
                agent_crisis_returns = agent_returns[agent_mask]
                
                if len(agent_crisis_returns) > 0:
                    crisis_metrics['agent'] = {
                        'returns': agent_crisis_returns,
                        'cagr': ep.cagr(agent_crisis_returns),
                        'volatility': ep.annual_volatility(agent_crisis_returns),
                        'sharpe_ratio': ep.sharpe_ratio(agent_crisis_returns),
                        'max_drawdown': ep.max_drawdown(agent_crisis_returns),
                        'calmar_ratio': ep.calmar_ratio(agent_crisis_returns)
                    }
                    logger.info(f"Agent crisis period {crisis_name} returns shape: {agent_crisis_returns.shape}")
                else:
                    logger.warning(f"No agent returns available for crisis period: {crisis_name}")
            
            # 60/40 performance
            if 'sixty_forty' in self.results:
                sixty_forty_returns = self.results['sixty_forty']['returns']
                sixty_forty_mask = (sixty_forty_returns.index >= info['start']) & (sixty_forty_returns.index <= info['end'])
                sixty_forty_crisis_returns = sixty_forty_returns[sixty_forty_mask]
                
                if len(sixty_forty_crisis_returns) > 0:
                    crisis_metrics['sixty_forty'] = {
                        'returns': sixty_forty_crisis_returns,
                        'cagr': ep.cagr(sixty_forty_crisis_returns),
                        'volatility': ep.annual_volatility(sixty_forty_crisis_returns),
                        'sharpe_ratio': ep.sharpe_ratio(sixty_forty_crisis_returns),
                        'max_drawdown': ep.max_drawdown(sixty_forty_crisis_returns),
                        'calmar_ratio': ep.calmar_ratio(sixty_forty_crisis_returns)
                    }
                    logger.info(f"60/40 crisis period {crisis_name} returns shape: {sixty_forty_crisis_returns.shape}")
                else:
                    logger.warning(f"No 60/40 returns available for crisis period: {crisis_name}")
            
            # Equal weight performance
            if 'equal_weight' in self.results:
                equal_weight_returns = self.results['equal_weight']['returns']
                equal_weight_mask = (equal_weight_returns.index >= info['start']) & (equal_weight_returns.index <= info['end'])
                equal_weight_crisis_returns = equal_weight_returns[equal_weight_mask]
                
                if len(equal_weight_crisis_returns) > 0:
                    crisis_metrics['equal_weight'] = {
                        'returns': equal_weight_crisis_returns,
                        'cagr': ep.cagr(equal_weight_crisis_returns),
                        'volatility': ep.annual_volatility(equal_weight_crisis_returns),
                        'sharpe_ratio': ep.sharpe_ratio(equal_weight_crisis_returns),
                        'max_drawdown': ep.max_drawdown(equal_weight_crisis_returns),
                        'calmar_ratio': ep.calmar_ratio(equal_weight_crisis_returns)
                    }
                    logger.info(f"Equal weight crisis period {crisis_name} returns shape: {equal_weight_crisis_returns.shape}")
                else:
                    logger.warning(f"No equal weight returns available for crisis period: {crisis_name}")
            
            if crisis_metrics:
                self.results[crisis_name] = crisis_metrics
                logger.info(f"Successfully calculated metrics for crisis period: {crisis_name}")
            else:
                logger.warning(f"No metrics calculated for crisis period: {crisis_name}")
        
        return self.results
    
    def calculate_statistical_significance(self) -> Dict[str, float]:
        """
        Calculate statistical significance of performance differences.
        
        Returns:
            Dictionary of p-values for different metrics
        """
        if 'agent' not in self.results or 'equal_weight' not in self.results:
            raise ValueError("Must evaluate agent and baselines first")
        
        # Compare returns
        agent_returns = self.performance_metrics['overall']['returns']
        equal_weight_returns = self.results['equal_weight']['returns']
        
        # Debug logging
        logger.info(f"Agent returns shape: {agent_returns.shape}")
        logger.info(f"Equal weight returns shape: {equal_weight_returns.shape}")
        logger.info(f"Agent returns sample:\n{agent_returns.head()}")
        logger.info(f"Equal weight returns sample:\n{equal_weight_returns.head()}")
        
        # Ensure indices match
        common_index = agent_returns.index.intersection(equal_weight_returns.index)
        if len(common_index) == 0:
            raise ValueError("No common dates between agent and equal weight returns")
        
        agent_returns = agent_returns[common_index]
        equal_weight_returns = equal_weight_returns[common_index]
        
        # Welch's t-test for returns
        t_stat, p_value = stats.ttest_ind(
            agent_returns,
            equal_weight_returns,
            equal_var=False
        )
        
        # Bootstrap confidence intervals for Sharpe ratio
        def bootstrap_sharpe(returns, n_samples=1000):
            sharpe_ratios = []
            for _ in range(n_samples):
                sample = returns.sample(n=len(returns), replace=True)
                sharpe_ratios.append(ep.sharpe_ratio(sample))
            return np.percentile(sharpe_ratios, [2.5, 97.5])
        
        agent_sharpe_ci = bootstrap_sharpe(agent_returns)
        equal_weight_sharpe_ci = bootstrap_sharpe(equal_weight_returns)
        
        significance = {
            'returns_p_value': p_value,
            'agent_sharpe_ci': agent_sharpe_ci,
            'equal_weight_sharpe_ci': equal_weight_sharpe_ci,
            't_statistic': t_stat,
            'common_dates_count': len(common_index)
        }
        
        # Log significance results
        logger.info("\nStatistical Significance Results:")
        logger.info(f"T-statistic: {t_stat:.4f}")
        logger.info(f"P-value: {p_value:.4f}")
        logger.info(f"Agent Sharpe 95% CI: [{agent_sharpe_ci[0]:.4f}, {agent_sharpe_ci[1]:.4f}]")
        logger.info(f"Equal Weight Sharpe 95% CI: [{equal_weight_sharpe_ci[0]:.4f}, {equal_weight_sharpe_ci[1]:.4f}]")
        logger.info(f"Number of common dates: {len(common_index)}")
        
        self.results['significance'] = significance
        return significance
    
    def plot_performance(self, save_path: Optional[str] = None) -> None:
        """
        Plot performance analysis emphasizing crisis period outperformance.
        Args:
            save_path: Optional path to save plots
        """
        # Use explicit crisis/normal masks
        returns = self.performance_metrics.get('overall', {}).get('returns')
        if returns is None or returns.empty:
            print('No returns data available for plotting.')
            return
        crisis_mask, normal_mask = self.get_crisis_and_normal_masks(returns.index)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 25))
        gs = fig.add_gridspec(5, 2)
        
        # Regime summary table
        crisis_days = crisis_mask.sum()
        normal_days = normal_mask.sum()
        summary_text = f"Regime Summary:\nCrisis: {crisis_days} days\nNormal: {normal_days} days"
        fig.text(0.01, 0.98, summary_text, fontsize=10, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Prepare strategies
        strategies = {
            'PPO Agent': self.performance_metrics.get('overall', {}).get('returns'),
            'Equal Weight': self.results.get('equal_weight', {}).get('returns'),
            '60/40 Portfolio': self.results.get('sixty_forty', {}).get('returns')
        }
        
        # Plot 1: Strategy Performance by Market Regime (using explicit masks)
        ax1 = fig.add_subplot(gs[0, :])
        regime_returns = {'normal': [], 'crisis': []}
        strategy_names = []
        for strategy_name, strat_returns in strategies.items():
            if strat_returns is not None and not strat_returns.empty:
                strat_returns = strat_returns.loc[returns.index]  # align
                regime_returns['crisis'].append(strat_returns[crisis_mask].mean() * 100 if crisis_days > 0 else 0)
                regime_returns['normal'].append(strat_returns[normal_mask].mean() * 100 if normal_days > 0 else 0)
                strategy_names.append(strategy_name)
        x = np.arange(len(strategy_names))
        width = 0.35
        ax1.bar(x - width/2, regime_returns['normal'], width, label='Normal Period', color='green', alpha=0.7)
        ax1.bar(x + width/2, regime_returns['crisis'], width, label='Crisis Period', color='red', alpha=0.7)
        for i, v in enumerate(regime_returns['normal']):
            ax1.text(i - width/2, v, f'{v:.1f}%', ha='center', va='bottom')
        for i, v in enumerate(regime_returns['crisis']):
            ax1.text(i + width/2, v, f'{v:.1f}%', ha='center', va='bottom')
        ax1.set_title('Strategy Performance by Market Regime (Explicit Crisis Dates)', fontsize=14, pad=20)
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Average Daily Return (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategy_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Crisis Period Drawdown Comparison
        ax2 = fig.add_subplot(gs[1, :])
        has_drawdown_data = False
        for strategy_name, strat_returns in strategies.items():
            if strat_returns is not None and not strat_returns.empty and crisis_days > 0:
                strat_returns = strat_returns.loc[returns.index]
                crisis_returns = strat_returns[crisis_mask]
                cum_returns = (1 + crisis_returns).cumprod()
                drawdown = (cum_returns / cum_returns.cummax() - 1)
                ax2.plot(drawdown.index, drawdown.values * 100, label=f'{strategy_name} Drawdown', alpha=0.7)
                has_drawdown_data = True
        if has_drawdown_data:
            ax2.set_title('Crisis Period Drawdown Comparison', fontsize=14, pad=20)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No crisis regime detected in data',
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='red')
            ax2.set_title('Crisis Period Drawdown Comparison')
            ax2.axis('off')
        
        # Plot 3: Rolling Sharpe Ratio During Crises
        ax3 = fig.add_subplot(gs[2, :])
        has_sharpe_data = False
        def calculate_rolling_sharpe(returns, window=60):
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            return (rolling_mean / rolling_std) * np.sqrt(252)
        for strategy_name, strat_returns in strategies.items():
            if strat_returns is not None and not strat_returns.empty and crisis_days > 0:
                strat_returns = strat_returns.loc[returns.index]
                crisis_returns = strat_returns[crisis_mask]
                crisis_sharpe = calculate_rolling_sharpe(crisis_returns)
                ax3.plot(crisis_sharpe.index, crisis_sharpe.values, label=f'{strategy_name} Sharpe', alpha=0.7)
                has_sharpe_data = True
        if has_sharpe_data:
            ax3.set_title('Rolling Sharpe Ratio During Crisis Periods (60-day window)', fontsize=14, pad=20)
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No crisis regime detected in data',
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='red')
            ax3.set_title('Rolling Sharpe Ratio During Crisis Periods')
            ax3.axis('off')
        
        # Plot 4: Crisis Period Recovery
        ax4 = fig.add_subplot(gs[3, :])
        has_recovery_data = False
        for strategy_name, strat_returns in strategies.items():
            if strat_returns is not None and not strat_returns.empty and crisis_days > 0:
                strat_returns = strat_returns.loc[returns.index]
                crisis_returns = strat_returns[crisis_mask]
                cum_returns = (1 + crisis_returns).cumprod()
                normalized_returns = cum_returns / cum_returns.iloc[0]
                ax4.plot(normalized_returns.index, normalized_returns.values, label=f'{strategy_name} Recovery', alpha=0.7)
                has_recovery_data = True
        if has_recovery_data:
            ax4.set_title('Crisis Period Recovery Comparison', fontsize=14, pad=20)
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Normalized Value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=1, color='black', linestyle='-', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No crisis regime detected in data',
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='red')
            ax4.set_title('Crisis Period Recovery Comparison')
            ax4.axis('off')
        
        # Plot 5: Crisis Period Risk Metrics
        ax5 = fig.add_subplot(gs[4, :])
        has_risk_data = False
        risk_metrics = {
            'Volatility': [],
            'Max Drawdown': [],
            'Sharpe Ratio': [],
            'Sortino Ratio': []
        }
        strategy_labels = []
        for strategy_name, strat_returns in strategies.items():
            if strat_returns is not None and not strat_returns.empty and crisis_days > 0:
                strat_returns = strat_returns.loc[returns.index]
                crisis_returns = strat_returns[crisis_mask]
                risk_metrics['Volatility'].append(ep.annual_volatility(crisis_returns) * 100)
                risk_metrics['Max Drawdown'].append(ep.max_drawdown(crisis_returns) * 100)
                risk_metrics['Sharpe Ratio'].append(ep.sharpe_ratio(crisis_returns))
                risk_metrics['Sortino Ratio'].append(ep.sortino_ratio(crisis_returns))
                strategy_labels.append(strategy_name)
                has_risk_data = True
        if has_risk_data:
            x = np.arange(len(strategy_labels))
            width = 0.2
            multiplier = 0
            for metric_name, values in risk_metrics.items():
                offset = width * multiplier
                rects = ax5.bar(x + offset, values, width, label=metric_name, alpha=0.7)
                for rect in rects:
                    height = rect.get_height()
                    ax5.text(rect.get_x() + rect.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom')
                multiplier += 1
            ax5.set_title('Crisis Period Risk Metrics Comparison', fontsize=14, pad=20)
            ax5.set_xlabel('Strategy')
            ax5.set_ylabel('Value')
            ax5.set_xticks(x + width * (len(risk_metrics) - 1) / 2)
            ax5.set_xticklabels(strategy_labels)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No crisis regime detected in data',
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='red')
            ax5.set_title('Crisis Period Risk Metrics Comparison')
            ax5.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()
        else:
            plt.show()
    
    def print_performance_summary(self) -> None:
        """Print detailed performance summary."""
        print("\nPerformance Summary:")
        print("=" * 80)
        
        def print_metrics(metrics: Dict, regime_name: str) -> None:
            """Helper function to print metrics with error handling."""
            print(f"\n{regime_name} Performance:")
            print("-" * 40)
            
            # Define expected metrics and their display names
            metric_names = {
                'cagr': 'CAGR',
                'volatility': 'Volatility',
                'sharpe_ratio': 'Sharpe Ratio',
                'sortino_ratio': 'Sortino Ratio',
                'max_drawdown': 'Max Drawdown',
                'calmar_ratio': 'Calmar Ratio',
                'omega_ratio': 'Omega Ratio',
                'tail_ratio': 'Tail Ratio'
            }
            
            # Print each metric if it exists
            for metric_key, display_name in metric_names.items():
                if metric_key in metrics and metrics[metric_key] is not None:
                    value = metrics[metric_key]
                    # Format as percentage for certain metrics
                    if metric_key in ['cagr', 'volatility', 'max_drawdown']:
                        print(f"{display_name}: {value:.2%}")
                    else:
                        print(f"{display_name}: {value:.2f}")
                else:
                    print(f"{display_name}: N/A")
        
        # Overall metrics
        if 'overall' in self.performance_metrics:
            print_metrics(self.performance_metrics['overall'], "Overall")
        
        # Regime-specific metrics
        for regime in ['normal', 'crisis']:
            if regime in self.performance_metrics and self.performance_metrics[regime]:
                print_metrics(self.performance_metrics[regime], f"{regime.capitalize()} Regime")
        
        # Position concentration analysis
        print("\nPosition Concentration Analysis:")
        print("-" * 40)
        if self.position_concentration:
            concentration_df = pd.DataFrame(self.position_concentration)
            for regime in ['normal', 'crisis']:
                regime_data = concentration_df[concentration_df['regime'] == regime]
                if not regime_data.empty:
                    print(f"\n{regime.capitalize()} Regime:")
                    print(f"Average HHI: {regime_data['hhi'].mean():.4f}")
                    print(f"Max HHI: {regime_data['hhi'].max():.4f}")
                    print(f"Min HHI: {regime_data['hhi'].min():.4f}")
        else:
            print("No position concentration data available")
        
        # Correlation analysis
        print("\nCorrelation Analysis:")
        print("-" * 40)
        if self.correlation_metrics:
            correlation_df = pd.DataFrame(self.correlation_metrics)
            for regime in ['normal', 'crisis']:
                regime_data = correlation_df[correlation_df['regime'] == regime]
                if not regime_data.empty:
                    print(f"\n{regime.capitalize()} Regime:")
                    print(f"Average Correlation: {regime_data['avg_correlation'].mean():.4f}")
                    print(f"Max Correlation: {regime_data['avg_correlation'].max():.4f}")
                    print(f"Min Correlation: {regime_data['avg_correlation'].min():.4f}")
        else:
            print("No correlation data available")
        
        # Print crisis period performance if available
        if any(k in self.results for k in self.CRISIS_PERIODS.keys()):
            print("\nCrisis Period Performance:")
            print("-" * 40)
            for crisis_name, crisis_info in self.CRISIS_PERIODS.items():
                if crisis_name in self.results:
                    print(f"\n{crisis_info['name']} ({crisis_info['start']} to {crisis_info['end']}):")
                    print(f"Description: {crisis_info['description']}")
                    for strategy, metrics in self.results[crisis_name].items():
                        print(f"\n{strategy.capitalize()} Strategy:")
                        print_metrics(metrics, "")

    def get_crisis_and_normal_masks(self, returns_index):
        crisis_mask = pd.Series(False, index=returns_index)
        for info in self.CRISIS_PERIODS.values():
            crisis_mask |= (returns_index >= info['start']) & (returns_index <= info['end'])
        normal_mask = ~crisis_mask
        return crisis_mask, normal_mask

    def plot_summary_performance(self, save_path: Optional[str] = None):
        """
        Plot summary performance: cumulative returns with shaded periods, bar plots for total return, max drawdown, Sharpe ratio, and rolling volatility, all by period and strategy.
        """
        returns = self.performance_metrics.get('overall', {}).get('returns')
        if returns is None or returns.empty:
            print('No returns data available for plotting.')
            return
        crisis_mask, normal_mask = self.get_crisis_and_normal_masks(returns.index)
        periods = list(self.CRISIS_PERIODS.keys())
        period_labels = [self.CRISIS_PERIODS[p]['name'] for p in periods]
        period_ranges = [(self.CRISIS_PERIODS[p]['start'], self.CRISIS_PERIODS[p]['end']) for p in periods]
        # Add "Normal Market Period" as a period if not already
        if 'normal_period' not in periods:
            period_labels.append('Normal Market Period')
            period_ranges.append((returns.index[normal_mask][0], returns.index[normal_mask][-1]))
        # Prepare strategies
        strategies = {
            'agent': self.performance_metrics.get('overall', {}).get('returns'),
            'equal_weight': self.results.get('equal_weight', {}).get('returns'),
            'sixty_forty': self.results.get('sixty_forty', {}).get('returns')
        }
        # --- Cumulative Returns Plot ---
        plt.figure(figsize=(16, 4))
        for name, strat_returns in strategies.items():
            if strat_returns is not None and not strat_returns.empty:
                aligned = strat_returns.loc[returns.index]
                cum = (1 + aligned).cumprod()
                plt.plot(cum.index, cum.values, label=name)
        # Shade crisis/normal periods
        ax = plt.gca()
        colors = ['#ffcccc', '#ffe4b2', '#c6f5d7']
        for i, (label, (start, end)) in enumerate(zip(period_labels, period_ranges)):
            color = colors[i % len(colors)]
            plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=color, alpha=0.4, label=label if i < 3 else None)
        plt.legend(loc='best')
        plt.title('Cumulative Returns with Crisis Periods')
        plt.ylabel('Cumulative Return')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_cumulative.png'), bbox_inches='tight', dpi=200)
        plt.show()
        # --- Bar Plots for Period Metrics ---
        # Prepare period masks
        period_masks = [(returns.index >= start) & (returns.index <= end) for (start, end) in period_ranges]
        # Compute metrics for each period and strategy
        metrics = {'Total Return': {}, 'Max Drawdown': {}, 'Sharpe Ratio': {}, 'Volatility': {}}
        for strat_name, strat_returns in strategies.items():
            if strat_returns is None or strat_returns.empty:
                continue
            aligned = strat_returns.loc[returns.index]
            for i, mask in enumerate(period_masks):
                period = period_labels[i]
                period_returns = aligned[mask]
                if len(period_returns) == 0:
                    continue
                metrics['Total Return'].setdefault(period, {})[strat_name] = (1 + period_returns).prod() - 1
                cum = (1 + period_returns).cumprod()
                drawdown = (cum / cum.cummax() - 1)
                metrics['Max Drawdown'].setdefault(period, {})[strat_name] = drawdown.min()
                metrics['Sharpe Ratio'].setdefault(period, {})[strat_name] = ep.sharpe_ratio(period_returns)
                metrics['Volatility'].setdefault(period, {})[strat_name] = ep.annual_volatility(period_returns)
        # Bar plot for Total Return
        df_return = pd.DataFrame(metrics['Total Return']).T
        df_return.plot(kind='bar', figsize=(12, 3), title='Crisis Period Returns')
        plt.ylabel('Total Return')
        plt.xlabel('Period')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_bar_return.png'), bbox_inches='tight', dpi=200)
        plt.show()
        # Bar plot for Max Drawdown
        df_dd = pd.DataFrame(metrics['Max Drawdown']).T
        df_dd.plot(kind='bar', figsize=(8, 3), title='Crisis Period Maximum Drawdowns')
        plt.ylabel('Maximum Drawdown')
        plt.xlabel('Period')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_bar_drawdown.png'), bbox_inches='tight', dpi=200)
        plt.show()
        # Bar plot for Sharpe Ratio
        df_sharpe = pd.DataFrame(metrics['Sharpe Ratio']).T
        df_sharpe.plot(kind='bar', figsize=(8, 3), title='Crisis Period Sharpe Ratios')
        plt.ylabel('Sharpe Ratio')
        plt.xlabel('Period')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_bar_sharpe.png'), bbox_inches='tight', dpi=200)
        plt.show()
        # --- Rolling Volatility Plot ---
        plt.figure(figsize=(16, 4))
        for name, strat_returns in strategies.items():
            if strat_returns is not None and not strat_returns.empty:
                aligned = strat_returns.loc[returns.index]
                rolling_vol = aligned.rolling(window=20).std() * np.sqrt(252)
                plt.plot(rolling_vol.index, rolling_vol.values, label=name)
        for i, (label, (start, end)) in enumerate(zip(period_labels, period_ranges)):
            color = colors[i % len(colors)]
            plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=color, alpha=0.4, label=label if i < 3 else None)
        plt.legend(loc='best')
        plt.title('Rolling Volatility (20-day) with Crisis Periods')
        plt.ylabel('Annualized Volatility')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_rolling_vol.png'), bbox_inches='tight', dpi=200)
        plt.show()

def main():
    """Parse arguments and evaluate the agent."""
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agent')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to trained model directory')
    parser.add_argument('--asset-class', type=str, required=True,
                      choices=['traditional', 'futures', 'high_vol'],
                      help='Asset class being traded')
    parser.add_argument('--regime', type=str, default='testing',
                      choices=list(REGIMES.keys()),
                      help='Market regime to evaluate on')
    parser.add_argument('--n-episodes', type=int, default=1,
                      help='Number of episodes to evaluate')
    parser.add_argument('--save-plots', action='store_true',
                      help='Save plots to file')
    parser.add_argument('--random-policy', action='store_true',
                      help='Use random actions for debugging')
    parser.add_argument('--no-normalize', action='store_true',
                      help='Disable VecNormalize for debugging')
    args = parser.parse_args()
    # Create evaluator
    class CustomPortfolioEvaluator(PortfolioEvaluator):
        def __init__(self, model_path, asset_class, regime, no_normalize=False):
            self.model_path = model_path
            self.asset_class = asset_class
            self.regime = regime
            
            # Initialize performance tracking attributes
            self.performance_metrics = {
                'normal': {},
                'crisis': {},
                'overall': {}
            }
            self.regime_periods = []
            self.position_concentration = []
            self.correlation_metrics = []
            self.results = {}
            
            # Check for best model
            best_model_path = os.path.join(model_path, 'best_model.zip')
            final_model_path = os.path.join(model_path, 'final_model.zip')
            
            if os.path.exists(best_model_path):
                logger.info("Using best model from training (best_model.zip)")
                model_to_load = best_model_path
            else:
                logger.info("Best model not found, using final model (final_model.zip)")
                model_to_load = final_model_path
            
            # Load the model first to get training configuration
            try:
                self.model = PPO.load(model_to_load)
                logger.info(f"Successfully loaded model from {model_to_load}")
                
                # Get lookback window from model's observation space
                model_obs_shape = self.model.observation_space.shape
                if len(model_obs_shape) == 2:  # (lookback, features)
                    self.lookback_window = model_obs_shape[0]
                    logger.info(f"Using lookback window from model: {self.lookback_window}")
                else:
                    raise ValueError(f"Unexpected model observation space shape: {model_obs_shape}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
            
            # Load data
            loader = DataLoader(asset_class=self.asset_class, regime=self.regime)
            price_data, returns_data, features_data = loader.load_data()
            if price_data is None or returns_data is None or features_data is None:
                raise RuntimeError(
                    "Data loading failed after download. Check if the data files are being saved and paths are correct.\n"
                    f"price_data: {type(price_data)}, returns_data: {type(returns_data)}, features_data: {type(features_data)}"
                )
            
            # Get the assets used in training (from returns data)
            self.assets = returns_data.columns.tolist()
            logger.info(f"Assets used in training: {self.assets}")
            
            # Filter price data to match training assets
            if isinstance(price_data.columns, pd.MultiIndex):
                # Handle multi-index columns (OHLCV data)
                price_data = price_data.loc[:, (self.assets, 'Close')]
                price_data.columns = self.assets  # Flatten to single index
            else:
                # Handle single index columns
                price_data = price_data[self.assets]
            
            # Filter features data to match training assets
            feature_columns = []
            for asset in self.assets:
                asset_features = [col for col in features_data.columns if col.startswith(f"{asset}_")]
                feature_columns.extend(asset_features)
            
            # Add any global features (like VIX)
            global_features = [col for col in features_data.columns if not any(col.startswith(f"{asset}_") for asset in self.assets)]
            feature_columns.extend(global_features)
            
            features_data = features_data[feature_columns]
            
            logger.info(f"Filtered data shapes:")
            logger.info(f"Price data: {price_data.shape}")
            logger.info(f"Returns data: {returns_data.shape}")
            logger.info(f"Features data: {features_data.shape}")
            
            self.price_data = price_data
            self.returns_data = returns_data
            self.features_data = features_data
            
            def make_env():
                env = PortfolioTradingEnv(
                    price_data=self.price_data,
                    returns_data=self.returns_data,
                    features_data=self.features_data,
                    lookback_window=self.lookback_window,
                    # Use env_config attributes with dot notation
                    transaction_cost=env_config.transaction_cost,
                    risk_penalty=env_config.risk_penalty,
                    max_position_size=env_config.max_position_size,
                    min_position_size=env_config.min_position_size,
                    crisis_threshold=env_config.crisis_threshold,
                    diversification_bonus=env_config.diversification_bonus,
                    initial_balance=env_config.initial_balance,
                    reward_scaling=env_config.reward_scaling
                )
                # Debug logging for environment configuration
                logger.info("Environment configuration:")
                logger.info(f"Transaction cost: {env.transaction_cost}")
                logger.info(f"Risk penalty: {env.risk_penalty}")
                logger.info(f"Max position size: {env.max_position_size}")
                logger.info(f"Min position size: {env.min_position_size}")
                logger.info(f"Crisis threshold: {env.crisis_threshold}")
                logger.info(f"Diversification bonus: {env.diversification_bonus}")
                logger.info(f"Current observation space shape: {env.observation_space.shape}")
                return env
            
            # Create a temporary environment to check shapes
            temp_env = make_env()
            logger.info(f"Number of assets: {temp_env.n_assets}")
            logger.info(f"Number of technical features: {self.features_data.shape[1]}")
            logger.info(f"Lookback window: {temp_env.lookback_window}")
            logger.info(f"Expected observation space shape: ({temp_env.lookback_window}, {temp_env.n_assets * 2 + self.features_data.shape[1] + 1 + temp_env.n_assets})")
            
            if not no_normalize:
                try:
                    # Try to load VecNormalize stats
                    vec_normalize_path = os.path.join(model_path, 'vec_normalize.pkl')
                    if os.path.exists(vec_normalize_path):
                        try:
                            self.vec_normalize = VecNormalize.load(
                                vec_normalize_path,
                                venv=DummyVecEnv([make_env])
                            )
                            logger.info("Successfully loaded VecNormalize stats")
                        except Exception as e:
                            logger.warning(f"Could not load VecNormalize stats: {str(e)}")
                            logger.info("Creating new VecNormalize instance")
                            self.vec_normalize = VecNormalize(
                                DummyVecEnv([make_env]),
                                norm_obs=True,
                                norm_reward=True,
                                clip_obs=10.0,
                                training=False
                            )
                    else:
                        logger.warning("VecNormalize file not found, creating new instance")
                        self.vec_normalize = VecNormalize(
                            DummyVecEnv([make_env]),
                            norm_obs=True,
                            norm_reward=True,
                            clip_obs=10.0,
                            training=False
                        )
                except Exception as e:
                    logger.error(f"Error setting up VecNormalize: {str(e)}")
                    logger.info("Falling back to no normalization")
                    self.vec_normalize = None
            else:
                self.vec_normalize = None
            
            # Create the evaluation environment using the same make_env function
            self.env = make_env()
    
    evaluator = CustomPortfolioEvaluator(
        model_path=args.model_path,
        asset_class=args.asset_class,
        regime=args.regime,
        no_normalize=args.no_normalize
    )
    # Evaluate agent and baselines
    evaluator.evaluate_agent(n_episodes=args.n_episodes, random_policy=args.random_policy)
    evaluator.evaluate_baselines()
    evaluator.evaluate_crisis_periods()
    evaluator.calculate_statistical_significance()
    evaluator.print_performance_summary()
    if args.save_plots:
        save_path = os.path.join(args.model_path, 'evaluation_plots.png')
        evaluator.plot_performance(save_path=save_path)
        print(f"\nPlots saved to: {save_path}")
    else:
        evaluator.plot_performance()
    evaluator.plot_summary_performance(save_path=save_path)

if __name__ == '__main__':
    main() 