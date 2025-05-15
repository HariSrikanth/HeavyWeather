"""
Training script for the PPO agent using stable-baselines3.
Handles model training, evaluation, and saving.
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Import baselines with error handling
try:
    import baselines
except RuntimeError as e:
    if "Could not parse python long as longdouble" in str(e):
        # Workaround for macOS scipy issue
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        # Force numpy to use float64 instead of longdouble
        np.set_printoptions(precision=16)
        import baselines
    else:
        raise

from data_loader import DataLoader
from trading_env import PortfolioTradingEnv
from config import (
    ppo_config,
    MODEL_DIR,
    LOG_DIR,
    REGIMES
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.portfolio_values = []
        self.returns = []
        self.volatilities = []
    
    def _on_step(self) -> bool:
        """
        Log custom metrics after each step.
        """
        # Get portfolio value from info
        portfolio_value = self.locals['infos'][0]['portfolio_value']
        self.portfolio_values.append(portfolio_value)
        # Print debug info: action, reward, state, reward components
        action = self.locals['actions'][0] if 'actions' in self.locals else None
        reward = self.locals['rewards'][0] if 'rewards' in self.locals else None
        obs = self.locals['new_obs'][0] if 'new_obs' in self.locals else None
        info = self.locals['infos'][0] if 'infos' in self.locals else None
        print(f"[TRAIN] Step: {self.num_timesteps}, Action: {action}, Reward: {reward}, Portfolio Value: {portfolio_value}, Obs: {obs}, Info: {info}")
        # Calculate daily return
        if len(self.portfolio_values) > 1:
            daily_return = (portfolio_value / self.portfolio_values[-2]) - 1
            self.returns.append(daily_return)
            # Calculate rolling volatility (20-day)
            if len(self.returns) >= 20:
                volatility = np.std(self.returns[-20:]) * np.sqrt(252)
                self.volatilities.append(volatility)
                # Log to tensorboard
                self.logger.record('portfolio/value', portfolio_value)
                self.logger.record('portfolio/daily_return', daily_return)
                self.logger.record('portfolio/volatility', volatility)
                self.logger.record('portfolio/sharpe_ratio', 
                                 np.mean(self.returns[-20:]) / (volatility + 1e-8) * np.sqrt(252))
        return True

def create_env(
    asset_class: str,
    regime: str,
    is_eval: bool = False,
    init_meanvar_weights: Optional[np.ndarray] = None
) -> PortfolioTradingEnv:
    """
    Create and configure the trading environment.
    
    Args:
        asset_class: Asset class to trade
        regime: Market regime to use
        is_eval: Whether this is for evaluation
        init_meanvar_weights: Optional initial weights for the portfolio
        
    Returns:
        Configured trading environment
    """
    # Load data
    loader = DataLoader(asset_class=asset_class, regime=regime)
    price_data, returns_data, features_data = loader.load_data()
    
    if price_data is None:
        logger.info(f"Downloading data for {asset_class} assets in {regime} regime")
        loader.download_data()
        loader.calculate_returns()
        loader.calculate_features()
        price_data, returns_data, features_data = loader.load_data()
    
    # Compute mean-variance weights if requested
    if init_meanvar_weights is not None:
        initial_weights = init_meanvar_weights
    else:
        initial_weights = None
    
    # Create environment
    env = PortfolioTradingEnv(
        price_data=price_data,
        returns_data=returns_data,
        features_data=features_data
    )
    
    # Patch env to use custom initial weights if provided
    if initial_weights is not None:
        orig_reset = env.reset
        def reset_with_weights(*args, **kwargs):
            obs, info = orig_reset(*args, **kwargs)
            env.portfolio_weights = initial_weights
            return obs, info
        env.reset = reset_with_weights
    
    # Wrap with Monitor for logging
    env = Monitor(env, filename=None if is_eval else os.path.join(LOG_DIR, 'monitor.csv'))
    
    return env

def train_agent(
    asset_class: str,
    regime: str = 'training',
    total_timesteps: int = ppo_config.total_timesteps,
    eval_freq: int = ppo_config.eval_freq,
    n_eval_episodes: int = ppo_config.n_eval_episodes,
    save_freq: int = 10000,
    tensorboard_log: str = ppo_config.tensorboard_log,
    seed: Optional[int] = None,
    init_meanvar: bool = False,
    n_envs: int = 1  # Changed default to 1 since we're using DummyVecEnv
) -> Tuple[PPO, str]:
    """
    Train the PPO agent.
    
    Args:
        asset_class: Asset class to trade
        regime: Market regime to use for training
        total_timesteps: Total number of timesteps to train
        eval_freq: How often to evaluate the agent
        n_eval_episodes: Number of episodes for evaluation
        save_freq: How often to save the model
        tensorboard_log: Directory for tensorboard logs
        seed: Random seed for reproducibility
        init_meanvar: Whether to initialize policy with mean-variance optimal weights
        n_envs: Number of environments (using DummyVecEnv, so this is effectively 1)
    """
    # Set random seeds
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Enable GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Compute mean-variance weights if requested
    init_meanvar_weights = None
    if init_meanvar:
        loader = DataLoader(asset_class=asset_class, regime=regime)
        _, returns_data, _ = loader.load_data()
        if returns_data is None:
            loader.download_data()
            loader.calculate_returns()
            _, returns_data, _ = loader.load_data()
        mv = baselines.MeanVarianceStrategy(returns_data)
        mv_weights = mv.run().weights.iloc[0].values
        init_meanvar_weights = mv_weights
    
    # Create environment using DummyVecEnv
    train_env = DummyVecEnv([lambda: create_env(asset_class, regime, init_meanvar_weights=init_meanvar_weights)])
    eval_env = DummyVecEnv([lambda: create_env(asset_class, 'testing', is_eval=True)])
    
    # Normalize observations and rewards
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=False
    )
    
    # Create model with optimized settings
    model = PPO(
        policy=ppo_config.policy,
        env=train_env,
        learning_rate=ppo_config.learning_rate,
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=ppo_config.clip_range,
        ent_coef=ppo_config.ent_coef,
        verbose=ppo_config.verbose,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            net_arch=ppo_config.net_arch,
            activation_fn=torch.nn.ReLU,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs=dict(weight_decay=0.01)
        ),
        device=device
    )
    
    # Create callbacks
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(MODEL_DIR, f'ppo_{asset_class}_{timestamp}')
    
    callbacks = [
        # Save model checkpoints
        CheckpointCallback(
            save_freq=save_freq,
            save_path=model_path,
            name_prefix='model'
        ),
        # Evaluate model
        EvalCallback(
            eval_env,
            best_model_save_path=model_path,
            log_path=os.path.join(model_path, 'eval'),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        ),
        # Custom metrics with reduced logging
        TensorboardCallback()
    ]
    
    # Train model with progress bar
    logger.info(f"Starting training for {asset_class} assets with {n_envs} parallel environments")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=f"ppo_{asset_class}"
    )
    
    # Save final model and normalization stats
    model.save(os.path.join(model_path, 'final_model'))
    train_env.save(os.path.join(model_path, 'vec_normalize.pkl'))
    
    logger.info(f"Training completed. Model saved to {model_path}")
    return model, model_path

def main():
    """Parse arguments and train the agent."""
    parser = argparse.ArgumentParser(description='Train PPO agent for portfolio allocation')
    parser.add_argument('--asset-class', type=str, default='traditional',
                      choices=['traditional', 'futures', 'high_vol'],
                      help='Asset class to trade')
    parser.add_argument('--regime', type=str, default='training',
                      choices=list(REGIMES.keys()),
                      help='Market regime to use for training')
    parser.add_argument('--timesteps', type=int, default=ppo_config.total_timesteps,
                      help='Total timesteps for training')
    parser.add_argument('--eval-freq', type=int, default=ppo_config.eval_freq,
                      help='Evaluation frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=ppo_config.n_eval_episodes,
                      help='Number of evaluation episodes')
    parser.add_argument('--save-freq', type=int, default=10000,
                      help='Model save frequency')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--init-meanvar', action='store_true',
                      help='Initialize policy with mean-variance optimal weights')
    parser.add_argument('--n-envs', type=int, default=1,  # Changed default to 1
                      help='Number of environments (using DummyVecEnv)')
    
    args = parser.parse_args()
    
    # Train agent
    model, model_path = train_agent(
        asset_class=args.asset_class,
        regime=args.regime,
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_freq=args.save_freq,
        seed=args.seed,
        init_meanvar=args.init_meanvar,
        n_envs=args.n_envs
    )
    
    print(f"\nTraining completed. Model saved to: {model_path}")
    print("\nTo load the model for evaluation:")
    print(f"model = PPO.load('{os.path.join(model_path, 'final_model')}')")

if __name__ == '__main__':
    main() 