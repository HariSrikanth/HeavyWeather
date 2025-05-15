"""
Custom Gymnasium environment for portfolio trading.
Implements state space, action space, and reward calculation for the PPO agent.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

from config import env_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioState:
    """Container for portfolio state information."""
    current_step: int
    portfolio_value: float
    portfolio_weights: np.ndarray
    asset_prices: np.ndarray
    features: np.ndarray
    done: bool = False

class PortfolioTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for portfolio trading.
    
    State space:
        - Asset prices and returns
        - Technical indicators (RSI, MACD, volatility)
        - Portfolio weights
        - Portfolio value
        - Market regime indicators (optional)
    
    Action space:
        - Continuous allocation weights (softmax to sum to 1)
        - Long-only portfolio (weights >= 0)
        - Maximum position size constraint
    
    Reward:
        - Log portfolio return
        - Transaction cost penalty
        - Risk penalty (optional)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        returns_data: pd.DataFrame,
        features_data: pd.DataFrame,
        initial_balance: float = 1_000_000,
        transaction_cost: float = 0.0005,  # Reduced from 0.001
        risk_penalty: float = 0.3,         # Reduced from 0.5
        max_position_size: float = 0.35,   # Reduced from 0.4
        lookback_window: int = 30,         # Increased from 20
        reward_scaling: float = 1.0,
        crisis_threshold: float = -0.015,  # Reduced from -0.02 for earlier crisis detection
        min_position_size: float = 0.05,   # Kept the same
        diversification_bonus: float = 0.15 # Increased from 0.1
    ):
        """
        Initialize the trading environment with improved risk management.
        
        Args:
            price_data: DataFrame of asset prices
            returns_data: DataFrame of asset returns
            features_data: DataFrame of technical features
            initial_balance: Initial portfolio value
            transaction_cost: Cost per trade as a fraction
            risk_penalty: Penalty for portfolio risk
            max_position_size: Maximum position size as a fraction
            lookback_window: Number of past days to observe
            reward_scaling: Scaling factor for rewards
            crisis_threshold: Return threshold for crisis detection
            min_position_size: Minimum position size as a fraction
            diversification_bonus: Bonus for portfolio diversification
        """
        super().__init__()
        
        # Store data
        self.price_data = price_data
        self.returns_data = returns_data
        self.features_data = features_data
        self.assets = returns_data.columns.tolist()
        self.n_assets = len(self.assets)
        
        # Store parameters
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        self.crisis_threshold = crisis_threshold
        self.diversification_bonus = diversification_bonus
        
        # Add crisis detection parameters
        self.crisis_window = 10  # Window for crisis detection
        self.crisis_memory = []  # Store recent market returns
        self.current_regime = 'normal'  # Track current market regime
        
        # Add momentum tracking
        self.momentum_window = 20
        self.returns_history = []
        
        # Debug logging for initialization
        logger.info(f"Initializing environment with lookback window: {self.lookback_window}")
        logger.info(f"Reward scaling: {self.reward_scaling}")
        logger.info(f"Crisis threshold: {self.crisis_threshold}")
        logger.info(f"Max position size: {self.max_position_size}")
        
        # Validate data alignment
        self._validate_data()
        
        # Calculate state and action space dimensions
        self._setup_spaces()
        
        # Initialize state
        self.reset()
    
    def _validate_data(self) -> None:
        """Validate that all data inputs are properly aligned."""
        # Check index alignment
        if not (self.price_data.index.equals(self.returns_data.index) and 
                self.returns_data.index.equals(self.features_data.index)):
            raise ValueError("Data indices must be aligned")
        
        # Check for missing values
        if self.price_data.isnull().any().any() or \
           self.returns_data.isnull().any().any() or \
           self.features_data.isnull().any().any():
            raise ValueError("Data contains missing values")
        
        logger.info("Data validation passed")
    
    def _setup_spaces(self) -> None:
        """Set up the observation and action spaces."""
        # Calculate state dimensions
        n_price_features = 2  # Close price and returns
        n_technical_features = self.features_data.shape[1]
        n_portfolio_features = 1 + self.n_assets  # Portfolio value and weights
        
        # State space: [lookback_window, n_assets * n_price_features + n_technical_features + n_portfolio_features]
        state_dim = (
            self.lookback_window,
            self.n_assets * n_price_features + n_technical_features + n_portfolio_features
        )
        
        # Action space: [n_assets] continuous values that sum to 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=state_dim,
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
    
    def _get_state(self) -> np.ndarray:
        """
        Construct the state observation.
        
        Returns:
            Array containing:
            - Historical prices and returns for each asset
            - Technical indicators
            - Current portfolio value and weights
        """
        # Get historical window
        start_idx = max(0, self.current_step - self.lookback_window + 1)
        end_idx = self.current_step + 1

        # Debug logging
        logger.debug(f"Current step: {self.current_step}, Lookback window: {self.lookback_window}")
        logger.debug(f"Start idx: {start_idx}, End idx: {end_idx}")

        # Initialize state array
        state = []

        # Add price and return data for each asset
        for asset in self.assets:
            # Get close prices and returns
            if isinstance(self.price_data.columns, pd.MultiIndex):
                prices = self.price_data[asset]['Close'].iloc[start_idx:end_idx].values
            else:
                prices = self.price_data[asset].iloc[start_idx:end_idx].values
            returns = self.returns_data[asset].iloc[start_idx:end_idx].values

            # Debug logging
            logger.debug(f"Asset {asset} - Prices shape: {prices.shape}, Returns shape: {returns.shape}")

            # Normalize prices to start at 1.0
            if len(prices) > 0:
                prices = prices / prices[0]

            # Pad with zeros if needed
            if len(prices) < self.lookback_window:
                pad_length = self.lookback_window - len(prices)
                prices = np.pad(prices, (pad_length, 0), mode='constant')
                returns = np.pad(returns, (pad_length, 0), mode='constant')
                logger.debug(f"Padded {asset} data to length {self.lookback_window}")

            # Ensure we have exactly lookback_window elements
            if len(prices) > self.lookback_window:
                prices = prices[-self.lookback_window:]
                returns = returns[-self.lookback_window:]
                logger.debug(f"Truncated {asset} data to length {self.lookback_window}")

            # Verify shapes
            assert len(prices) == self.lookback_window, f"Prices length {len(prices)} != lookback window {self.lookback_window}"
            assert len(returns) == self.lookback_window, f"Returns length {len(returns)} != lookback window {self.lookback_window}"

            # Reshape to 2D
            prices = prices.reshape(-1, 1)
            returns = returns.reshape(-1, 1)
            state.extend([prices, returns])

        # Add technical indicators (already 2D)
        tech_features = self.features_data.iloc[start_idx:end_idx].values
        logger.debug(f"Technical features shape before padding: {tech_features.shape}")

        if len(tech_features) < self.lookback_window:
            pad_length = self.lookback_window - len(tech_features)
            tech_features = np.pad(
                tech_features,
                ((pad_length, 0), (0, 0)),
                mode='constant'
            )
            logger.debug(f"Padded technical features to shape {tech_features.shape}")
        elif len(tech_features) > self.lookback_window:
            tech_features = tech_features[-self.lookback_window:]
            logger.debug(f"Truncated technical features to shape {tech_features.shape}")

        # Verify technical features shape
        assert tech_features.shape[0] == self.lookback_window, \
            f"Technical features rows {tech_features.shape[0]} != lookback window {self.lookback_window}"

        state.append(tech_features)

        # Add portfolio information
        portfolio_value = np.full(self.lookback_window, self.portfolio_value).reshape(-1, 1)
        portfolio_weights = np.tile(self.portfolio_weights, (self.lookback_window, 1))
        state.extend([portfolio_value, portfolio_weights])

        # Concatenate all features along axis=1
        state = np.concatenate(state, axis=1)

        # Final shape verification
        expected_shape = (self.lookback_window, self.n_assets * 2 + self.features_data.shape[1] + 1 + self.n_assets)
        assert state.shape == expected_shape, \
            f"State shape {state.shape} != expected shape {expected_shape}"

        logger.debug(f"Final state shape: {state.shape}")
        return state.astype(np.float32)
    
    def _calculate_reward(self, action: np.ndarray, info: Dict) -> float:
        """
        Calculate reward with enhanced risk management and crisis handling.
        
        Args:
            action: Portfolio weights
            info: Step information dictionary
            
        Returns:
            Reward value
        """
        # Normalize weights
        weights = self._normalize_weights(action)
        
        # Get portfolio metrics
        portfolio_return = info['portfolio_return']
        portfolio_value = info['portfolio_value']
        
        # Update crisis detection
        market_returns = self.returns_data.iloc[self.current_step].mean()
        self.crisis_memory.append(market_returns)
        if len(self.crisis_memory) > self.crisis_window:
            self.crisis_memory.pop(0)
        
        # Detect crisis based on rolling window
        crisis_score = np.mean(self.crisis_memory) if self.crisis_memory else 0
        is_crisis = crisis_score < self.crisis_threshold
        self.current_regime = 'crisis' if is_crisis else 'normal'
        
        # Calculate transaction costs with dynamic scaling
        prev_weights = self.portfolio_weights
        weight_changes = np.abs(weights - prev_weights)
        base_transaction_cost = self.transaction_cost * np.sum(weight_changes)
        
        # Scale transaction costs based on regime
        if is_crisis:
            transaction_cost = base_transaction_cost * 1.5  # Higher costs during crisis
        else:
            transaction_cost = base_transaction_cost
        
        # Calculate risk metrics using a longer window
        returns_window = 30
        if len(self.returns_history) >= returns_window:
            recent_returns = pd.Series(self.returns_history[-returns_window:])
            volatility = recent_returns.std() * np.sqrt(252)
            sharpe = recent_returns.mean() / recent_returns.std() * np.sqrt(252) if recent_returns.std() > 0 else 0
            drawdown = (recent_returns.cumsum() - recent_returns.cumsum().cummax()).min()
            
            # Calculate momentum
            momentum = recent_returns.mean() * np.sqrt(252)
        else:
            volatility = 0
            sharpe = 0
            drawdown = 0
            momentum = 0
        
        # Calculate crisis severity
        crisis_severity = max(0, -crisis_score / self.crisis_threshold)
        
        # Calculate diversification score (1 - Herfindahl-Hirschman Index)
        hhi = np.sum(weights ** 2)
        diversification_score = 1 - hhi
        
        # Calculate correlation penalty
        if len(self.returns_history) >= returns_window:
            asset_returns = self.returns_data.iloc[self.current_step - returns_window:self.current_step]
            portfolio_returns = pd.Series(self.returns_history[-returns_window:])
            correlations = asset_returns.apply(lambda x: x.corr(portfolio_returns))
            correlation_penalty = -0.1 * np.sum(weights * correlations.abs())
        else:
            correlation_penalty = 0
        
        # Base reward components
        return_reward = portfolio_return
        risk_penalty = -self.risk_penalty * volatility
        drawdown_penalty = -0.5 * abs(drawdown)  # Increased penalty for drawdowns
        transaction_penalty = transaction_cost
        diversification_reward = self.diversification_bonus * diversification_score
        
        # Crisis-specific adjustments
        if is_crisis:
            # Increase risk penalties during crises
            risk_penalty *= (1 + crisis_severity)
            drawdown_penalty *= (1 + crisis_severity)
            
            # Add defensive positioning reward
            defensive_reward = 0.2 * np.sum(weights * (1 - crisis_severity))
            
            # Add momentum penalty during crises
            momentum_penalty = -0.3 * np.sum(weights * (momentum < 0))
            
            # Add correlation penalty during crises
            correlation_penalty *= 2
            
            # Combine crisis-specific rewards
            crisis_reward = defensive_reward + momentum_penalty
        else:
            # Normal regime rewards
            momentum_reward = 0.1 * np.sum(weights * (momentum > 0))
            crisis_reward = momentum_reward
        
        # Add position limit penalties with dynamic scaling
        position_limit_penalty = -0.15 * np.sum(np.maximum(0, weights - self.max_position_size))
        min_position_penalty = -0.15 * np.sum(np.maximum(0, self.min_position_size - weights[weights > 0]))
        
        # Scale position penalties during crisis
        if is_crisis:
            position_limit_penalty *= 1.5
            min_position_penalty *= 1.5
        
        # Combine all reward components
        reward = (
            return_reward +
            risk_penalty +
            drawdown_penalty +
            transaction_penalty +
            diversification_reward +
            crisis_reward +
            correlation_penalty +
            position_limit_penalty +
            min_position_penalty
        ) * self.reward_scaling
        
        # Log reward components for debugging
        if self.current_step % 100 == 0:  # Log every 100 steps
            logger.info(f"\nStep {self.current_step} Reward Components:")
            logger.info(f"Current Regime: {self.current_regime}")
            logger.info(f"Crisis Severity: {crisis_severity:.4f}")
            logger.info(f"Portfolio Return: {portfolio_return:.4f}")
            logger.info(f"Sharpe Ratio: {sharpe:.4f}")
            logger.info(f"Volatility: {volatility:.4f}")
            logger.info(f"Drawdown: {drawdown:.4f}")
            logger.info(f"Transaction Cost: {transaction_cost:.4f}")
            logger.info(f"Diversification Score: {diversification_score:.4f}")
            logger.info(f"Position Weights: {dict(zip(self.assets, weights))}")
            logger.info(f"Total Reward: {reward:.4f}")
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if the episode is done."""
        return self.current_step >= len(self.returns_data) - 1
    
    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:
        weights = np.array(action, dtype=np.float32)
        weights = np.maximum(weights, 0)

        # Enforce min and max position size
        weights = np.where(weights > 0, np.maximum(weights, self.min_position_size), 0)
        weights = np.minimum(weights, self.max_position_size)

        # If all weights are zero, assign equal weights
        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / len(weights)

        # Normalize to sum to 1
        weights = weights / np.sum(weights)

        # After normalization, clip again to max and renormalize if needed
        weights = np.minimum(weights, self.max_position_size)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / np.sum(weights)

        # Final check for min position size
        weights = np.where(weights > 0, np.maximum(weights, self.min_position_size), 0)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / np.sum(weights)

        return weights
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Raw action from the agent
            
        Returns:
            observation: New state
            reward: Reward for the step
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Normalize weights
        normalized_action = self._normalize_weights(action)
        
        # Calculate current returns and portfolio return
        current_returns = self.returns_data.iloc[self.current_step].values
        portfolio_return = np.sum(current_returns * normalized_action)
        
        # Calculate reward using normalized weights and current portfolio return
        reward = self._calculate_reward(
            normalized_action,
            {
                'portfolio_return': portfolio_return,
                'portfolio_value': self.portfolio_value,
                'current_regime': None  # Placeholder, can be set if regime logic is available
            }
        )
        
        # Update portfolio
        self.portfolio_weights = normalized_action
        self.portfolio_value *= (1 + portfolio_return)
        
        # Optionally, keep a history of returns for risk metrics
        if not hasattr(self, 'returns_history'):
            self.returns_history = []
        self.returns_history.append(portfolio_return)
        
        # Move to next step
        self.current_step += 1
        
        # Get new state
        observation = self._get_state()
        
        # Check if done
        done = self._is_done()
        
        # Additional information
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_weights': self.portfolio_weights,
            'step': self.current_step
        }
        
        return observation, reward, done, False, info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset portfolio
        self.portfolio_value = self.initial_balance
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets  # Equal weight
        self.current_step = self.lookback_window - 1  # Start after lookback window
        
        # Get initial state
        observation = self._get_state()
        
        # Additional information
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_weights': self.portfolio_weights,
            'step': self.current_step
        }
        
        return observation, info
    
    def render(self):
        """Render the current state (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources (not implemented)."""
        pass

def main():
    """Example usage of the trading environment."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader(asset_class='traditional', regime='training')
    price_data, returns_data, features_data = loader.load_data()
    
    if price_data is None:
        loader.download_data()
        loader.calculate_returns()
        loader.calculate_features()
        price_data, returns_data, features_data = loader.load_data()
    
    # Create environment
    env = PortfolioTradingEnv(
        price_data=price_data,
        returns_data=returns_data,
        features_data=features_data
    )
    
    # Test environment
    obs, info = env.reset()
    print(f"Initial state shape: {obs.shape}")
    print(f"Initial portfolio value: ${info['portfolio_value']:,.2f}")
    
    # Run a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"\nStep {info['step']}")
        print(f"Portfolio value: ${info['portfolio_value']:,.2f}")
        print(f"Reward: {reward:.4f}")
        print(f"Done: {done}")
        
        if done:
            break

if __name__ == '__main__':
    main() 