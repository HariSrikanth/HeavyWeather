"""
Test script to verify the trading environment changes.
"""

import numpy as np
import pandas as pd
from trading_env import PortfolioTradingEnv
from data_loader import DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_weight_normalization():
    """Test the weight normalization function."""
    # Load minimal data for environment creation
    loader = DataLoader(asset_class='traditional', regime='training')
    price_data, returns_data, features_data = loader.load_data()
    
    if price_data is None:
        loader.download_data()
        loader.calculate_returns()
        loader.calculate_features()
        price_data, returns_data, features_data = loader.load_data()
    
    # Create environment with new parameters
    env = PortfolioTradingEnv(
        price_data=price_data,
        returns_data=returns_data,
        features_data=features_data,
        max_position_size=0.4,
        min_position_size=0.05,
        risk_penalty=0.5,
        transaction_cost=0.001,
        lookback_window=20,
        crisis_threshold=-0.02,
        diversification_bonus=0.1
    )
    
    # Test cases for weight normalization
    test_cases = [
        # Test 1: All equal weights
        np.ones(env.n_assets),
        
        # Test 2: Single large weight
        np.array([0.9] + [0.1/(env.n_assets-1)] * (env.n_assets-1)),
        
        # Test 3: Multiple small weights
        np.array([0.01] * env.n_assets),
        
        # Test 4: Zero weights
        np.zeros(env.n_assets),
        
        # Test 5: Negative weights
        np.array([-0.5, 0.5, 0.5, 0.5])
    ]
    
    logger.info("\nTesting weight normalization:")
    for i, test_case in enumerate(test_cases, 1):
        normalized = env._normalize_weights(test_case)
        logger.info(f"\nTest case {i}:")
        logger.info(f"Input weights: {test_case}")
        logger.info(f"Normalized weights: {normalized}")
        logger.info(f"Sum of weights: {np.sum(normalized):.6f}")
        logger.info(f"Max weight: {np.max(normalized):.6f}")
        logger.info(f"Min non-zero weight: {np.min(normalized[normalized > 0]):.6f}")
        
        # Verify constraints
        assert np.all(normalized >= 0), f"Test {i}: Found negative weights"
        assert abs(np.sum(normalized) - 1.0) < 1e-6, f"Test {i}: Weights don't sum to 1"
        assert np.max(normalized) <= env.max_position_size, f"Test {i}: Max weight exceeds limit"
        non_zero = normalized[normalized > 0]
        if len(non_zero) > 0:
            assert np.min(non_zero) >= env.min_position_size, f"Test {i}: Non-zero weight below minimum"

def test_reward_calculation():
    """Test the reward calculation with different scenarios."""
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
        features_data=features_data,
        max_position_size=0.4,
        min_position_size=0.05,
        risk_penalty=0.5,
        transaction_cost=0.001,
        lookback_window=20,
        crisis_threshold=-0.02,
        diversification_bonus=0.1
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Test cases for reward calculation
    test_cases = [
        # Test 1: Equal weights
        np.ones(env.n_assets) / env.n_assets,
        
        # Test 2: Concentrated position
        np.array([0.9] + [0.1/(env.n_assets-1)] * (env.n_assets-1)),
        
        # Test 3: Defensive position (high cash)
        np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # Assuming last asset is cash
    ]
    
    logger.info("\nTesting reward calculation:")
    for i, test_case in enumerate(test_cases, 1):
        # Step environment
        obs, reward, done, truncated, info = env.step(test_case)
        
        logger.info(f"\nTest case {i}:")
        logger.info(f"Action weights: {test_case}")
        logger.info(f"Portfolio weights: {info['portfolio_weights']}")
        logger.info(f"Portfolio value: ${info['portfolio_value']:,.2f}")
        logger.info(f"Reward: {reward:.6f}")
        
        # Verify reward components
        assert isinstance(reward, float), f"Test {i}: Reward is not a float"
        assert not np.isnan(reward), f"Test {i}: Reward is NaN"
        assert not np.isinf(reward), f"Test {i}: Reward is infinite"

def test_crisis_handling():
    """Test crisis detection and handling."""
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
        features_data=features_data,
        max_position_size=0.4,
        min_position_size=0.05,
        risk_penalty=0.5,
        transaction_cost=0.001,
        lookback_window=20,
        crisis_threshold=-0.02,
        diversification_bonus=0.1
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Find a crisis period (e.g., 2008)
    crisis_start = returns_data.index.get_loc('2008-09-01')
    crisis_end = returns_data.index.get_loc('2009-03-31')
    
    # Test normal period
    logger.info("\nTesting normal period:")
    env.current_step = crisis_start - 100  # Before crisis
    normal_action = np.ones(env.n_assets) / env.n_assets
    obs, normal_reward, done, truncated, info = env.step(normal_action)
    logger.info(f"Normal period reward: {normal_reward:.6f}")
    
    # Test crisis period
    logger.info("\nTesting crisis period:")
    env.current_step = crisis_start
    crisis_action = np.ones(env.n_assets) / env.n_assets
    obs, crisis_reward, done, truncated, info = env.step(crisis_action)
    logger.info(f"Crisis period reward: {crisis_reward:.6f}")
    
    # Verify crisis handling
    assert crisis_reward < normal_reward, "Crisis reward should be lower than normal period reward"

def main():
    """Run all tests."""
    logger.info("Starting environment tests...")
    
    try:
        test_weight_normalization()
        logger.info("\nWeight normalization tests passed!")
        
        test_reward_calculation()
        logger.info("\nReward calculation tests passed!")
        
        test_crisis_handling()
        logger.info("\nCrisis handling tests passed!")
        
        logger.info("\nAll tests passed successfully!")
        
    except AssertionError as e:
        logger.error(f"\nTest failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"\nUnexpected error: {str(e)}")
        raise

if __name__ == '__main__':
    main() 