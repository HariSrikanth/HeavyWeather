"""
Implementation of baseline portfolio strategies for comparison.
Includes equal-weight, 60/40, and Black-Scholes-inspired hedging strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy import stats
import empyrical as ep
import scipy.optimize as sco

from config import metrics_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics."""
    returns: pd.Series
    weights: pd.DataFrame
    cagr: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    omega_ratio: float
    tail_ratio: float

class BaseStrategy:
    """Base class for portfolio strategies."""
    
    def __init__(
        self,
        returns_data: pd.DataFrame,
        rebalance_freq: str = 'daily',
        transaction_cost: float = 0.001
    ):
        """
        Initialize the strategy.
        
        Args:
            returns_data: DataFrame of asset returns
            rebalance_freq: Rebalancing frequency ('daily' or 'monthly')
            transaction_cost: Cost per trade (as a fraction)
        """
        self.returns_data = returns_data
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.assets = returns_data.columns.tolist()
        self.n_assets = len(self.assets)
    
    def _calculate_metrics(self, returns: pd.Series, weights: pd.DataFrame) -> PortfolioMetrics:
        """Calculate portfolio performance metrics."""
        return PortfolioMetrics(
            returns=returns,
            weights=weights,
            cagr=ep.cagr(returns),
            volatility=ep.annual_volatility(returns),
            sharpe_ratio=ep.sharpe_ratio(returns),
            sortino_ratio=ep.sortino_ratio(returns),
            max_drawdown=ep.max_drawdown(returns),
            calmar_ratio=ep.calmar_ratio(returns),
            omega_ratio=ep.omega_ratio(returns),
            tail_ratio=ep.tail_ratio(returns)
        )
    
    def _apply_rebalancing(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Apply rebalancing frequency to weights."""
        if self.rebalance_freq == 'daily':
            return weights
        elif self.rebalance_freq == 'monthly':
            # Resample to monthly frequency, forward fill
            monthly_weights = weights.resample('M').first().ffill()
            # Reindex to daily frequency
            return monthly_weights.reindex(weights.index, method='ffill')
        else:
            raise ValueError(f"Unknown rebalancing frequency: {self.rebalance_freq}")
    
    def _calculate_transaction_costs(
        self,
        weights: pd.DataFrame,
        portfolio_values: pd.Series
    ) -> pd.Series:
        """Calculate transaction costs for rebalancing."""
        # Calculate weight changes
        weight_changes = weights.diff().abs()
        
        # Calculate transaction costs
        costs = weight_changes.sum(axis=1) * self.transaction_cost
        
        # Apply costs to portfolio values
        return portfolio_values * (1 - costs)
    
    def run(self) -> PortfolioMetrics:
        """Run the strategy and return performance metrics."""
        raise NotImplementedError

class EqualWeightStrategy(BaseStrategy):
    """Equal-weight portfolio strategy."""
    
    def run(self) -> PortfolioMetrics:
        """Run equal-weight strategy."""
        # Create equal weights
        weights = pd.DataFrame(
            np.ones((len(self.returns_data), self.n_assets)) / self.n_assets,
            index=self.returns_data.index,
            columns=self.assets
        )
        
        # Apply rebalancing frequency
        weights = self._apply_rebalancing(weights)
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns_data * weights).sum(axis=1)
        
        # Calculate portfolio values
        portfolio_values = (1 + portfolio_returns).cumprod()
        
        # Apply transaction costs
        portfolio_values = self._calculate_transaction_costs(weights, portfolio_values)
        
        # Recalculate returns after costs
        returns = portfolio_values.pct_change().dropna()
        
        return self._calculate_metrics(returns, weights)

class SixtyFortyStrategy(BaseStrategy):
    """60/40 portfolio strategy (stocks/bonds)."""
    
    def __init__(
        self,
        returns_data: pd.DataFrame,
        stock_asset: str = 'SPY',
        bond_asset: str = 'TLT',
        stock_weight: float = 0.6,
        rebalance_freq: str = 'daily',
        transaction_cost: float = 0.001
    ):
        """
        Initialize 60/40 strategy.
        
        Args:
            returns_data: DataFrame of asset returns
            stock_asset: Stock asset ticker
            bond_asset: Bond asset ticker
            stock_weight: Weight in stocks (default: 0.6)
            rebalance_freq: Rebalancing frequency
            transaction_cost: Cost per trade
        """
        super().__init__(returns_data, rebalance_freq, transaction_cost)
        
        if stock_asset not in self.assets or bond_asset not in self.assets:
            raise ValueError(f"Required assets {stock_asset} and/or {bond_asset} not found")
        
        self.stock_asset = stock_asset
        self.bond_asset = bond_asset
        self.stock_weight = stock_weight
        self.bond_weight = 1 - stock_weight
    
    def run(self) -> PortfolioMetrics:
        """Run 60/40 strategy."""
        # Create weights
        weights = pd.DataFrame(
            0.0,
            index=self.returns_data.index,
            columns=self.assets
        )
        weights[self.stock_asset] = self.stock_weight
        weights[self.bond_asset] = self.bond_weight
        
        # Apply rebalancing frequency
        weights = self._apply_rebalancing(weights)
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns_data * weights).sum(axis=1)
        
        # Calculate portfolio values
        portfolio_values = (1 + portfolio_returns).cumprod()
        
        # Apply transaction costs
        portfolio_values = self._calculate_transaction_costs(weights, portfolio_values)
        
        # Recalculate returns after costs
        returns = portfolio_values.pct_change().dropna()
        
        return self._calculate_metrics(returns, weights)

class BlackScholesHedgingStrategy(BaseStrategy):
    """
    Black-Scholes-inspired hedging strategy.
    Uses option-like hedging based on volatility and market conditions.
    """
    
    def __init__(
        self,
        returns_data: pd.DataFrame,
        volatility_data: pd.DataFrame,
        stock_asset: str = 'SPY',
        hedge_asset: str = 'VIXY',
        base_weight: float = 0.6,
        vol_threshold: float = 0.2,
        rebalance_freq: str = 'daily',
        transaction_cost: float = 0.001
    ):
        """
        Initialize Black-Scholes hedging strategy.
        
        Args:
            returns_data: DataFrame of asset returns
            volatility_data: DataFrame of volatility measures
            stock_asset: Primary stock asset
            hedge_asset: Volatility hedge asset
            base_weight: Base weight in stocks
            vol_threshold: Volatility threshold for hedging
            rebalance_freq: Rebalancing frequency
            transaction_cost: Cost per trade
        """
        super().__init__(returns_data, rebalance_freq, transaction_cost)
        
        if stock_asset not in self.assets or hedge_asset not in self.assets:
            raise ValueError(f"Required assets {stock_asset} and/or {hedge_asset} not found")
        
        self.stock_asset = stock_asset
        self.hedge_asset = hedge_asset
        self.base_weight = base_weight
        self.vol_threshold = vol_threshold
        self.volatility_data = volatility_data
    
    def _calculate_hedge_ratio(self, volatility: float) -> float:
        """
        Calculate hedge ratio based on volatility.
        Uses a sigmoid function to smoothly transition between hedged and unhedged states.
        """
        # Sigmoid function for smooth transition
        x = (volatility - self.vol_threshold) * 10  # Scale for sharper transition
        hedge_ratio = 1 / (1 + np.exp(-x))
        
        # Scale to reasonable range (0 to 0.4)
        return hedge_ratio * 0.4
    
    def run(self) -> PortfolioMetrics:
        """Run Black-Scholes hedging strategy."""
        # Initialize weights
        weights = pd.DataFrame(
            0.0,
            index=self.returns_data.index,
            columns=self.assets
        )
        
        # Calculate hedge ratios based on volatility
        hedge_ratios = self.volatility_data[self.stock_asset].apply(self._calculate_hedge_ratio)
        
        # Set weights
        weights[self.stock_asset] = self.base_weight * (1 - hedge_ratios)
        weights[self.hedge_asset] = self.base_weight * hedge_ratios
        weights['CASH'] = 1 - weights.sum(axis=1)  # Remaining in cash
        
        # Apply rebalancing frequency
        weights = self._apply_rebalancing(weights)
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns_data * weights).sum(axis=1)
        
        # Calculate portfolio values
        portfolio_values = (1 + portfolio_returns).cumprod()
        
        # Apply transaction costs
        portfolio_values = self._calculate_transaction_costs(weights, portfolio_values)
        
        # Recalculate returns after costs
        returns = portfolio_values.pct_change().dropna()
        
        return self._calculate_metrics(returns, weights)

class MeanVarianceStrategy(BaseStrategy):
    """Mean-variance (Markowitz) portfolio optimization strategy."""
    def run(self) -> PortfolioMetrics:
        # Calculate mean and covariance of returns
        mu = self.returns_data.mean() * 252  # annualized mean
        cov = self.returns_data.cov() * 252  # annualized covariance
        n = self.n_assets
        # Objective: maximize Sharpe ratio (risk-free rate from metrics_config)
        def neg_sharpe(weights):
            port_return = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            sharpe = (port_return - metrics_config.risk_free_rate) / (port_vol + 1e-8)
            return -sharpe
        # Constraints: weights sum to 1, weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        x0 = np.ones(n) / n
        result = sco.minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints)
        opt_weights = result.x
        # Create weights DataFrame (constant weights)
        weights = pd.DataFrame(
            np.tile(opt_weights, (len(self.returns_data), 1)),
            index=self.returns_data.index,
            columns=self.assets
        )
        # Apply rebalancing frequency
        weights = self._apply_rebalancing(weights)
        # Calculate portfolio returns
        portfolio_returns = (self.returns_data * weights).sum(axis=1)
        # Calculate portfolio values
        portfolio_values = (1 + portfolio_returns).cumprod()
        # Apply transaction costs
        portfolio_values = self._calculate_transaction_costs(weights, portfolio_values)
        # Recalculate returns after costs
        returns = portfolio_values.pct_change().dropna()
        return self._calculate_metrics(returns, weights)

def run_all_strategies(
    returns_data: pd.DataFrame,
    volatility_data: Optional[pd.DataFrame] = None,
    rebalance_freq: str = 'daily',
    transaction_cost: float = 0.001
) -> Dict[str, PortfolioMetrics]:
    """
    Run all baseline strategies and return their performance metrics.
    
    Args:
        returns_data: DataFrame of asset returns
        volatility_data: DataFrame of volatility measures (for Black-Scholes)
        rebalance_freq: Rebalancing frequency
        transaction_cost: Cost per trade
        
    Returns:
        Dictionary of strategy names to performance metrics
    """
    strategies = {
        'equal_weight': EqualWeightStrategy(
            returns_data=returns_data,
            rebalance_freq=rebalance_freq,
            transaction_cost=transaction_cost
        ),
        'sixty_forty': SixtyFortyStrategy(
            returns_data=returns_data,
            rebalance_freq=rebalance_freq,
            transaction_cost=transaction_cost
        ),
        'mean_variance': MeanVarianceStrategy(
            returns_data=returns_data,
            rebalance_freq=rebalance_freq,
            transaction_cost=transaction_cost
        )
    }
    
    # Add Black-Scholes strategy if volatility data is available
    if volatility_data is not None:
        strategies['black_scholes'] = BlackScholesHedgingStrategy(
            returns_data=returns_data,
            volatility_data=volatility_data,
            rebalance_freq=rebalance_freq,
            transaction_cost=transaction_cost
        )
    
    # Run all strategies
    results = {}
    for name, strategy in strategies.items():
        try:
            results[name] = strategy.run()
            logger.info(f"Successfully ran {name} strategy")
        except Exception as e:
            logger.error(f"Error running {name} strategy: {str(e)}")
    
    return results

def main():
    """Example usage of baseline strategies."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader(asset_class='traditional', regime='testing')
    price_data, returns_data, features_data = loader.load_data()
    
    if price_data is None:
        loader.download_data()
        loader.calculate_returns()
        loader.calculate_features()
        price_data, returns_data, features_data = loader.load_data()
    
    # Extract volatility data
    volatility_data = features_data.filter(regex='_volatility$')
    
    # Run strategies
    results = run_all_strategies(
        returns_data=returns_data,
        volatility_data=volatility_data,
        rebalance_freq='daily',
        transaction_cost=0.001
    )
    
    # Print results
    print("\nStrategy Performance Summary")
    print("=" * 50)
    
    metrics_to_print = [
        'cagr', 'volatility', 'sharpe_ratio', 'sortino_ratio',
        'max_drawdown', 'calmar_ratio', 'omega_ratio', 'tail_ratio'
    ]
    
    for name, metrics in results.items():
        print(f"\n{name.replace('_', ' ').title()}")
        print("-" * 30)
        for metric in metrics_to_print:
            value = getattr(metrics, metric)
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

if __name__ == '__main__':
    main() 