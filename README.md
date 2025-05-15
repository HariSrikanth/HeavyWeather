# HeavyWeather: RL-Driven Portfolio Allocation under Regime Shifts

In 2020, Oil futures went into the negative, and hundreds of millions were lost. Crisis scenarios challenge existing models, and thats why I've experimented in building Heavy Weather, a reinforcement learning system for dynamic portfolio allocation across traditional, futures, and high-volatility asset classes. This project implements a Proximal Policy Optimization (PPO) agent that adapts to different market regimes and is benchmarked against traditional portfolio strategies.

## 📄 Research Paper

[📘 Heavy Weather: PPO–Based Reinforcement Learning for Adaptive Portfolio Management](./Heavy_Weather__PPO_Based_RL_for_Multi_Asset_Trading.pdf)

## 🎯 Features

- **Advanced RL Agent**: PPO-based portfolio allocation agent with adaptive learning
- **Multi-Asset Support**: Comprehensive coverage across:
  - Traditional assets (equities, bonds, gold)
  - Futures proxies (oil, volatility, currency)
  - High-volatility assets (crypto, tech stocks)
- **Regime-Aware Trading**: Specialized testing across market regimes:
  - Global Financial Crisis (2007-2009)
  - COVID Crash (2020)
  - Tech/Crypto Crash (2021-2023)
- **Robust Evaluation**: Comprehensive benchmarking against:
  - Equal-weight portfolio
  - 60/40 strategy
  - Black-Scholes-inspired hedging
- **Technical Features**:
  - Dynamic position sizing
  - Crisis detection and adaptation
  - Transaction cost modeling
  - Diversification bonuses
  - Momentum tracking
  - Volatility-based risk management

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heavyWeather.git
cd heavyWeather
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
project_root/
├── data_loader.py              # Data download and preprocessing
├── trading_env.py              # Custom Gymnasium environment
├── train_agent.py              # PPO training pipeline
├── evaluate_agent.py           # Performance evaluation
├── baselines.py                # Benchmark strategies
├── performance_metrics.py      # Return and risk metrics
├── stat_tests.py              # Statistical validation
├── config.py                   # Configuration parameters
├── notebooks/                  # Analysis notebooks
├── models/                     # Saved model weights
└── logs/                       # Training logs
```

## 🚀 Usage

### 1. Data Collection
Download and preprocess market data:
```bash
python data_loader.py --asset-class traditional --start-date 2003-01-01 --end-date 2023-12-31
```

### 2. Training
Train the PPO agent:
```bash
python train_agent.py --asset-class traditional --regime training --timesteps 200000
```

Optional training parameters:
- `--eval-freq`: Evaluation frequency (default: 10000)
- `--n-eval-episodes`: Episodes per evaluation (default: 5)
- `--init-meanvar`: Initialize with mean-variance weights
- `--seed`: Random seed for reproducibility

### 3. Evaluation
Evaluate trained model:
```bash
python evaluate_agent.py --model-path models/ppo_traditional --asset-class traditional --regime testing
```

Optional evaluation parameters:
- `--n-episodes`: Number of evaluation episodes
- `--save-plots`: Save performance plots
- `--random-policy`: Test with random actions

## 📊 Asset Classes

### Traditional Assets
- SPY (S&P 500 ETF)
- TLT (20+ Year Treasury Bond ETF)
- GLD (Gold ETF)
- SHY (Short Treasury Bond ETF)

### Futures Proxies
- USO (Crude Oil ETF)
- VIXY/VXX (Volatility ETFs)
- UUP (USD Index ETF)

### High-Volatility Assets
- BTC-USD (Bitcoin)
- ETH-USD (Ethereum)
- ARKK (ARK Innovation ETF)
- TSLA (Tesla)
- COIN (Coinbase)

## 📈 Market Regimes

| Regime             | Purpose                    | Dates            | Description                    |
|--------------------|----------------------------|------------------|--------------------------------|
| GFC (2007-2009)    | Stress testing            | 2007-2009        | Global Financial Crisis        |
| COVID Crash        | Volatility testing        | Jan-Jun 2020     | Pandemic market crash          |
| Tech/Crypto Crash  | Speculation testing       | 2021-2023        | Tech/crypto market correction  |
| Training           | Model development         | 2003-2019        | Normal market conditions       |
| Testing            | OOS validation            | 2020-2023        | Recent market performance      |

## 🧪 Evaluation Metrics

### Return Metrics
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Sortino Ratio
- Omega Ratio
- Tail Ratio

### Risk Metrics
- Volatility
- Maximum Drawdown
- Calmar Ratio
- Value at Risk (VaR)

### Statistical Validation
- Welch's t-test
- Bootstrap confidence intervals
- Regime consistency analysis
- Crisis period performance

## 🔧 Configuration

Key parameters in `config.py`:
- Environment settings (lookback window, transaction costs)
- Trading constraints (position limits, rebalancing)
- Technical indicators (RSI, MACD, volatility)
- Reward function parameters
- Crisis detection thresholds

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📚 References

- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"
- Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"
- Modern Portfolio Theory (Markowitz, 1952)
- Black-Scholes Option Pricing Model 
