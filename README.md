# Walk-Forward Random Forest Trading Strategy

A quantitative trading strategy combining **machine learning prediction** with **multi-layer structural risk control**, backtested on China A-share equities with Monte Carlo permutation testing for statistical validation.

## Strategy Overview

This system uses a Walk-Forward Rolling Random Forest to predict short-term price movements, filtered through five independent risk control layers before execution. The core philosophy: **only trade the highest-conviction setups, and manage exits with asymmetric profit-taking.**

### Architecture

```
Market Data → Feature Engineering (16 indicators)
                      ↓
              Walk-Forward RF Training (500-day window, 60-day step)
                      ↓
              5-Layer Entry Filter → Position Sizing by Confidence
                      ↓
              Multi-Stage Exit Engine (Stop Loss / Trailing TP / BBI Ladder)
                      ↓
              Monte Carlo Permutation Test (200 shuffles, p-value)
```

## Key Components

### 1. Feature Engineering
16 technical features across four categories:
- **Momentum**: Lagged returns (1/2/3 day)
- **Mean Reversion**: RSI-14, KDJ-J oscillator
- **Trend Deviation**: Price deviation from MA5/10/60, EMA13, BBI, Bull-Bear Line
- **Market Structure**: Volume change, amplitude, volatility, MA crossover

### 2. Walk-Forward Rolling Training
- Trains on the most recent **500 trading days**, predicts the next **60 days**
- Retrains every 60 days to adapt to market regime changes (concept drift)
- Avoids look-ahead bias — each prediction uses only past data

### 3. Entry Logic — Five-Layer Filter
All five conditions must be satisfied simultaneously:
1. **ML Signal**: Random Forest predicts bullish (y_pred = 1)
2. **Macro Trend**: MA120 slope > 0 (long-term uptrend)
3. **Sentiment Extreme**: KDJ J-value < 20 (panic selling / oversold)
4. **Strength Confirmation**: Price ≥ Bull-Bear Line
5. **Cooldown**: ≥ 120 trading days since last exit

### 4. Exit Logic — "Fishing Strategy"
Asymmetric exit design: **catch small fish quickly, let big fish run.**

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 (Highest) | Price < Bull-Bear Line | Forced stop loss |
| 2 | Price < Entry-day low − 0.05 | Entry stop loss |
| 3 | Unrealized gain < 3% AND model turns bearish | Take profit (small fish) |
| 4 | Unrealized gain ≥ 3% AND price drops 5% from peak | Trailing stop (big fish) |
| 5 | Price > BBI + 3% AND big bullish candle (>2%) | Reduce position by 50% |

### 5. Dynamic Position Sizing
Position size scales with model confidence:
```
Confidence 50% → 0% position (no trade)
Confidence 60% → 33% position
Confidence 70% → 67% position  
Confidence 80% → 100% position (full)
```

### 6. Monte Carlo Statistical Validation
- **Null Hypothesis**: ML predictions add no value; returns come purely from risk control rules
- **Method**: Shuffle prediction signals 200 times while keeping risk indicators in original order
- **Interpretation**: p < 0.05 → ML signal significantly outperforms random noise

## Results

Backtested across multiple A-share equities spanning AI/semiconductor, new energy, defense, and consumer sectors. Strategy demonstrates:
- Positive risk-adjusted returns (Sharpe > 0) on majority of tested assets
- Statistically significant ML signal contribution (p < 0.05) on select assets
- Maximum drawdown consistently lower than buy-and-hold benchmark

## Requirements

```
Python 3.10+
pandas
numpy
matplotlib
scikit-learn
yfinance
```

## Usage

```python
from strategy import StrategyConfig, main

# Run with default configuration (8 stocks)
main()

# Or customize parameters
config = StrategyConfig(
    train_window=500,
    retrain_every=60,
    n_shuffles=200,
    cooldown_days=120,
)
```

## Project Structure

```
├── strategy.py          # Core strategy: features, training, backtest, validation
├── README.md
├── LICENSE              # MIT License
└── .gitignore
```

## Academic Context

This project implements a research pipeline for an upcoming SSRN working paper exploring the intersection of non-linear ML predictions and structural risk management rules in low signal-to-noise financial markets.

## License

MIT License — see [LICENSE](LICENSE) for details.
