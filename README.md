# Multi-Asset Walk-Forward Random Forest Portfolio Rotation Strategy
## (多资产海选轮动与多层级动态风险管理量化系统)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Quant Grade](https://img.shields.io/badge/Grade-Institutional-success.svg)](#)

An institutional-grade quantitative backtesting framework combining rolling **non-linear Machine Learning forecasts** with **multi-frequency structural risk controls**, **portfolio-level shared-capital rotation**, and **Monte Carlo statistical validation**.

本项目实现了一个“预测交给机器学习，风控留给结构化规则”的动态组合轮动交易系统。

---

## 📈 1. System Architecture (系统架构)

The V4.0 system manages a shared-capital pool, executing cross-sectional scanning and ML confidence-based portfolio rotation on a daily scale.

```
       [Market Data Pool] → Feature Engineering (16 overlays)
                                      ↓
                     [Walk-Forward Rolling RF Training]
                      (500-day window, 60-day step)
                        * Leak-free Train/Test slice
                                      ↓
                     [Cross-Sectional Rank & Screening]
                        * ML Probability Sorting
                        * 5-Layer Entry Filters
                                      ↓
                     [Shared Capital Pool Allocation]
                      (CNY 1,000,000, Max 4 Holdings)
                      (Single Stock Limit: 25% Weight)
                                      ↓
                     [Multi-Stage Asymmetric Exit Engine]
                      * 2% Adaptive Stop-Loss (Price-scaled)
                      * Trailing Stop-Profit (Big fish)
                      * BBI-Deviation Position Halving
                                      ↓
                     [Monte Carlo Permutation Validation]
```

---

## 🚨 2. Version History & Quantitative Audits (版本迭代与重大审计日志)

| Version | Release Date | Key Upgrades & Architectural Changes | Quant & Mathematical Audits (量化与数学审计) |
| :--- | :---: | :--- | :--- |
| **v1.0 - v3.0** | *Historical* | Single-Asset backtester using fixed price tick stop-losses. | ⚠️ **High-Risk Vulnerabilities Identified:**<br>1. **Look-ahead data leakage** in rolling walk-forward training windows.<br>2. **Serial return compounding error** in daily return updates.<br>3. **Stop-loss scale bias** on high-priced assets. |
| **v4.0 (Current)** | **2026-05-23** | 1. Upgraded to **Multi-Asset Shared Capital Pool Rotation**.<br>2. Implemented **2% Adaptive Stop-Loss** based on purchase-day low.<br>3. Integrated **ProcessPoolExecutor** for parallel computing.<br>4. Added **LaTeX Booktabs Table** auto-generation. | ✅ **Audit Resolution Passed:**<br>1. **Leakage Purged**: Sliced training window by subtracting `future_return_days` (5 days).<br>2. **Parallel Return Update**: Summed weighted returns first, updating capital pool once daily: `capital = capital * (1 + sum(daily_returns * weights))`.<br>3. **Dynamic wind-risk**: Price-scaled 2% stop loss, eliminating fixed tick bias. |

---

## 🛠️ 3. Key Components (核心功能组件)

### A. Feature Engineering (16 Features)
*   **Momentum & Lags**: Lagged returns (1/2/3 day) to capture short-term memory.
*   **Mean Reversion**: RSI-14, KDJ-J oscillator detecting overbought/oversold boundaries.
*   **Trend Deviation**: Price ratios relative to MA5/10/60, EMA13, BBI, and Bull-Bear line.
*   **Market Structure**: Day-on-day volume changes, price amplitude, rolling volatility.

### B. Leak-Free Walk-Forward ML Engine
To adapt to market regime drift without overfitting, models are retrained every **60 days** using a rolling **500-day window**. 
We strictly enforce a **5-day gap** at the end of each training slice to ensure the target label (future 5-day return) does not leak test-set closing prices into the training set:
```python
# Purged training window slicing
X_train = X[train_end - train_window : train_end - config.future_return_days]
y_train = y[train_end - train_window : train_end - config.future_return_days]
```

### C. 5-Layer Cross-Sectional Entry Filter
For an asset to enter the rotation pool, it must pass five concurrent tests:
1.  **ML Signal**: Random Forest predicts positive direction (`y_pred == 1`).
2.  **Long-term Trend**: MA120 slope > 0 (filter out structural bear markets).
3.  **Fear Regime**: KDJ J-value < 20 (exploit panic oversold regimes).
4.  **Regime Strength**: Closing price $\ge$ Bull-Bear Line.
5.  **Risk Cooldown**: At least 120 trading days have elapsed since the last liquidation of this asset.

### D. Asymmetric exit Mechanics
We operate on an asymmetric design: **catch small swing profits quickly, let large-trend profits run.**

| Exit Level | Trigger Condition | Action | Rationale |
| :---: | :--- | :---: | :--- |
| **Level 1 (Forced)** | Price < Bull-Bear Line | Liquidate | Structural trend breakdown |
| **Level 2 (Risk)** | Price < Purchase-Day Low * (1 - 2%) | Liquidate | **2% Adaptive Stop-Loss** (no scale bias) |
| **Level 3 (Swing)** | Floating profit < 3% AND ML turns bearish | Liquidate | Model-driven exit (lock minor gains) |
| **Level 4 (Trend)** | Floating profit $\ge$ 3% AND price drops 5% from peak | Liquidate | Trailing Stop-Profit (catch big trends) |
| **Level 5 (Scale)** | Price $\ge$ BBI + 3% AND daily bullish candle $\ge$ 2% | Halve Position | BBI profit-ladder reduction |

---

## 📊 4. Performance & Validation (绩效与统计验证)

### Dynamic Position Sizing
Single-stock portfolio allocation dynamically scales with ML probability outputs (confidence):
$$\text{Weight} = \min\left( \max\left( \text{Min\_Size}, (P_{\text{ML}} - 0.5) \times \text{Scale} \right) \times \text{Max\_Weight\_Per\_Stock}, \text{Remaining\_Capacity} \right)$$
This limits capital allocation during low-conviction periods and focuses leverage on high-probability setups.

### Monte Carlo Permutation Test
To prove that our ML models add real value beyond the structural risk rules, we run a **Permutation Test**:
1.  Shuffle ML prediction signals randomly while keeping raw technical indicators aligned.
2.  Run the full backtest 50+ times to construct a randomized return distribution.
3.  Compute $p$-value. A $p$-value < 0.05 rejects the null hypothesis that ML adds no value.

---

## 💻 5. Usage & Local Run (运行与配置)

### Requirements
```bash
pip install numpy pandas matplotlib scikit-learn yfinance
```

### Run Backtester
Use the designated Python environment to run the strategy:
```bash
& 'C:\Users\qwe\AppData\Local\Programs\Python\Python310\python.exe' strategy.py
```

### Parameter Tuning
All quantitative limits are centralized in the `StrategyConfig` class for easy tuning:
```python
config = StrategyConfig(
    portfolio_capital=1000000.0,
    max_holdings=4,
    max_weight_per_stock=0.25,
    adaptive_stop_pct=0.02,   # 2% dynamic stop
    n_shuffles=50,            # Monte Carlo shuffles
)
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
