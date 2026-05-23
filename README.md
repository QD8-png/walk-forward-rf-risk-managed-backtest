# Multi-Asset Walk-Forward Random Forest Portfolio Rotation Strategy
## 多资产海选轮动与多层级动态风险管理量化系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Quant Grade](https://img.shields.io/badge/Grade-Institutional-success.svg)](#)

An institutional-grade quantitative backtesting framework combining rolling **non-linear Machine Learning forecasts** with **multi-frequency structural risk controls**, **portfolio-level shared-capital rotation**, and **Monte Carlo statistical validation**.

> **Core Philosophy**: *Predict with Machine Learning, control with Structural Rules.*
>
> 核心哲学：预测交给机器学习，风控留给结构化规则。

---

## 📈 System Architecture

```
    ┌─────────────────────────────────────────────────┐
    │              Market Data Pool (CSV / yfinance)   │
    └──────────────────────┬──────────────────────────┘
                           ▼
              Feature Engineering (16 technical overlays)
                           ▼
            ┌──────────────────────────────────┐
            │  Walk-Forward Rolling RF Training │
            │  (500-day window, 60-day step)   │
            │  ★ Leak-free purged train slice  │
            └──────────────┬───────────────────┘
                           ▼
            ┌──────────────────────────────────┐
            │  Cross-Sectional Rank & Screen   │
            │  • ML probability sorting        │
            │  • 5-layer concurrent filters    │
            └──────────────┬───────────────────┘
                           ▼
            ┌──────────────────────────────────┐
            │  Shared Capital Pool Allocation   │
            │  ¥1,000,000 · Max 4 holdings     │
            │  Single stock cap: 25% weight    │
            └──────────────┬───────────────────┘
                           ▼
            ┌──────────────────────────────────┐
            │  Multi-Stage Asymmetric Exits    │
            │  • 2% adaptive stop-loss         │
            │  • Trailing stop-profit          │
            │  • BBI ladder position halving   │
            └──────────────┬───────────────────┘
                           ▼
              Monte Carlo Permutation Validation
              (p-value statistical significance)
```

---

## 🚨 Version History & Quantitative Audit Log

### v4.0 — Current Release (2026-05-23)

**Architecture upgrade**: Single-asset backtester → Multi-asset shared-capital pool rotation.

| Category | What Changed | Technical Detail |
| :--- | :--- | :--- |
| 🔴 **Bug Fix** | Walk-forward look-ahead data leakage eliminated | Training window now subtracts `future_return_days` (5 days) to prevent target labels from seeing test-set closing prices. |
| 🔴 **Bug Fix** | Serial return compounding error corrected | Daily returns from all holdings are summed in parallel first, then applied to the capital pool once: `capital *= (1 + Σ(return_i × weight_i))`. |
| 🔴 **Bug Fix** | Fixed-tick stop-loss scale bias removed | Replaced absolute ¥0.05 stop-loss with **2% adaptive stop-loss** relative to purchase-day low. High-priced stocks (e.g. Moutai at ¥1500+) no longer get instantly stopped out by normal volatility. |
| 🟢 **Feature** | Portfolio-level capital rotation | Shared ¥1M capital pool, cross-sectional ML confidence ranking, max 4 concurrent holdings at 25% weight each. |
| 🟢 **Feature** | Multiprocessing acceleration | `ProcessPoolExecutor` + `n_jobs=-1` for parallel walk-forward training across assets. |
| 🟢 **Feature** | LaTeX Booktabs table generator | Auto-generates publication-ready three-line tables for academic papers. |

### v1.0–v3.0 — Historical

Single-asset independent backtester with fixed-price stop-losses. Contained the three critical vulnerabilities listed above (leakage, compounding, scale bias).

---

## 🛠️ Key Components

### 1. Feature Engineering (16 Features)

| Category | Features | Purpose |
| :--- | :--- | :--- |
| Momentum | Lagged returns (1/2/3 day) | Short-term memory capture |
| Mean Reversion | RSI-14, KDJ-J oscillator | Overbought/oversold detection |
| Trend Deviation | Price deviation from MA5/10/60, EMA13, BBI, Bull-Bear Line | Structural trend positioning |
| Market Structure | Volume change, amplitude, rolling volatility, MA crossover | Regime and microstructure signals |

### 2. Leak-Free Walk-Forward ML Engine

Models retrain every **60 days** on a rolling **500-day** window. A strict **5-day purge gap** at the training boundary prevents future label contamination:

```python
# Purged training window — no test-set price information leaks into labels
X_train = X[train_end - train_window : train_end - config.future_return_days]
y_train = y[train_end - train_window : train_end - config.future_return_days]
X_test  = X[train_end : test_end]
```

### 3. Five-Layer Cross-Sectional Entry Filter

All five conditions must be satisfied **simultaneously** for a buy signal:

| Layer | Condition | Rationale |
| :---: | :--- | :--- |
| 1 | RF predicts bullish (`y_pred == 1`) | ML conviction gate |
| 2 | MA120 slope > 0 | Long-term structural uptrend |
| 3 | KDJ J-value < 20 | Panic oversold / fear regime |
| 4 | Price ≥ Bull-Bear Line | Regime strength confirmation |
| 5 | ≥ 120 days since last exit on this asset | Risk cooldown period |

### 4. Asymmetric Exit Mechanics

Design principle: **catch small fish quickly, let big fish run.**

| Priority | Trigger | Action | Category |
| :---: | :--- | :---: | :--- |
| 1 (Highest) | Price < Bull-Bear Line | Full exit | Structural stop |
| 2 | Price < Entry-day Low × 0.98 | Full exit | 2% adaptive stop-loss |
| 3 | Unrealized gain < 3% AND ML turns bearish | Full exit | Swing take-profit |
| 4 | Unrealized gain ≥ 3% AND price drops 5% from peak | Full exit | Trailing stop-profit |
| 5 | Price ≥ BBI+3% AND bullish candle ≥ 2% | Halve position | BBI ladder reduction |

### 5. Dynamic Position Sizing

Capital allocation scales linearly with ML prediction confidence:

```
ML Confidence 50%  →  0% allocation (no trade)
ML Confidence 60%  →  33% allocation
ML Confidence 70%  →  67% allocation
ML Confidence 80%  →  100% allocation (full weight)
```

### 6. Monte Carlo Permutation Test

Statistical validation that ML predictions add genuine alpha beyond structural rules:
1. Shuffle ML prediction signals randomly (keeping technical indicators intact).
2. Re-run the full backtest 50+ times to build a null return distribution.
3. Compute empirical *p*-value. If *p* < 0.05, reject the null hypothesis.

---

## 📂 Project Structure

```
├── strategy.py      # Core: features, walk-forward RF, portfolio rotation, exits, MC validation
├── README.md        # This document
├── LICENSE          # MIT License
└── .gitignore
```

---

## 💻 Quick Start

### Requirements

```bash
pip install numpy pandas matplotlib scikit-learn yfinance
```

### Run

```bash
python strategy.py
```

The system will automatically download historical data via `yfinance`, run walk-forward training on all assets in parallel, execute the 2024-present portfolio rotation backtest, and output an equity curve plot to `plots/`.

### Configuration

All parameters are centralized in `StrategyConfig` — no magic numbers scattered in code:

```python
config = StrategyConfig(
    portfolio_capital=1_000_000.0,
    max_holdings=4,
    max_weight_per_stock=0.25,
    adaptive_stop_pct=0.02,     # 2% dynamic stop-loss
    train_window=500,
    retrain_every=60,
    n_shuffles=50,              # Monte Carlo iterations
)
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
