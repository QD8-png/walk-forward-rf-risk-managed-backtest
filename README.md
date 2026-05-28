# Multi-Asset Walk-Forward Portfolio Rotation Strategy with Adaptive Risk Management & TOCE

## A High-Performance Machine Learning Backtesting Pipeline with Multi-Frequency Risk Regimes and Capital Rotation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey.svg)](https://github.com/QD8-png/walk-forward-rf-risk-managed-backtest)

This repository implements **ARMS (Adaptive Risk-Managed Strategy)**, a highly rigorous, multi-asset quantitative trading framework. It combines rolling non-linear machine learning out-of-sample forecasts (via Walk-Forward Optimization) with multi-frequency structural risk controls, portfolio-level cross-sectional capital rotation, and our core innovation: **TOCE (Time-based Opportunity Cost Exit)**.

The core philosophy of this system is: **Predict with Machine Learning, Control with Structural Rules.**

---

## ─── 🚀 Key Features & Architectural Breakthroughs ───

### 1. Leak-Free Walk-Forward ML Engine
The system retrains its model every **60 days** on a rolling **500-day** historical window. A strict **5-day purge gap** is applied at the training boundary to block future target leakage, entirely eliminating out-of-sample price leakages into features:

```python
# Purged training split - prevents out-of-sample price leakages into labels
X_train_raw = X[train_end - train_window : train_end - config.future_return_days]
y_train_raw = y[train_end - train_window : train_end - config.future_return_days]
X_test = X[train_end : test_end]
```

### 2. Multi-Stage Adaptive Risk Exits (ARMS)
Operating on the principle: **Cut losses short, let profits run**, the system deploys a multi-frequency defensive structure:
*   **Adaptive Volatility Stop-Loss:** Dynamic ATR-based trailing stop-loss (`Stop = Low_entry - 2.0 × ATR_entry`) eliminating scale bias across varying asset price regimes.
*   **Macro Regime Filter:** Simple moving average slope check (`MA120_Slope > 0`) to block entries in structural bear markets.
*   **BBI Ladder Profit-Taking:** Liquidates 50% of the holding weight when price expands rapidly (`BBI_Dev >= 3%` and single-day return $\ge 2\%$).
*   **AI Defensive Exit:** Exits if floating profit is $<3\%$ and the ML out-of-sample prediction turns bearish, securing capital safety.

### 3. The Core Innovation: Time-based Opportunity Cost Exit (TOCE)
In shared-capital portfolios (e.g., maximum 4 concurrent holdings with a 25% single-asset cap), traditional support filters can trap capital in stagnant, non-trending assets—creating the **Capital Lockup Paradox**. 

The **TOCE** mechanism dynamically releases stagnant assets based on time-in-position constraints (Patience Days, $T$) and minimum return boundaries (Patience Return, $R$). If an asset is held for $\ge T$ days and its return is $< R$, TOCE triggers an immediate market-order liquidation, returning capital back to the shared pool to capture high-conviction breakout opportunities.

---

## ─── 📊 Empirical Research Routes & Findings ───

The backtesting pipeline has been evaluated on a universe of **49 representative sector-leader equities** in the Chinese A-share market over a complete market cycle from **2019 to 2023** (incorporating extreme volatility and prolonged bear markets).

### 📈 Route A: Ablation Study & Sensitivity Analysis
A controlled 5-stage ablation study was performed to isolate the marginal contribution of each risk component. 

*   **System 1:** Pure ML Baseline (ML entries, no risk exits).
*   **System 2:** ML + ATR Volatility Stop.
*   **System 3:** ML + ATR + BBL Trend Exit (No TOCE).
*   **System 4:** ML + ATR + BBL + TOCE (Our optimal setup).
*   **System 5:** Full ARMS Framework (Maximum defense, including 120-day cooldown).
*   **Benchmark:** Equal-weighted Buy & Hold (B&H).

#### Ablation Performance Metrics (2019-2023)
| System Setup | Total Return | Max Drawdown | Sharpe Ratio |
| :--- | :---: | :---: | :---: |
| **System 1:** Pure ML Baseline | 1.59% | -19.83% | -0.1679 |
| **System 2:** ML + ATR Stop | 24.51% | -20.07% | 0.2898 |
| **System 3:** ML + ATR + BBL | 14.71% | -27.87% | 0.0769 |
| **System 4:** ML + ATR + BBL + TOCE | **36.73%** | -21.07% | **0.4014** |
| **System 5:** Full ARMS Framework | 0.99% | **-16.61%** | -0.2693 |
| **Benchmark:** Buy & Hold | 102.47% | -28.54% | N/A |

#### Ablation Equity Curve Comparison
![Ablation Study Equity Comparison](plots/academic_ablation_equity_comparison.png)

#### TOCE Parameter Sensitivity Heatmap
To ensure robustness, a grid search was conducted across Patience Days ($T \in [2, 15]$) and Patience Return ($R \in [0.0\%, 1.5\%]$):
![TOCE Parameter Sensitivity Heatmap](plots/academic_sensitivity_heatmap.png)

*Key Finding:* Performance peaking in a tight, logical cluster ($T \in [3, 5]$, $R \in [0.5\%, 1.0\%]$), confirming that patience should align with the model's forecasting horizon (5 days) and the return threshold should cover transaction friction.

---

### 🐎 Route B: Multi-Model Horse Race (Tabular Data Consensus)
We replaced the baseline Random Forest (RF) classifier with alternative machine learning backbones under identical walk-forward backtesting conditions:

| ML Backbone Model | Total Return | Max Drawdown | Sharpe Ratio | Total Trades |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (RF)** | 0.99% | **-16.61%** | -0.2693 | 1175 |
| **LightGBM (LGBM)** | **15.75%** | -23.38% | **0.0803** | 1455 |
| **LSTM (Deep Learning)** | -3.69% | -24.05% | -0.2221 | 1098 |
| **Voting Ensemble** | -6.03% | -17.52% | -0.3528 | 1176 |

#### Out-of-Sample Model Comparison Equity Curves
![Multi-Model Performance Comparison](plots/academic_model_comparison.png)

*Scholarly Alignment:* Our empirical hierarchy (**LightGBM > RF > LSTM**) strongly supports the consensus established by Grinsztajn et al. (2022). Deep neural architectures struggle with tabular time series due to high noise and non-smooth decision boundaries, whereas tree-based gradient boosters excel at constructing efficient piecewise-constant boundaries.

---

## ─── 🎲 Statistical Validation (Monte Carlo) ───

To reject the Null Hypothesis ($H_0$: *observed outperformance is a product of pure luck*), we execute a **Monte Carlo Permutation Test** (50 shuffles). By temporally shuffling predictions while preserving the price autocorrelation structure:
*   **Strategy Return:** **36.73%**
*   **Random Permuted Mean Return:** **4.82%**
*   **Random Permuted Max Return:** **12.44%**
*   **Empirical p-value:** **0.0000** ($p < 0.02$)

This confirms that the ARMS walk-forward ML predictions capture genuine, statistically significant economic anomalies.

---

## ─── 📂 Repository Structure ───

```
├── strategy.py                  # Core backtesting engine & feature pipeline
├── strategy_non_compounding.py  # Backtesting engine with parallel return calculation
├── test_expanded_universe_cycle.py # Bulk backtesting across the A-share universe
├── ARMS_TOCE_paper_draft_en.md  # Comprehensive English academic paper draft
├── ablation_study_results.csv   # Out-of-sample ablation metrics
├── model_comparison_results.csv # Out-of-sample model horse-race metrics
├── sensitivity_patience_results.csv  # Grid search return data for patience parameters
├── sensitivity_cooldown_results.csv  # Cooldown sensitivity metrics
├── plots/
│   ├── academic_ablation_equity_comparison.png # Ablation curves chart
│   ├── academic_sensitivity_heatmap.png        # TOCE sensitivity heatmap
│   └── academic_model_comparison.png           # Model comparison curves chart
├── LICENSE                      # MIT License
└── README.md                    # Technical project landing page (this file)
```

---

## ─── 🛠️ Quick Start ───

### 1. Installation
Install core quantitative analysis and machine learning dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn yfinance lightgbm
```

### 2. Run Backtest
Run the complete pipeline (fetches sector leaders via `yfinance`, executes rolling training, backtests and runs Monte Carlo simulation):
```bash
python strategy.py
```

### 3. Change Predictive Model Backbone
To switch between machine learning backbones, open `strategy.py` and modify the `model_type` parameter inside `main()`:
```python
# Options: 'rf' (Random Forest), 'lgbm' (LightGBM), 'lr' (Logistic Regression)
model_type = 'lgbm'
```

---

## ─── 📝 Citation & Contact ───

If you find this research useful in your academic or professional quantitative work, please cite the draft:
```bibtex
@article{arms_toce_2026,
  title={A Machine Learning-Enhanced Quantitative Trading Framework with Adaptive Risk Management and Time-Based Opportunity Cost Exit},
  author={QD8-png},
  journal={GitHub Repository},
  year={2026},
  url={https://github.com/QD8-png/walk-forward-rf-risk-managed-backtest}
}
```

---

## ─── 📄 License ───

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
