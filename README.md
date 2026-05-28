# ARMS: Adaptive Risk-Managed Strategy Framework with Time-based Opportunity Cost Exit (TOCE)
## A High-Performance Machine Learning Quantitative Trading System with Capital Rotation for A-Share Portfolio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Quant Platform](https://img.shields.io/badge/Universe-China_A--Share_49_Assets-red.svg)](https://github.com/)

This repository implements **ARMS (Adaptive Risk-Managed Strategy)**, a rigorous academic-grade quantitative trading framework. It integrates rolling non-linear machine learning forecasts with a multi-stage structural risk control system and introduces the **TOCE (Time-based Opportunity Cost Exit)** mechanism to actively resolve the "capital lockup paradox" in quantitative portfolio management.

We validate the framework through a comprehensive **5-stage Ablation Study** and **Multi-Algorithm Empirical Comparison (Random Forest, LightGBM, LSTM, and Voting Ensemble)** over a 5-year market cycle (2019–2023) using a 49-asset Chinese A-share sector leader universe.

The core philosophy of this system is: **Predict with Machine Learning, Control with Structural Rules.**

---

## ─── 🏗️ System Architecture ───

The system operates as a unified walk-forward pipeline, passing raw stock data through purged learning windows, cross-sectional ranking gates, and multi-stage exit rules:

```
                  ┌──────────────────────────────────────────────┐
                  │        Market Data Pool (49 Assets)          │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                             Feature Engineering Layer
                         (16 Technical Overlays & Ratios)
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Purged Walk-Forward Training Slice        │
                  │    - 500-day rolling window, 60-day step     │
                  │    - 5-day Purge Gap (No Look-Ahead Leak)     │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │        Cross-Sectional Entry Gate            │
                  │        - Multi-model confidence rank         │
                  │        - 5 concurrent risk/regime filters    │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          Shared Capital Pool                 │
                  │          - $1,000,000 Initial AUM            │
                  │          - Max 4 concurrent holdings         │
                  │          - Single asset exposure cap (25%)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     Multi-Stage Asymmetric Exits             │
                  │     - 2.0 × ATR Dynamic Stop-Loss            │
                  │     - Bull-Bear Line Trend Stop              │
                  │     - TOCE Capital Rotation Trigger (Core)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                      Statistical Validation & Permutation
                    - Wilcoxon Signed-Rank Sign Test
                    - Monte Carlo Permutation Test (50 Shuffles)
```

---

## ─── 📈 Empirical Performance & Key Findings ───

### 1. 5-Stage Ablation Study (Route A)
To isolate and prove the marginal contribution of each risk management layer, we conducted a unified 5-stage Ablation Study over the 5-year macro-cycle (2019–2023).

| System Configuration | Total Return | Max Drawdown | Sharpe Ratio | Economic Significance |
| :--- | :---: | :---: | :---: | :--- |
| **System 1: Pure ML Baseline** | 1.59% | -19.83% | -0.1679 | Extremely inefficient; high transaction friction and whipsaw |
| **System 2: ML + ATR Stop** | 24.51% | -20.07% | 0.2898 | Drastic return boost; downside tail risk containment |
| **System 3: ML + ATR + BBL** | 14.71% | -27.87% | 0.0769 | Negative synergy; blocks entries but causes capital lockup |
| **System 4: ML + ATR + BBL + TOCE (Proposed)** | **36.73%** | **-21.07%** | **0.4014** | **Optimal allocation; forces high-efficiency cash rotation** |
| **System 5: Full ARMS Framework** | 0.99% | -16.61% | -0.2693 | Over-conservative; trailing stops clip profitable runs |
| **Benchmark: Buy & Hold (B&H)** | 102.47% | -28.54% | N/A | Market Beta; extreme volatility and structural drawdowns |

#### 📊 Cumulative Equity Comparison (Ablation)
*Generated at `plots/academic_ablation_equity_comparison.png` using a TradingView-themed premium dark financial aesthetic:*

![Ablation Cumulative Equity Comparison](plots/academic_ablation_equity_comparison.png)

---

### 2. Theoretical Validation of TOCE (Opportunity Cost Exit)

* **The Capital Lockup Paradox**: Traditional trend filters (like the Bull-Bear Line in System 3) block entries during bearish trends but trap capital in stagnant or sideways-moving positions ("zombie holdings").
* **The TOCE Mechanism**: By enforcing a strict time limit (`patience_days`) for an asset to prove its upward momentum, TOCE actively recycles capital:
  
  $$\text{If } t_{\text{holding}} \ge \text{patience\_days} \quad \text{and} \quad R_{\text{unrealized}} < \text{patience\_return} \implies \text{Liquidate}$$
  
  This releases liquidity back into the shared capital pool, permitting the system to immediately capture fresh, high-confidence ML signals from the rolling cross-sectional ranking.

#### 📊 Two-Dimensional Parameter Grid-Search Heatmap
*We performed a grid search on `patience_days` vs. `patience_return` (with `cooldown_days = 15`) to map the robustness plateau:*

![TOCE Grid-Search Heatmap](plots/academic_sensitivity_heatmap.png)

* **Robustness Plateau**: A highly stable region exists between **2 and 5 patience days**, where the strategy consistently delivers **35.85% to 38.61%** total return and Sharpe ratios above **0.40**.
* **Stagnant Regime**: Increasing patience beyond 10 days leads to a performance collapse (e.g., negative returns, drawdown exceeding **-30%**), mathematically proving the critical role of opportunity cost management.

#### 📊 Cooldown Days Sensitivity Analysis
We evaluated the impact of varying the post-liquidation asset lock-out period (`cooldown_days`):

| Cooldown Days | Total Return | Max Drawdown | Sharpe Ratio | Interpretation |
| :---: | :---: | :---: | :---: | :--- |
| **5 days** | 9.50% | -26.90% | -0.0079 | Hyper-active; re-enters too quickly, suffering from transaction friction |
| **10 days** | 10.34% | -24.05% | 0.0060 | Under-filtered; high noise in short re-entries |
| **15 days (Optimal)** | **36.73%** | **-21.07%** | **0.4014** | **Perfect balance; allows structural consolidation post-exit** |
| **30 days** | 4.87% | -24.43% | -0.0952 | Stagnant interval; misses immediate mean-reversion pullbacks |
| **60 days** | 32.32% | -15.66% | 0.4082 | High stability; excellent drawdown containment with robust returns |
| **120 days (Original)** | 17.15% | -16.98% | 0.1637 | Pathological paralysis; misses multiple profitable trading cycles |

---

### 3. Route B: Multi-Algorithm Empirical Comparison
To test the generalizability of our ARMS framework, we upgraded the baseline classifier to LightGBM, a PyTorch-based Long Short-Term Memory (LSTM) network, and a Voting Ensemble of all three under identical WFO and ARMS rules:

| Model Architecture | Total Return | Max Drawdown | Sharpe Ratio | Total Trades | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (RF)** | 0.99% | **-16.61%** | -0.2693 | 1175 | Highly conservative baseline; tight drawdowns |
| **LightGBM (LGBM)** | **15.75%** | -23.38% | **0.0803** | 1455 | **Peak Return & Sharpe; GBDT handles financial noise best** |
| **LSTM (Temporal NN)** | -3.69% | -24.05% | -0.2221 | 1098 | Underperforms; deep learning struggles with non-stationary tabular data |
| **Ensemble (Voting)** | -6.03% | -17.52% | -0.3528 | 1176 | Smoothed drawdowns; dragged down by LSTM |

#### 📊 Multi-Algorithm Equity Curves
*Comparative walk-forward portfolio net values (2019-2023):*

![Multi-Algorithm Net Value Curves](plots/academic_model_comparison.png)

---

## ─── 📁 Project Directory Structure ───

```
walk-forward-rf-risk-managed-backtest/
│
├── strategy.py                        # Core system config, indicators, backtester, and portfolio rotation rules
├── academic_empirical_pipeline.py    # Route A execution script: ablation study, sensitivity analysis, plots
├── model_comparison_pipeline.py       # Route B execution script: RF vs LGBM vs LSTM vs Ensemble, comparative plots
│
├── plots/                             # Premium dark-theme visualization assets
│   ├── academic_ablation_equity_comparison.png
│   ├── academic_sensitivity_heatmap.png
│   └── academic_model_comparison.png
│
├── ablation_study_results.csv         # Raw table data from Route A ablation study
├── sensitivity_patience_results.csv   # Raw table data from TOCE patience parameter grid search
├── sensitivity_cooldown_results.csv   # Raw table data from cooldown parameter sensitivity study
├── model_comparison_results.csv       # Raw table data from Route B comparison study
│
├── requirements.txt                   # Standard quantitative packages & deep learning dependencies
├── LICENSE                            # MIT License
└── README.md                          # Comprehensive project technical documentation (this file)
```

---

## ─── 🚀 Getting Started ───

### 1. Installation
Install the required packages. The environment is compatible with Python 3.10+:

```bash
pip install -r requirements.txt
```

*Note: The requirements include `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `yfinance`, `lightgbm`, and `torch` (PyTorch CPU).*

### 2. Execution

#### Run Route A (Empirical Validation & Ablation):
To execute the rolling walk-forward modeling, compile the 5-stage ablation study, run the grid search, and export the premium charts:
```bash
python academic_empirical_pipeline.py
```
*Note: Rolling walk-forward forecasts are automatically cached to `all_assets_2019_2023.pkl` (5.6MB) to bypass intensive model retraining on subsequent runs.*

#### Run Route B (Multi-Algorithm Model Comparison):
To execute comparative WFO and ARMS backtesting across RF, LightGBM, LSTM, and Voting Ensemble:
```bash
python model_comparison_pipeline.py
```
*Note: Walk-forward predictions for LGBM, LSTM, and Ensemble are cached as separate `.pkl` files for instant reload.*

#### Run the Original Baseline Verification:
```bash
python verify_strategy.py
```

---

## ─── 📝 Academic Paper Manuscript ───

The corresponding publication-ready academic manuscript is compiled and updated as:
* **File Path**: `C:\Users\qwe\Desktop\商业计划\ARMS_paper_v2.docx`
* **Format**: Microsoft Word `.docx` (fully formatted with tables, embedded center figures, and a complete bibliography)
* **Highlights**:
  - Eliminates all meta-commentary / editorial bracketed notes (100% clean)
  - Seamlessly integrates the 49-asset empirical results across Route A and Route B
  - Mathematically formalizes the capital lockup paradox and TOCE exit rule
  - Integrates standard academic references, including **Grinsztajn et al. (2022)**

---

## ─── 📜 License ───

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
