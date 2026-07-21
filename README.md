# Multi-Asset Walk-Forward Portfolio Rotation Strategy with Adaptive Risk Management & TOCE

## A High-Performance Machine Learning Backtesting Pipeline with Multi-Frequency Risk Regimes and Capital Rotation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey.svg)](https://github.com/QD8-png/walk-forward-rf-risk-managed-backtest)

This repository implements **ARMS (Adaptive Risk-Managed Strategy)**, a highly rigorous, multi-asset quantitative trading framework. It combines rolling non-linear machine learning out-of-sample forecasts (via Purged Walk-Forward Optimization) with multi-frequency structural risk controls, portfolio-level cross-sectional capital rotation, and our core innovation: **TOCE (Time-based Opportunity Cost Exit)**.

The core philosophy of this system is: **Predict with Machine Learning, Control with Structural Rules.**

---

## 📁 Repository Structure

```
walk-forward-rf-risk-managed-backtest/
├── README.md                           # Main repository documentation & sitemap
├── requirements.txt                    # Project python dependencies
├── LICENSE                             # MIT License
├── strategy.py                         # Core strategy engine & unified portfolio backtesting module
│
├── experiments/                        # Reviewer Response Experiment Suite (A1 - F1)
│   ├── run_exp_A1_seed_sampling.py    # A1: 30-seed resampling (230 source pool, 80 assets per group)
│   ├── run_exp_B1_etf_stratification.py# B1: ETF vs Pure Equities stratification (49 Leaders)
│   ├── run_exp_C1_oos2024.py           # C1: 2024 Out-of-Sample validation & N=2000 Monte Carlo test
│   ├── run_exp_D1_fee_sensitivity.py   # D1: Fee sensitivity decay matrix (13/20/30 bps)
│   ├── run_exp_E1_atr_grid.py          # E1: ATR stop loss multiplier grid (1.5x / 2.0x / 2.5x)
│   └── run_exp_F1_signal_eval.py       # F1: Pointwise signal evaluation & 16D SHAP feature importance
│
├── data_results/                       # All Experimental CSV Deliverables
│   ├── exp_A1_seed_sampling_results.csv
│   ├── exp_B1_etf_stratification.csv
│   ├── exp_C1_oos2024_all49.csv
│   ├── exclusion_35_list.csv          # 35-stock candidate exclusion audit list
│   ├── exp_D1_fee_sensitivity.csv
│   ├── exp_E1_atr_grid.csv
│   ├── exp_F1_signal_evaluation.csv  # ROC-AUC / PR-AUC / Brier metrics
│   ├── exp_F1_calibration_curve.csv   # Probability calibration curve data
│   └── exp_F1_shap_importance.csv     # SHAP mean(|SHAP|) feature rankings
│
├── figures/                            # High-Resolution Academic Visualizations
│   ├── fig9_seed_pairing_v76.png      # Fig 9: 30-Seed paired Calmar comparison (System 5 vs MA)
│   ├── fig12_shap_v76.png             # Fig 12: 16-Dimensional SHAP Feature Importance Ranking
│   ├── fig13_calibration_v76.png      # Fig 13: Out-of-Sample Probability Calibration Curve
│   └── plots/                          # Additional academic figures
│
├── scripts/                            # Core Pipeline & Diagnostic Scripts
│   ├── academic_empirical_pipeline.py
│   ├── model_comparison_pipeline.py
│   ├── run_full_academic_pipeline.py
│   └── calculate_dsr.py
│
├── notebooks/                          # Interactive Jupyter Notebooks
│   ├── ARMS_Complexity_Trap_Reproduction.ipynb
│   └── run_full_academic_pipeline.ipynb
│
└── data_cache/                         # Pre-computed WFO Pickles (.pkl)
    ├── all_assets_rf_2019_2023.pkl
    └── sci_all_assets_2019_2023.pkl
```

---

## ─── 📊 Reviewer Response Experiment Suite (A1 - F1 Summary) ───

| Exp ID | Target Reviewer Issue | Design & Universe | Empirical Findings & Statistical Metrics | Data Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | Reviewer #1 (Figure 9 Mismatch) & Reviewer #6 | **30-Seed Resampling**<br>(230 source pool, 80 assets/group, 2020-06-01 start) | - **Paired $\Delta\text{Calmar}$ Median**: **`+0.3005`**<br>- **95% Confidence Interval**: **`[+0.0737, +0.6732]`**<br>- **Sign Test**: **30/30 positive (100% win rate), $p = 0.0000$** | [exp_A1_seed_sampling_results.csv](data_results/exp_A1_seed_sampling_results.csv) |
| **B1** | Reviewer #3 (ETF Stratification) | **49 Sector Leaders $\rightarrow$ 11 ETFs + 38 Equities** | - **Full 49 Pool**: System 4 Return 36.73%, MDD -21.07%<br>- **11 ETFs**: System 4 Return -7.29%, MDD -9.60%<br>- **38 Equities**: System 4 Return **27.94%**, MDD -24.14% | [exp_B1_etf_stratification.csv](data_results/exp_B1_etf_stratification.csv) |
| **C1** | Reviewer #4 (OOS Bias & Exclusion Audit) | **2024 Out-of-Sample Full Year Validation**<br>($N=2000$ Monte Carlo shuffles) | - **2024 Actual Return**: **`-4.01%`** (MDD -18.25%)<br>- **MC Permuted Mean**: `-1.33%`<br>- **Empirical $p$-value ($N=2000$)**: **`0.6730`** (evidencing concept drift) | [exp_C1_oos2024_all49.csv](data_results/exp_C1_oos2024_all49.csv)<br>[exclusion_35_list.csv](data_results/exclusion_35_list.csv) |
| **D1** | Reviewer #5 (Fee Sensitivity) | **Fee Tier Decay Matrix**<br>(13bps / 20bps / 30bps, 112 random pool) | - **13 bps**: System 5 Calmar = -0.1492 \| MA Calmar = 0.4787<br>- **20 bps**: System 5 Calmar = -0.2282 \| MA Calmar = 0.4178<br>- **30 bps**: System 5 Calmar = -0.2714 \| MA Calmar = 0.3350 | [exp_D1_fee_sensitivity.csv](data_results/exp_D1_fee_sensitivity.csv) |
| **E1** | Reviewer #7 (ATR Stop Grid) | **ATR Stop Multiplier Grid**<br>(1.5x / 2.0x / 2.5x $\times$ Pure Rules+ATR) | - **1.5x ATR**: MDD = -69.56%, Calmar = -0.2028<br>- **2.0x ATR**: MDD = -63.23%, Calmar = -0.2047<br>- **2.5x ATR**: MDD = -49.01%, Calmar = -0.1837 | [exp_E1_atr_grid.csv](data_results/exp_E1_atr_grid.csv) |
| **F1** | Reviewer #8 (Signal Layer & SHAP) | **Pointwise Signal Evaluation + SHAP Explainer**<br>(27,269 OOS points + TreeExplainer 16D features) | - **ROC-AUC**: **`0.5069`**, **PR-AUC**: **`0.3662`**, **Brier**: `0.2575`<br>- **SHAP Importance Top 3**:<br>  1. `Bull/Bear Line Deviation` (`0.029659`)<br>  2. `Trend Line Deviation` (`0.023231`)<br>  3. `Amplitude` (`0.019471`) | [exp_F1_signal_evaluation.csv](data_results/exp_F1_signal_evaluation.csv)<br>[exp_F1_calibration_curve.csv](data_results/exp_F1_calibration_curve.csv)<br>[exp_F1_shap_importance.csv](data_results/exp_F1_shap_importance.csv) |

---

## ─── 🏃 Quick Start: Executing Experiments ───

To run any of the reviewer response experiments directly:

```bash
# Clone the repository
git clone https://github.com/QD8-png/walk-forward-rf-risk-managed-backtest.git
cd walk-forward-rf-risk-managed-backtest

# Run Experiment A1: 30-Seed Resampling
python experiments/run_exp_A1_seed_sampling.py

# Run Experiment B1: ETF Stratification
python experiments/run_exp_B1_etf_stratification.py

# Run Experiment C1: 2024 OOS & N=2000 Permutation Test
python experiments/run_exp_C1_oos2024.py

# Run Experiment F1: SHAP Feature Importance Analysis
python experiments/run_exp_F1_signal_eval.py
```

---

## ─── 🚀 Key Features & Architectural Breakthroughs ───

### 1. Leak-Free Walk-Forward ML Engine
The system retrains its model every **60 days** on a rolling **500-day** historical window. A strict **5-day purge gap** is applied at the training boundary to block future target leakage:

```python
# Purged training split - prevents out-of-sample price leakages into labels
X_train_raw = X[train_end - train_window : train_end - config.future_return_days]
y_train_raw = y[train_end - train_window : train_end - config.future_return_days]
X_test = X[train_end : test_end]
```

### 2. Multi-Stage Adaptive Risk Exits (ARMS)
Operating on the principle: **Cut losses short, let profits run**, the system deploys a multi-frequency defensive structure:
*   **Adaptive Volatility Stop-Loss:** Dynamic ATR-based trailing stop-loss (`Stop = Low_entry - 2.0 × ATR_entry`).
*   **Macro Regime Filter:** Moving average slope check (`MA120_Slope > 0`).
*   **BBI Ladder Profit-Taking:** Liquidates 50% of holding weight on rapid expansion (`BBI_Dev >= 3%` and return $\ge 2\%$).
*   **AI Defensive Exit:** Exits if floating profit is $<3\%$ and ML out-of-sample prediction turns bearish.

### 3. The Core Innovation: Time-based Opportunity Cost Exit (TOCE)
The **TOCE** mechanism dynamically releases stagnant assets based on time-in-position constraints (Patience Days, $T$) and minimum return boundaries (Patience Return, $R$). If an asset is held for $\ge T$ days and its return is $< R$, TOCE triggers an immediate market-order liquidation, returning capital back to the shared pool to capture high-conviction breakout opportunities.

---

## ─── 📜 License & Citation ───

This project is released under the **MIT License**.
