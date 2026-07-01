# V2 Experiment: The Conditional Complexity Trap

## Overview
This directory contains the Version 2 (V2) experiment codebase and results for the ARMS (Adaptive Risk Management System) framework.

Unlike V1, which was tested on 49 highly liquid blue-chip stocks (potentially suffering from survivorship bias), V2 introduces a completely unbiased, randomized broad-market universe of 108 stocks spanning the A-share market (from large caps to micro caps) over a full cycle (2019-2023).

## Key Discoveries
The results from V2 reveal the **Conditional Complexity Trap** of Financial Machine Learning:
1. **Full-Cycle Underperformance**: The highly complex ML + Rules system (System 5) yields a cumulative return of ~3.55%, drastically underperforming a simple Moving Average (MA) crossover strategy (~92.09%).
2. **The Defensive Shield (MDD Cap)**: Despite the low absolute return, System 5 suppresses Maximum Drawdown (MDD) to an incredible **-16.68%**, while the simple MA and B&H strategies suffered catastrophic drawdowns of -45% and -23% respectively. 
3. **Regime Dependency (2023 Bear Market)**: In massive bull markets (2019, 2022), the complex system suffers from "Defense Overload" and misses trends. However, in the brutal bear market of 2023, System 5 successfully generated a positive **+8.50%** return, while the MA system failed.

## File Structure
- `strategy.py`: The core logic containing the Purged Walk-Forward RF model and cascading risk layers (ATR, BBL, TOCE).
- `run_full_academic_pipeline.py`: The execution script that runs the pure rules ablation, baselines, and generates deep analysis.
- `sci_ablation_results_v2.csv`: The numerical proof of the Complexity Trap.
- `yearly_breakdown.csv`: The year-by-year performance breakdown proving the regime dependency.
