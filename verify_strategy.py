import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import strategy
    print("strategy.py successfully imported!")
    config = strategy.StrategyConfig()
    print("StrategyConfig initialized successfully!")
    print("HAS_TORCH:", strategy.HAS_TORCH)
except Exception as e:
    print("Error importing strategy.py:")
    import traceback
    traceback.print_exc()
