import pickle
import os
import pandas as pd

files = [
    'sci_baostock_assets_2016_2023.pkl',
    'sci_baostock_assets_200_robust.pkl',
    'sci_all_assets_2019_2023.pkl'
]

for f in files:
    if os.path.exists(f):
        with open(f, 'rb') as pf:
            data = pickle.load(pf)
            min_dates = [pd.to_datetime(v['dates']).min() for v in data.values()]
            max_dates = [pd.to_datetime(v['dates']).max() for v in data.values()]
            print(f"{f}:")
            print(f"  Min start date: {min(min_dates)}")
            print(f"  Max start date: {max(min_dates)}")
            print(f"  Min end date: {min(max_dates)}")
            print(f"  Max end date: {max(max_dates)}")
            print(f"  Number of stocks: {len(data)}")
