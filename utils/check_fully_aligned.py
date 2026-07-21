import pickle
import os
import pandas as pd

files = [
    'sci_baostock_assets_2016_2023.pkl',
    'sci_baostock_assets_200_robust.pkl',
    'sci_all_assets_2019_2023.pkl'
]

all_stocks = {}
for f in files:
    if os.path.exists(f):
        with open(f, 'rb') as pf:
            data = pickle.load(pf)
            for k, v in data.items():
                if k not in all_stocks:
                    all_stocks[k] = v

aligned_stocks = {}
for k, v in all_stocks.items():
    dates = pd.to_datetime(v['dates'])
    if len(dates) == 0:
        continue
    # Check if stock has data starting in 2018 or 2019, and ending in late 2023
    min_d = dates.min()
    max_d = dates.max()
    if min_d <= pd.Timestamp('2019-01-10') and max_d >= pd.Timestamp('2023-12-25'):
        aligned_stocks[k] = v

print(f"Total unique stocks: {len(all_stocks)}")
print(f"Fully aligned stocks (Jan 2019 - Dec 2023): {len(aligned_stocks)}")
