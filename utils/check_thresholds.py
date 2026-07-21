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

print(f"Total unique stocks: {len(all_stocks)}")

for year in [2019, 2020, 2021]:
    date_str = f"{year}-06-01"
    cutoff = pd.Timestamp(date_str)
    count = 0
    for k, v in all_stocks.items():
        dates = pd.to_datetime(v['dates'])
        if dates.min() <= cutoff and dates.max() >= pd.Timestamp('2023-12-25'):
            count += 1
    print(f"Aligned (start <= {date_str} and end >= 2023-12-25): {count}")
