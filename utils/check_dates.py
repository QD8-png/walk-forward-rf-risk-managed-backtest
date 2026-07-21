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

date_lengths = []
min_dates = []
max_dates = []
invalid_stocks = []

for k, v in all_stocks.items():
    dates = pd.to_datetime(v['dates'])
    if len(dates) == 0:
        invalid_stocks.append(k)
        continue
    date_lengths.append(len(dates))
    min_dates.append(dates.min())
    max_dates.append(dates.max())

print(f"Min length of dates: {min(date_lengths)}")
print(f"Max length of dates: {max(date_lengths)}")
print(f"Earliest date range: {min(min_dates)} to {max(min_dates)}")
print(f"Latest date range: {min(max_dates)} to {max(max_dates)}")
if invalid_stocks:
    print(f"Invalid stocks: {invalid_stocks}")
