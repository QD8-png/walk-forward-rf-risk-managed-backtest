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

print("Min start date in all_stocks:", min(pd.to_datetime(v['dates']).min() for v in all_stocks.values()))
print("Max start date in all_stocks:", max(pd.to_datetime(v['dates']).min() for v in all_stocks.values()))
print("Min end date in all_stocks:", min(pd.to_datetime(v['dates']).max() for v in all_stocks.values()))
print("Max end date in all_stocks:", max(pd.to_datetime(v['dates']).max() for v in all_stocks.values()))

# Let's see how many stocks have data starting on or before 2021-01-01 and ending on or after 2023-12-25
count = 0
for k, v in all_stocks.items():
    dates = pd.to_datetime(v['dates'])
    if dates.min() <= pd.Timestamp('2021-01-01') and dates.max() >= pd.Timestamp('2023-12-25'):
        count += 1
print(f"Stocks aligned from 2021-01-01 to 2023-12-25: {count}")

# Let's print a sample stock's dates
sample_k = list(all_stocks.keys())[0]
dates = pd.to_datetime(all_stocks[sample_k]['dates'])
print(f"Sample stock {sample_k} dates: count={len(dates)}, min={dates.min()}, max={dates.max()}")
