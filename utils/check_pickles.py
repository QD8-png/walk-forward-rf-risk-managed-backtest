import pickle
import os

files = [
    'sci_baostock_assets_2016_2023.pkl',
    'sci_baostock_assets_200_robust.pkl',
    'sci_all_assets_2019_2023.pkl',
    'all_assets_2019_2023.pkl',
    'all_assets_rf_2019_2023.pkl'
]

for f in files:
    if os.path.exists(f):
        with open(f, 'rb') as pf:
            data = pickle.load(pf)
            print(f"{f}: {len(data)} keys (type: {type(data)})")
            # print some sample keys
            keys = list(data.keys())
            print(f"  First 5 keys: {keys[:5]}")
    else:
        print(f"{f} does not exist")
