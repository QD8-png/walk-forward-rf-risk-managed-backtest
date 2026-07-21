import pickle
import os

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
                    all_stocks[k] = {
                        'file': f,
                        'keys': list(v.keys())
                    }

print(f"Total unique stocks across pickles: {len(all_stocks)}")
print("Sample stock keys format:")
sample_k = list(all_stocks.keys())[0]
print(f"Stock: {sample_k} (from {all_stocks[sample_k]['file']})")
print(f"Keys inside dict: {all_stocks[sample_k]['keys']}")

# Check overlap of keys inside dict
common_keys = set(all_stocks[sample_k]['keys'])
for k, v in all_stocks.items():
    common_keys = common_keys.intersection(set(v['keys']))

print(f"Common keys inside stock dicts: {common_keys}")
