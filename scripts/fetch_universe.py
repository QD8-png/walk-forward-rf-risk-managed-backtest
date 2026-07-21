import os
import pandas as pd

def format_yahoo_ticker(ticker):
    ticker_str = str(ticker).zfill(6)
    if ticker_str.startswith('6'):
        return f"{ticker_str}.SS"
    elif ticker_str.startswith(('0', '3')):
        return f"{ticker_str}.SZ"
    elif ticker_str.startswith(('4', '8')):
        return f"{ticker_str}.BJ"
    return ticker_str

def fetch_and_save_index_data():
    try:
        import akshare as ak
    except ImportError:
        print("akshare is not installed. Please run: pip install akshare")
        return

    output_dir = r"C:\Users\qwe\.gemini\antigravity\scratch\walk-forward-rf-risk-managed-backtest"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "csi_300_500_constituents.csv")

    tickers = []
    
    # Fetch CSI 300
    try:
        print("Fetching current constituents for CSI 300 (000300)...")
        df_300 = ak.index_stock_cons(symbol="000300")
        col_name = "品种代码" if "品种代码" in df_300.columns else df_300.columns[0]
        tickers.extend(df_300[col_name].tolist())
    except Exception as e:
        print(f"Error fetching CSI 300: {e}")

    # Fetch CSI 500
    try:
        print("Fetching current constituents for CSI 500 (000905)...")
        df_500 = ak.index_stock_cons(symbol="000905")
        col_name = "品种代码" if "品种代码" in df_500.columns else df_500.columns[0]
        tickers.extend(df_500[col_name].tolist())
    except Exception as e:
        print(f"Error fetching CSI 500: {e}")

    if not tickers:
        print("Failed to fetch any constituents.")
        return

    # Remove duplicates
    unique_tickers = list(set(tickers))
    
    # Format for Yahoo Finance
    yahoo_tickers = [format_yahoo_ticker(t) for t in unique_tickers]

    df_out = pd.DataFrame({
        "local_ticker": unique_tickers,
        "yahoo_ticker": yahoo_tickers
    })
    
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Successfully saved {len(df_out)} tickers to {output_path}")

if __name__ == "__main__":
    fetch_and_save_index_data()
