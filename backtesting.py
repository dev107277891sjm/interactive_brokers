import os
import time
import pandas as pd
import yfinance as yf

DATA_DIR = "data"

def download_yfinance(ticker, start, end, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data is not None and not data.empty:
                return data
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            else:
                pass
    return None

def download_stooq(ticker, start, end):
    """Fallback: Stooq (different server; US tickers use suffix .US)."""
    try:
        import pandas_datareader.data as web
        stooq_ticker = f"{ticker}.US" if not ticker.endswith(".US") else ticker
        # Stooq expects start < end (chronological)
        df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
        if df is not None and not df.empty:
            df = df.sort_index()
            return df
    except Exception:
        pass
    return None

def load_local_csv(ticker, start, end):
    """Fallback: load from data/{ticker}.csv if present."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.loc[start:end]
        if df.empty:
            return None
        return df
    except Exception:
        return None

def get_sample_data(ticker, start, end):
    """Minimal sample OHLCV so the script runs when all sources fail."""
    dates = pd.date_range(start=start, end=end, freq="B")[:20]
    return pd.DataFrame(
        {
            "Open": 170.0,
            "High": 172.0,
            "Low": 169.0,
            "Close": 171.0,
            "Volume": 1_000_000,
        },
        index=dates,
    )

def fetch_data(ticker, start, end, allow_sample=True):
    data = download_yfinance(ticker, start, end)
    if data is not None and not data.empty:
        return data, "yfinance"

    data = download_stooq(ticker, start, end)
    if data is not None and not data.empty:
        return data, "Stooq"

    data = load_local_csv(ticker, start, end)
    if data is not None and not data.empty:
        return data, "local CSV"

    if allow_sample:
        data = get_sample_data(ticker, start, end)
        return data, "sample (no network data)"
    return None, None

if __name__ == "__main__":
    ticker, start, end = "AAPL", "2023-01-01", "2024-01-01"
    data, source = fetch_data(ticker, start, end)

    if data is not None and not data.empty:
        print(f"Data source: {source}")
        print(f"Rows: {len(data)}, columns: {list(data.columns)}\n")
        print("First 10 rows:")
        print(data.head(10))
        if source == "sample (no network data)":
            print("\nTip: Check VPN/firewall for Yahoo, or add data/AAPL.csv for offline use.")
    else:
        print("No data received. Check network or add data/AAPL.csv.")
