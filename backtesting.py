"""
Backtesting script: fetches OHLCV data, runs a simple MA20 crossover strategy,
and plots results in a TradingView-style candlestick chart with buy/sell markers.
"""

import os
import time
import pandas as pd
import yfinance as yf

# Folder for optional local CSV fallback (e.g. data/AAPL.csv)
DATA_DIR = "data"


def normalize_ohlcv(df):
    """Ensure OHLCV columns and DatetimeIndex for plotting (yfinance uses MultiIndex)."""
    df = df.copy()
    # Flatten MultiIndex columns from yfinance (e.g. (Close, AAPL) -> Close)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Keep only standard OHLCV columns so all sources (yfinance, Stooq, CSV) match
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]
    df.index = pd.to_datetime(df.index)
    return df

def download_yfinance(ticker, start, end, max_attempts=3):
    """Download OHLCV from Yahoo Finance with retries (handles connection resets / timeouts)."""
    for attempt in range(max_attempts):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data is not None and not data.empty:
                return data
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # Backoff: 1s, 2s, 4s...
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            else:
                pass
    return None

def download_stooq(ticker, start, end):
    """Fallback when Yahoo fails: Stooq (different server). US tickers need suffix .US."""
    try:
        import pandas_datareader.data as web
        stooq_ticker = f"{ticker}.US" if not ticker.endswith(".US") else ticker
        df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
        if df is not None and not df.empty:
            df = df.sort_index()  # Stooq returns newest first; we want oldest first
            return df
    except Exception:
        pass
    return None

def load_local_csv(ticker, start, end):
    """Fallback for offline use: load OHLCV from data/{ticker}.csv (index=date, cols=Open,High,Low,Close,Volume)."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.loc[start:end]  # Slice to requested date range
        if df.empty:
            return None
        return df
    except Exception:
        return None

def get_sample_data(ticker, start, end):
    """Last-resort fallback: fake OHLCV for ~20 business days so the script runs and chart logic can be tested."""
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
    """Try data sources in order: Yahoo -> Stooq -> local CSV -> (optional) sample data. Returns (DataFrame, source_name)."""
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

def plot_backtest(ticker, data, trade_log, title=None):
    """Draw TradingView-style chart: candlesticks, volume bar, MA20 line, and buy (^) / sell (v) markers."""
    import mplfinance as mpf

    df = data.copy()
    df.index = pd.to_datetime(df.index)

    # Build series aligned to df index: price at trade dates, NaN elsewhere (mplfinance needs same-length series)
    buy_prices = pd.Series(float("nan"), index=df.index)
    sell_prices = pd.Series(float("nan"), index=df.index)
    for date, action, price in trade_log:
        if date in buy_prices.index:
            if action == "BUY":
                buy_prices.loc[date] = price
            else:
                sell_prices.loc[date] = price

    # Extra overlays on the main price panel (panel=0): MA20 line, buy triangles, sell triangles
    addplots = [
        mpf.make_addplot(df["MA20"], color="orange", width=0.8, panel=0),
        mpf.make_addplot(buy_prices, type="scatter", marker="^", color="lime", markersize=80, panel=0),
        mpf.make_addplot(sell_prices, type="scatter", marker="v", color="red", markersize=80, panel=0),
    ]

    mpf.plot(
        df,
        type="candle",
        volume=True,
        addplot=addplots,
        title=title or f"{ticker} backtest",
        ylabel="Price",
        ylabel_lower="Volume",
        style="charles",
        figsize=(14, 8),
    )


if __name__ == "__main__":
    # --- 1) Configuration and data fetch ---
    ticker, start, end = "AAPL", "2023-01-01", "2024-01-01"
    data, source = fetch_data(ticker, start, end)

    if data is not None and not data.empty:
        # --- 2) Normalize columns (yfinance MultiIndex -> flat OHLCV) and print summary ---
        data = normalize_ohlcv(data)
        print(f"Data source: {source}")
        print(f"Rows: {len(data)}, columns: {list(data.columns)}\n")
        print("First 10 rows:")
        print(data.head(10))
        if source == "sample (no network data)":
            print("\nTip: Check VPN/firewall for Yahoo, or add data/AAPL.csv for offline use.")

        # --- 3) Strategy: MA20 crossover (long when Close > MA20, flat when Close < MA20) ---
        data["MA20"] = data["Close"].rolling(20).mean()
        data["Signal"] = 0
        data.loc[data["Close"] > data["MA20"], "Signal"] = 1   # Above MA -> bullish
        data.loc[data["Close"] < data["MA20"], "Signal"] = -1  # Below MA -> bearish

        # --- 4) Backtest loop: start with cash, buy full position on Signal=1, sell on Signal=-1 ---
        cash = 10000
        position = 0
        trade_log = []  # List of (date, "BUY"|"SELL", price)

        for i in range(len(data)):
            date = data.index[i]
            price = data["Close"].iloc[i]
            signal = data["Signal"].iloc[i]

            if signal == 1 and position == 0:
                # Go long: spend all cash at current close
                position = cash / price
                cash = 0
                trade_log.append((date, "BUY", price))

            elif signal == -1 and position > 0:
                # Exit long: sell all shares for cash
                cash = position * price
                position = 0
                trade_log.append((date, "SELL", price))

        # --- 5) Results: final portfolio value and list of trades ---
        final_value = cash + position * data["Close"].iloc[-1]
        print("\nFinal portfolio value:", final_value)
        print("Trade log:")
        for t in trade_log:
            print(t)

        # --- 6) Plot: candlesticks + volume + MA20 + buy/sell markers ---
        plot_backtest(ticker, data, trade_log, title=f"{ticker} MA20 crossover | {source}")

    else:
        print("No data received. Check network or add data/AAPL.csv.")
