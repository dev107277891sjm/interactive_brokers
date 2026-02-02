import os
import time
import pandas as pd
import yfinance as yf

DATA_DIR = "data"


def normalize_ohlcv(df):
    """Ensure OHLCV columns and DatetimeIndex for plotting (yfinance uses MultiIndex)."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]
    df.index = pd.to_datetime(df.index)
    return df

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

def plot_backtest(ticker, data, trade_log, title=None):
    """TradingView-style chart: candlesticks, volume, MA20, buy/sell markers."""
    import mplfinance as mpf

    df = data.copy()
    df.index = pd.to_datetime(df.index)

    # Series for buy/sell markers (same index as df, NaN where no trade)
    buy_prices = pd.Series(float("nan"), index=df.index)
    sell_prices = pd.Series(float("nan"), index=df.index)
    for date, action, price in trade_log:
        if date in buy_prices.index:
            if action == "BUY":
                buy_prices.loc[date] = price
            else:
                sell_prices.loc[date] = price

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
    ticker, start, end = "AAPL", "2023-01-01", "2024-01-01"
    data, source = fetch_data(ticker, start, end)

    if data is not None and not data.empty:
        data = normalize_ohlcv(data)
        print(f"Data source: {source}")
        print(f"Rows: {len(data)}, columns: {list(data.columns)}\n")
        print("First 10 rows:")
        print(data.head(10))
        if source == "sample (no network data)":
            print("\nTip: Check VPN/firewall for Yahoo, or add data/AAPL.csv for offline use.")

        data["MA20"] = data["Close"].rolling(20).mean()
        data["Signal"] = 0
        data.loc[data["Close"] > data["MA20"], "Signal"] = 1
        data.loc[data["Close"] < data["MA20"], "Signal"] = -1

        cash = 10000
        position = 0
        trade_log = []  # (date, "BUY"|"SELL", price)

        for i in range(len(data)):
            date = data.index[i]
            price = data["Close"].iloc[i]
            signal = data["Signal"].iloc[i]

            if signal == 1 and position == 0:
                position = cash / price
                cash = 0
                trade_log.append((date, "BUY", price))

            elif signal == -1 and position > 0:
                cash = position * price
                position = 0
                trade_log.append((date, "SELL", price))

        final_value = cash + position * data["Close"].iloc[-1]
        print("\nFinal portfolio value:", final_value)
        print("Trade log:")
        for t in trade_log:
            print(t)

        plot_backtest(ticker, data, trade_log, title=f"{ticker} MA20 crossover | {source}")

    else:
        print("No data received. Check network or add data/AAPL.csv.")
