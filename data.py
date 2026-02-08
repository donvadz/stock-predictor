from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from cache import stock_data_cache
from config import LOOKBACK_DAYS, STOCK_DATA_CACHE_TTL


def fetch_stock_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily stock data from Yahoo Finance.

    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    Returns None if no data available.
    """
    ticker = ticker.upper()

    # Check cache first
    cached = stock_data_cache.get(ticker)
    if cached is not None:
        return cached

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    # Fetch from Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval="1d")
    except Exception:
        return None

    if df.empty:
        return None

    # Rename columns to match expected format
    df = df.reset_index()
    df = df.rename(columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    # Keep only needed columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Cache the result
    stock_data_cache.set(ticker, df, STOCK_DATA_CACHE_TTL)

    return df
