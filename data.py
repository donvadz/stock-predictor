import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from cache import stock_data_cache
from config import LOOKBACK_DAYS_SHORT, LOOKBACK_DAYS_LONG, STOCK_DATA_CACHE_TTL

# Suppress yfinance error messages for ETFs without fundamentals
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# Rate limit handling
_last_request_time = 0
_request_delay = 0.5  # 500ms between requests to avoid rate limiting

# Sector encoding for ML model
SECTOR_ENCODING = {
    "Technology": 1,
    "Healthcare": 2,
    "Financial Services": 3,
    "Consumer Cyclical": 4,
    "Communication Services": 5,
    "Industrials": 6,
    "Consumer Defensive": 7,
    "Energy": 8,
    "Utilities": 9,
    "Real Estate": 10,
    "Basic Materials": 11,
}


def fetch_fundamental_data(ticker: str, max_retries: int = 3) -> Optional[dict]:
    """
    Fetch fundamental data from Yahoo Finance.

    Returns dict with company info, analyst recommendations, earnings, etc.
    Returns None if data unavailable.
    """
    global _last_request_time

    ticker = ticker.upper()

    # Check cache
    cache_key = f"{ticker}:fundamentals"
    cached = stock_data_cache.get(cache_key)
    if cached is not None:
        return cached

    # Rate limiting - ensure minimum delay between requests
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _request_delay:
        time.sleep(_request_delay - elapsed)
    _last_request_time = time.time()

    # Retry with exponential backoff for rate limits
    stock = None
    info = None
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info:
                break  # Success, exit retry loop
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate' in error_msg or 'too many' in error_msg or '429' in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                    logger.warning(f"Rate limited fetching {ticker}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Rate limit exceeded for {ticker} after {max_retries} retries")
                    return None
            else:
                logger.warning(f"Error fetching fundamentals for {ticker}: {e}")
                return None

    if not info:
        return None

    try:
        # Extract company info
        fundamentals = {
            # Name
            "name": info.get("shortName") or info.get("longName") or ticker,

            # Valuation metrics
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),

            # Size metrics
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),

            # Profitability
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),

            # Growth
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),

            # Dividends
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),

            # Financial health
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),

            # Trading info
            "beta": info.get("beta"),
            "fifty_two_week_change": info.get("52WeekChange"),
            "short_ratio": info.get("shortRatio"),
            "short_percent": info.get("shortPercentOfFloat"),

            # Sector (encoded)
            "sector": SECTOR_ENCODING.get(info.get("sector"), 0),
        }

        # Get analyst recommendations
        try:
            recs = stock.recommendations
            if recs is not None and len(recs) > 0:
                recent_recs = recs.tail(30)  # Last 30 recommendations
                # Count recommendation types
                rec_counts = recent_recs["To Grade"].value_counts()
                total_recs = len(recent_recs)

                # Calculate sentiment score (-1 to 1)
                bullish = sum(rec_counts.get(g, 0) for g in ["Buy", "Strong Buy", "Outperform", "Overweight"])
                bearish = sum(rec_counts.get(g, 0) for g in ["Sell", "Strong Sell", "Underperform", "Underweight"])

                if total_recs > 0:
                    fundamentals["analyst_sentiment"] = (bullish - bearish) / total_recs
                    fundamentals["analyst_count"] = total_recs
                else:
                    fundamentals["analyst_sentiment"] = 0
                    fundamentals["analyst_count"] = 0
            else:
                fundamentals["analyst_sentiment"] = 0
                fundamentals["analyst_count"] = 0
        except Exception:
            fundamentals["analyst_sentiment"] = 0
            fundamentals["analyst_count"] = 0

        # Get earnings surprise
        try:
            earnings = stock.earnings_history
            if earnings is not None and len(earnings) > 0:
                recent = earnings.iloc[-1]  # Most recent
                if pd.notna(recent.get("epsActual")) and pd.notna(recent.get("epsEstimate")):
                    if recent["epsEstimate"] != 0:
                        fundamentals["earnings_surprise"] = (
                            (recent["epsActual"] - recent["epsEstimate"]) / abs(recent["epsEstimate"])
                        )
                    else:
                        fundamentals["earnings_surprise"] = 0
                else:
                    fundamentals["earnings_surprise"] = 0
            else:
                fundamentals["earnings_surprise"] = 0
        except Exception:
            fundamentals["earnings_surprise"] = 0

        # Get news count (proxy for attention/momentum)
        try:
            news = stock.news
            fundamentals["news_count"] = len(news) if news else 0
        except Exception:
            fundamentals["news_count"] = 0

        # Get institutional ownership
        try:
            holders = stock.institutional_holders
            if holders is not None and len(holders) > 0:
                fundamentals["institutional_holders"] = len(holders)
            else:
                fundamentals["institutional_holders"] = 0
        except Exception:
            fundamentals["institutional_holders"] = 0

        # Cache the result
        stock_data_cache.set(cache_key, fundamentals, STOCK_DATA_CACHE_TTL)

        return fundamentals

    except Exception:
        return None


def fetch_stock_data(ticker: str, prediction_days: int = 5) -> Optional[pd.DataFrame]:
    """
    Fetch daily stock data from Yahoo Finance.

    Uses 1 year of data for short-term predictions (1-7 days)
    and 3 years for longer-term predictions (8+ days).

    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    Returns None if no data available.
    """
    ticker = ticker.upper()

    # Choose lookback period based on prediction horizon
    lookback_days = LOOKBACK_DAYS_SHORT if prediction_days <= 7 else LOOKBACK_DAYS_LONG

    # Check cache first (include lookback in cache key)
    cache_key = f"{ticker}:{lookback_days}"
    cached = stock_data_cache.get(cache_key)
    if cached is not None:
        return cached

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    # Rate limiting
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _request_delay:
        time.sleep(_request_delay - elapsed)
    _last_request_time = time.time()

    # Fetch from Yahoo Finance with retry
    df = None
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval="1d")
            break
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate' in error_msg or 'too many' in error_msg:
                if attempt < 2:
                    wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    continue
            return None

    if df is None or df.empty:
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
    stock_data_cache.set(cache_key, df, STOCK_DATA_CACHE_TTL)

    return df


def fetch_stock_data_extended(ticker: str, years: int = 10) -> Optional[pd.DataFrame]:
    """
    Fetch extended historical stock data from Yahoo Finance.

    Supports up to 10 years of daily data for long-term backtesting.

    Args:
        ticker: Stock ticker symbol
        years: Number of years of history (1-10)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        Returns None if no data available.
    """
    ticker = ticker.upper()
    years = min(max(years, 1), 10)  # Clamp to 1-10 years
    lookback_days = years * 365

    # Check cache first
    cache_key = f"{ticker}:extended:{years}y"
    cached = stock_data_cache.get(cache_key)
    if cached is not None:
        return cached

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    # Rate limiting
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _request_delay:
        time.sleep(_request_delay - elapsed)
    _last_request_time = time.time()

    # Fetch from Yahoo Finance with retry
    df = None
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval="1d")
            break
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate' in error_msg or 'too many' in error_msg:
                if attempt < 2:
                    wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    continue
            return None

    if df is None or df.empty:
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

    # Cache the result (longer TTL for extended data since it changes less)
    stock_data_cache.set(cache_key, df, STOCK_DATA_CACHE_TTL * 2)

    return df
