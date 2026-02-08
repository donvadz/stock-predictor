import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from cache import stock_data_cache
from config import LOOKBACK_DAYS_SHORT, LOOKBACK_DAYS_LONG, STOCK_DATA_CACHE_TTL

# Suppress yfinance error messages for ETFs without fundamentals
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

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


def fetch_fundamental_data(ticker: str) -> Optional[dict]:
    """
    Fetch fundamental data from Yahoo Finance.

    Returns dict with company info, analyst recommendations, earnings, etc.
    Returns None if data unavailable.
    """
    ticker = ticker.upper()

    # Check cache
    cache_key = f"{ticker}:fundamentals"
    cached = stock_data_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

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
    stock_data_cache.set(cache_key, df, STOCK_DATA_CACHE_TTL)

    return df
