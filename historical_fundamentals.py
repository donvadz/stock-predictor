"""
Historical Fundamentals Calculator

Calculates fundamental metrics from SEC EDGAR quarterly data.
Returns metrics compatible with composite_score.py format.

Key difference from yfinance:
- yfinance returns CURRENT fundamentals (look-ahead bias in backtests)
- This module returns fundamentals AS OF a specific historical date

Usage:
    from historical_fundamentals import get_fundamentals_at_date, is_sec_covered

    # Get fundamentals as they were known on 2021-03-15
    fundamentals = get_fundamentals_at_date("AAPL", "2021-03-15")
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sec_edgar import get_sec_client, SECEdgarClient

logger = logging.getLogger(__name__)


def _get_ttm_value(
    quarters: List[Dict], as_of_date: str, metric: str
) -> Optional[float]:
    """
    Calculate trailing twelve months (TTM) value for an income statement metric.

    Handles both quarterly (10-Q) and annual (10-K) reports:
    - 10-K reports contain full year data
    - 10-Q reports are quarterly

    For TTM when we have a 10-K followed by 10-Qs:
    TTM = 10-K value + sum(subsequent 10-Qs) - sum(same quarters from prior year)

    Args:
        quarters: List of quarterly data sorted by period_end
        as_of_date: Only use quarters filed before this date
        metric: Metric name (e.g., "revenues", "net_income")

    Returns:
        TTM value or None if insufficient data
    """
    # Filter to quarters filed before as_of_date
    available = [
        q for q in quarters
        if q.get("filing_date") and q["filing_date"] <= as_of_date
    ]

    if not available:
        return None

    # Check if most recent is a 10-K (annual report)
    latest = available[-1]
    if latest.get("form") == "10-K":
        # 10-K contains full year data, use it directly
        return latest.get(metric)

    # For 10-Q, we need to calculate TTM
    # Strategy: Find the most recent 10-K, then:
    # TTM = 10-K + subsequent 10-Qs - corresponding prior year quarters

    # Find the most recent 10-K
    last_10k_idx = None
    for i in range(len(available) - 1, -1, -1):
        if available[i].get("form") == "10-K":
            last_10k_idx = i
            break

    if last_10k_idx is not None:
        last_10k = available[last_10k_idx]
        last_10k_value = last_10k.get(metric)

        if last_10k_value is None:
            return None

        # Get 10-Qs after the 10-K
        subsequent_10qs = [
            q for q in available[last_10k_idx + 1:]
            if q.get("form") == "10-Q"
        ]

        if not subsequent_10qs:
            # No 10-Qs after 10-K, just return 10-K value
            return last_10k_value

        # We need to find the corresponding quarters from the prior year
        # (the quarters that are included in the 10-K that we need to subtract)
        # Get quarter end months for subsequent 10-Qs
        subsequent_values = []
        prior_year_values = []

        for q in subsequent_10qs:
            val = q.get(metric)
            if val is not None:
                subsequent_values.append(val)

                # Find corresponding quarter from prior year
                period_end = q.get("period_end", "")
                if period_end:
                    try:
                        dt = datetime.strptime(period_end, "%Y-%m-%d")
                        prior_year_dt = dt.replace(year=dt.year - 1)
                        prior_year_date = prior_year_dt.strftime("%Y-%m-%d")

                        # Find matching quarter
                        for pq in available:
                            if pq.get("period_end") == prior_year_date:
                                pval = pq.get(metric)
                                if pval is not None:
                                    prior_year_values.append(pval)
                                break
                    except ValueError:
                        pass

        # Calculate TTM: 10-K + new quarters - old quarters
        if len(subsequent_values) == len(prior_year_values):
            ttm = last_10k_value + sum(subsequent_values) - sum(prior_year_values)
            return ttm
        elif subsequent_values:
            # Fallback: can't find all prior year quarters
            # Use approximation: 10-K value (still representative of annual)
            return last_10k_value

    # No 10-K found - try summing 4 consecutive 10-Qs
    q_only = [q for q in available if q.get("form") == "10-Q"]
    if len(q_only) >= 4:
        last_4 = q_only[-4:]
        values = [q.get(metric) for q in last_4]
        if None not in values:
            return sum(values)

    # Last resort: use latest value
    return latest.get(metric)


def _get_yoy_growth(
    quarters: List[Dict], as_of_date: str, metric: str
) -> Optional[float]:
    """
    Calculate year-over-year growth for a metric.

    Compares most recent quarter to same quarter last year.

    Returns:
        Growth rate as decimal (e.g., 0.15 for 15% growth) or None
    """
    # Filter to quarters filed before as_of_date
    available = [
        q for q in quarters
        if q.get("filing_date") and q["filing_date"] <= as_of_date
    ]

    if len(available) < 5:
        return None

    current = available[-1]
    year_ago = available[-5]  # Same quarter last year (4 quarters back)

    current_val = current.get(metric)
    year_ago_val = year_ago.get(metric)

    if current_val is None or year_ago_val is None:
        return None

    if year_ago_val == 0:
        return None

    return (current_val - year_ago_val) / abs(year_ago_val)


def _get_latest_value(
    quarters: List[Dict], as_of_date: str, metric: str
) -> Optional[float]:
    """
    Get the most recent value for a balance sheet metric.

    Balance sheet items (assets, liabilities, equity) are point-in-time
    and don't need TTM calculation.
    """
    # Filter to quarters filed before as_of_date
    available = [
        q for q in quarters
        if q.get("filing_date") and q["filing_date"] <= as_of_date
    ]

    if not available:
        return None

    return available[-1].get(metric)


def get_fundamentals_at_date(
    ticker: str,
    as_of_date: str,
    historical_price: Optional[float] = None,
) -> Optional[Dict]:
    """
    Get fundamental metrics as they were known on a specific date.

    This is the main entry point for historical fundamental analysis.
    Returns a dict compatible with composite_score.py and composite_backtest_grades.py.

    Args:
        ticker: Stock symbol
        as_of_date: Date string (YYYY-MM-DD) - only use filings before this
        historical_price: Stock price on as_of_date (for valuation metrics)

    Returns:
        Dict with fundamental metrics, or None if ticker not SEC-covered
        or insufficient data

    Example return:
        {
            "roe": 0.25,              # Return on equity
            "roa": 0.12,              # Return on assets
            "profit_margin": 0.20,    # Net income / Revenue
            "operating_margin": 0.30, # Operating income / Revenue
            "revenue_growth": 0.15,   # YoY revenue growth
            "earnings_growth": 0.20,  # YoY net income growth
            "debt_to_equity": 1.5,    # Total liabilities / Equity
            "current_ratio": 1.2,     # Current assets / Current liabilities
            "pe_ratio": 25.0,         # Price / EPS (if price provided)
            "price_to_book": 5.0,     # Price / Book value (if price provided)
        }
    """
    client = get_sec_client()

    if not client.is_sec_covered(ticker):
        logger.debug(f"{ticker}: Not SEC-covered (ETF or foreign)")
        return None

    quarters = client.get_quarterly_financials(ticker)
    if not quarters:
        logger.debug(f"{ticker}: No quarterly data available")
        return None

    # Filter to only data available at as_of_date
    available = [
        q for q in quarters
        if q.get("filing_date") and q["filing_date"] <= as_of_date
    ]

    if len(available) < 4:
        logger.debug(f"{ticker}: Insufficient data before {as_of_date}")
        return None

    fundamentals = {}

    # === PROFITABILITY RATIOS ===

    # ROE = Net Income (TTM) / Stockholders Equity
    ttm_net_income = _get_ttm_value(quarters, as_of_date, "net_income")
    equity = _get_latest_value(quarters, as_of_date, "stockholders_equity")

    if ttm_net_income is not None and equity and equity > 0:
        fundamentals["roe"] = ttm_net_income / equity
    else:
        fundamentals["roe"] = None

    # ROA = Net Income (TTM) / Total Assets
    total_assets = _get_latest_value(quarters, as_of_date, "total_assets")

    if ttm_net_income is not None and total_assets and total_assets > 0:
        fundamentals["roa"] = ttm_net_income / total_assets
    else:
        fundamentals["roa"] = None

    # Profit Margin = Net Income (TTM) / Revenue (TTM)
    ttm_revenue = _get_ttm_value(quarters, as_of_date, "revenues")

    if ttm_net_income is not None and ttm_revenue and ttm_revenue > 0:
        fundamentals["profit_margin"] = ttm_net_income / ttm_revenue
    else:
        fundamentals["profit_margin"] = None

    # Operating Margin = Operating Income (TTM) / Revenue (TTM)
    ttm_op_income = _get_ttm_value(quarters, as_of_date, "operating_income")

    if ttm_op_income is not None and ttm_revenue and ttm_revenue > 0:
        fundamentals["operating_margin"] = ttm_op_income / ttm_revenue
    else:
        fundamentals["operating_margin"] = None

    # === GROWTH RATES ===

    # Revenue Growth (YoY)
    fundamentals["revenue_growth"] = _get_yoy_growth(quarters, as_of_date, "revenues")

    # Earnings Growth (YoY)
    fundamentals["earnings_growth"] = _get_yoy_growth(quarters, as_of_date, "net_income")

    # === FINANCIAL HEALTH ===

    # Debt-to-Equity = Total Liabilities / Stockholders Equity
    total_liabilities = _get_latest_value(quarters, as_of_date, "total_liabilities")

    if total_liabilities is not None and equity and equity > 0:
        fundamentals["debt_to_equity"] = (total_liabilities / equity) * 100  # As percentage like yfinance
    else:
        fundamentals["debt_to_equity"] = None

    # Current Ratio = Current Assets / Current Liabilities
    current_assets = _get_latest_value(quarters, as_of_date, "current_assets")
    current_liabilities = _get_latest_value(quarters, as_of_date, "current_liabilities")

    if current_assets and current_liabilities and current_liabilities > 0:
        fundamentals["current_ratio"] = current_assets / current_liabilities
    else:
        fundamentals["current_ratio"] = None

    # === VALUATION METRICS (require historical price) ===

    if historical_price and historical_price > 0:
        # EPS (TTM) - use TTM net income / shares for accuracy
        eps = None

        # First try calculating from TTM net income and shares outstanding
        shares = _get_latest_value(quarters, as_of_date, "shares_outstanding")
        if ttm_net_income and shares and shares > 0:
            eps = ttm_net_income / shares
        else:
            # Fallback to TTM of reported EPS
            ttm_eps = _get_ttm_value(quarters, as_of_date, "eps_basic")
            if ttm_eps is not None:
                eps = ttm_eps
            else:
                # Last resort: use latest single quarter EPS (less accurate)
                eps = _get_latest_value(quarters, as_of_date, "eps_basic")

        # P/E Ratio
        if eps and eps > 0:
            fundamentals["pe_ratio"] = historical_price / eps
        else:
            fundamentals["pe_ratio"] = None

        # Price-to-Book = Price / (Equity / Shares)
        shares = _get_latest_value(quarters, as_of_date, "shares_outstanding")
        if equity and shares and shares > 0:
            book_value_per_share = equity / shares
            if book_value_per_share > 0:
                fundamentals["price_to_book"] = historical_price / book_value_per_share
            else:
                fundamentals["price_to_book"] = None
        else:
            fundamentals["price_to_book"] = None

    else:
        fundamentals["pe_ratio"] = None
        fundamentals["price_to_book"] = None

    # Add some derived metrics for compatibility with composite_score.py
    # These won't be available from SEC data
    fundamentals["peg_ratio"] = None  # Would need analyst estimates
    fundamentals["forward_pe"] = None  # Would need analyst estimates
    fundamentals["analyst_sentiment"] = None  # Not available historically
    fundamentals["analyst_count"] = None
    fundamentals["institutional_holders"] = None
    fundamentals["beta"] = None  # Would need to calculate from price data

    # Log what we found
    available_metrics = sum(1 for v in fundamentals.values() if v is not None)
    logger.debug(f"{ticker} at {as_of_date}: {available_metrics} metrics available")

    return fundamentals


def is_sec_covered(ticker: str) -> bool:
    """Check if a ticker files with SEC (US domestic stocks only)."""
    return get_sec_client().is_sec_covered(ticker)


def get_fundamental_data_coverage(ticker: str) -> Dict:
    """
    Get information about fundamental data coverage for a ticker.

    Returns:
        Dict with:
        - is_covered: Whether ticker files with SEC
        - earliest_date: Earliest available filing date
        - latest_date: Most recent filing date
        - quarters_available: Number of quarters of data
    """
    client = get_sec_client()

    if not client.is_sec_covered(ticker):
        return {
            "is_covered": False,
            "earliest_date": None,
            "latest_date": None,
            "quarters_available": 0,
        }

    quarters = client.get_quarterly_financials(ticker)

    if not quarters:
        return {
            "is_covered": True,
            "earliest_date": None,
            "latest_date": None,
            "quarters_available": 0,
        }

    filing_dates = [q["filing_date"] for q in quarters if q.get("filing_date")]

    return {
        "is_covered": True,
        "earliest_date": min(filing_dates) if filing_dates else None,
        "latest_date": max(filing_dates) if filing_dates else None,
        "quarters_available": len(quarters),
    }
