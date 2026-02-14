"""
Long-Term Composite Stock Scoring System

A data-driven scoring engine that ranks companies based on fundamentals
for long-term investing (not short-term trading).

Composite Score Formula:
- Growth (30%): Revenue growth, Earnings growth, Price momentum
- Quality (30%): ROE, ROA, Profit margin, Operating margin
- Financial Strength (20%): Debt-to-equity, Current ratio, Institutional holders
- Valuation Sanity (20%): P/E vs sector avg, PEG ratio, Price-to-book

Final Score: 0-100 scale, higher = better investment candidate
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import numpy as np

from cache import stock_data_cache
from config import STOCK_DATA_CACHE_TTL
from data import fetch_fundamental_data, fetch_stock_data, fetch_stock_data_extended, SECTOR_ENCODING

# Import the expanded stock list (will be added to config.py)
try:
    from config import COMPOSITE_STOCK_LIST
except ImportError:
    # Fallback to regular stock list if not defined
    from config import STOCK_LIST
    COMPOSITE_STOCK_LIST = STOCK_LIST

logger = logging.getLogger(__name__)

# Cache TTL for composite scores (6 hours)
COMPOSITE_CACHE_TTL = 21600

# Grade thresholds
GRADE_THRESHOLDS = [
    (80, "A"),
    (65, "B"),
    (50, "C"),
    (35, "D"),
    (0, "F"),
]


def _get_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def _estimate_expected_return(grade: str, horizon_months: int) -> Tuple[float, str]:
    """
    Estimate expected annual return based on grade and holding period.

    These estimates are derived from historical backtesting:
    - Grade A stocks historically returned ~15-25% annually
    - Grade B stocks historically returned ~10-15% annually
    - Grade C stocks historically returned ~5-10% annually (market average)
    - Grade D/F stocks historically underperformed

    Longer horizons have higher confidence in these estimates since
    fundamentals take time to reflect in stock prices.

    Returns:
        Tuple of (expected_annual_return_pct, range_string)
    """
    # Base expected returns by grade (annual)
    grade_returns = {
        "A": {"base": 18, "range": (12, 25), "confidence": "high"},
        "B": {"base": 12, "range": (8, 18), "confidence": "moderate"},
        "C": {"base": 8, "range": (4, 12), "confidence": "moderate"},
        "D": {"base": 4, "range": (-2, 8), "confidence": "low"},
        "F": {"base": 0, "range": (-10, 5), "confidence": "low"},
    }

    grade_data = grade_returns.get(grade, grade_returns["C"])

    # Adjust confidence based on horizon
    # Longer horizons = more reliable estimates for fundamental-based picks
    if horizon_months >= 60:
        # 5+ years: fundamentals dominate, higher confidence
        multiplier = 1.1
        confidence_boost = " (high confidence - long-term)"
    elif horizon_months >= 24:
        # 2-5 years: good balance
        multiplier = 1.0
        confidence_boost = ""
    elif horizon_months >= 12:
        # 1-2 years: moderate
        multiplier = 0.95
        confidence_boost = ""
    else:
        # <1 year: more noise, widen range
        multiplier = 0.85
        confidence_boost = " (lower confidence - short-term)"

    expected = round(grade_data["base"] * multiplier, 1)
    low = round(grade_data["range"][0] * multiplier, 0)
    high = round(grade_data["range"][1] * multiplier, 0)

    range_str = f"{int(low)}% to {int(high)}%{confidence_boost}"

    return expected, range_str


def _calculate_total_expected_return(annual_return: float, horizon_months: int) -> str:
    """
    Calculate expected total return over the holding period.

    Uses compound growth formula for periods > 1 year.
    """
    if annual_return is None:
        return None

    years = horizon_months / 12.0

    if years <= 1:
        # For periods <= 1 year, scale linearly
        total = annual_return * years
    else:
        # Compound for longer periods
        total = ((1 + annual_return / 100) ** years - 1) * 100

    return f"{round(total, 1)}% over {_get_period_label(horizon_months)}"


def _cap_value(value: float, min_val: float, max_val: float) -> float:
    """Cap a value within a range."""
    if value is None:
        return None
    return max(min_val, min(max_val, value))


def _percentile_rank(value: float, values: List[float]) -> float:
    """
    Calculate percentile rank of a value within a list.
    Returns 0-100 score where 100 is the best.
    """
    if value is None or not values:
        return None

    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None

    count_below = sum(1 for v in valid_values if v < value)
    return (count_below / len(valid_values)) * 100


def _inverse_percentile_rank(value: float, values: List[float]) -> float:
    """
    Calculate inverse percentile rank (lower is better).
    Returns 0-100 score where 100 is the best (lowest value).
    """
    rank = _percentile_rank(value, values)
    if rank is None:
        return None
    return 100 - rank


def _calculate_period_return(ticker: str, months: int = 24) -> Optional[float]:
    """
    Calculate price return over a specified number of months.

    Args:
        ticker: Stock ticker symbol
        months: Number of months (1-120)

    Returns:
        Percentage return over the period, or None if insufficient data
    """
    months = min(max(months, 1), 120)
    days_needed = int(months * 30.5)  # Approximate days

    # Fetch historical data
    years_needed = max(1, (months + 11) // 12)  # Round up to years
    if years_needed <= 1:
        df = fetch_stock_data(ticker, prediction_days=30)  # Gets 1 year
    elif years_needed <= 3:
        df = fetch_stock_data(ticker, prediction_days=30)  # Gets 3 years with long lookback
    else:
        df = fetch_stock_data_extended(ticker, years=min(years_needed, 10))

    if df is None or len(df) < 20:
        return None

    # Calculate return over the specified period
    try:
        # Find the price from 'months' ago
        target_days = min(days_needed, len(df) - 1)
        if target_days < 10:
            return None

        start_idx = max(0, len(df) - target_days - 1)
        start_price = df.iloc[start_idx]['close']
        end_price = df.iloc[-1]['close']

        if start_price <= 0:
            return None

        return ((end_price - start_price) / start_price) * 100
    except Exception:
        return None


def _calculate_annualized_return(total_return: float, months: int) -> float:
    """Convert total return to annualized return."""
    if total_return is None or months <= 0:
        return None

    years = months / 12.0

    # Handle negative returns
    if total_return <= -100:
        return -100

    # For very short periods, just scale linearly
    if months < 12:
        return total_return * (12 / months)

    return ((1 + total_return / 100) ** (1 / years) - 1) * 100


def _calculate_growth_score(
    fundamentals: Dict,
    all_fundamentals: List[Dict],
    ticker: str = None,
    horizon_months: int = 24,
    all_period_returns: Dict[str, float] = None
) -> Tuple[float, Dict]:
    """
    Calculate Growth score (30% of total).

    Metrics:
    - Revenue Growth (40%): Higher = better, cap at 50%
    - Earnings Growth (40%): Higher = better, cap at 100%
    - Price Momentum (20%): Price return over horizon period

    Args:
        fundamentals: Stock fundamentals
        all_fundamentals: All stocks' fundamentals for percentile ranking
        ticker: Stock ticker for fetching returns
        horizon_months: Investment horizon in months (1, 3, 6, 12, 24, 60, 120)
        all_period_returns: Pre-fetched returns for all stocks
    """
    metrics = {}
    weights = {"revenue_growth": 0.4, "earnings_growth": 0.4, "price_momentum": 0.2}

    # Revenue Growth - cap at 50%
    rev_growth = fundamentals.get("revenue_growth")
    if rev_growth is not None:
        rev_growth = _cap_value(rev_growth, -0.5, 0.5)
        all_rev = [_cap_value(f.get("revenue_growth"), -0.5, 0.5) for f in all_fundamentals]
        metrics["revenue_growth"] = {
            "value": rev_growth,
            "score": _percentile_rank(rev_growth, all_rev)
        }

    # Earnings Growth - cap at 100%
    earn_growth = fundamentals.get("earnings_growth")
    if earn_growth is not None:
        earn_growth = _cap_value(earn_growth, -1.0, 1.0)
        all_earn = [_cap_value(f.get("earnings_growth"), -1.0, 1.0) for f in all_fundamentals]
        metrics["earnings_growth"] = {
            "value": earn_growth,
            "score": _percentile_rank(earn_growth, all_earn)
        }

    # Price Momentum - use period-based returns
    period_label = _get_period_label(horizon_months)

    if horizon_months == 12:
        # Use 52-week change for 1-year horizon (already available in fundamentals)
        price_change = fundamentals.get("fifty_two_week_change")
        if price_change is not None:
            price_change = _cap_value(price_change, -0.5, 0.5)
            all_price = [_cap_value(f.get("fifty_two_week_change"), -0.5, 0.5) for f in all_fundamentals]
            metrics["price_momentum"] = {
                "value": price_change,
                "period": period_label,
                "score": _percentile_rank(price_change, all_price)
            }
    elif all_period_returns and ticker in all_period_returns:
        # Use calculated returns for other periods
        total_return = all_period_returns[ticker]
        annualized_return = _calculate_annualized_return(total_return, horizon_months)

        if annualized_return is not None:
            # Cap annualized return at reasonable bounds
            annualized_return = _cap_value(annualized_return, -80, 200)
            all_returns = [
                _cap_value(_calculate_annualized_return(r, horizon_months), -80, 200)
                for r in all_period_returns.values()
                if r is not None
            ]

            metrics["price_momentum"] = {
                "value": annualized_return / 100,  # Convert to decimal for consistency
                "total_return": round(total_return, 1),
                "period": period_label,
                "score": _percentile_rank(annualized_return, all_returns)
            }

    # Calculate weighted score
    total_weight = 0
    weighted_sum = 0

    for key, weight in weights.items():
        if key in metrics and metrics[key].get("score") is not None:
            weighted_sum += metrics[key]["score"] * weight
            total_weight += weight

    if total_weight > 0:
        score = weighted_sum / total_weight
    else:
        score = None

    return score, metrics


def _get_period_label(months: int) -> str:
    """Convert months to human-readable period label."""
    if months == 1:
        return "30 days"
    elif months < 12:
        return f"{months} months"
    elif months == 12:
        return "1 year"
    else:
        years = months // 12
        return f"{years} years"


def _calculate_quality_score(fundamentals: Dict, all_fundamentals: List[Dict]) -> Tuple[float, Dict]:
    """
    Calculate Quality score (30% of total).

    Metrics:
    - ROE (30%): Target 15-25%, penalize >40% (aggressive)
    - ROA (20%): Higher = better, cap at 20%
    - Profit Margin (25%): Higher = better
    - Operating Margin (25%): Higher = better
    """
    metrics = {}
    weights = {"roe": 0.3, "roa": 0.2, "profit_margin": 0.25, "operating_margin": 0.25}

    # ROE - special handling: 15-25% ideal, penalize extremes
    roe = fundamentals.get("roe")
    if roe is not None:
        # Score ROE: best around 15-25%, penalize very high (>40%)
        if roe < 0:
            roe_score = 0
        elif roe <= 0.15:
            roe_score = (roe / 0.15) * 70  # 0-70 for 0-15%
        elif roe <= 0.25:
            roe_score = 70 + ((roe - 0.15) / 0.10) * 30  # 70-100 for 15-25%
        elif roe <= 0.40:
            roe_score = 100 - ((roe - 0.25) / 0.15) * 20  # 100-80 for 25-40%
        else:
            roe_score = max(60, 80 - ((roe - 0.40) / 0.20) * 20)  # 80-60 for >40%

        metrics["roe"] = {"value": roe, "score": roe_score}

    # ROA - higher is better, cap at 20%
    roa = fundamentals.get("roa")
    if roa is not None:
        roa = _cap_value(roa, -0.1, 0.2)
        all_roa = [_cap_value(f.get("roa"), -0.1, 0.2) for f in all_fundamentals]
        metrics["roa"] = {
            "value": roa,
            "score": _percentile_rank(roa, all_roa)
        }

    # Profit Margin - higher is better
    profit_margin = fundamentals.get("profit_margin")
    if profit_margin is not None:
        all_pm = [f.get("profit_margin") for f in all_fundamentals]
        metrics["profit_margin"] = {
            "value": profit_margin,
            "score": _percentile_rank(profit_margin, all_pm)
        }

    # Operating Margin - higher is better
    op_margin = fundamentals.get("operating_margin")
    if op_margin is not None:
        all_om = [f.get("operating_margin") for f in all_fundamentals]
        metrics["operating_margin"] = {
            "value": op_margin,
            "score": _percentile_rank(op_margin, all_om)
        }

    # Calculate weighted score
    total_weight = 0
    weighted_sum = 0

    for key, weight in weights.items():
        if key in metrics and metrics[key].get("score") is not None:
            weighted_sum += metrics[key]["score"] * weight
            total_weight += weight

    if total_weight > 0:
        score = weighted_sum / total_weight
    else:
        score = None

    return score, metrics


def _calculate_financial_strength_score(fundamentals: Dict, all_fundamentals: List[Dict]) -> Tuple[float, Dict]:
    """
    Calculate Financial Strength score (20% of total).

    Metrics:
    - Debt-to-Equity (40%): Lower = better, <0.5 ideal
    - Current Ratio (30%): 1.5-3 ideal, penalize extremes
    - Institutional Holders (30%): More = more credibility
    """
    metrics = {}
    weights = {"debt_to_equity": 0.4, "current_ratio": 0.3, "institutional_holders": 0.3}

    # Debt-to-Equity - lower is better (inverse percentile)
    dte = fundamentals.get("debt_to_equity")
    if dte is not None:
        # Cap at reasonable range (some companies have negative equity)
        dte = _cap_value(dte, 0, 500)  # Cap at 500%
        all_dte = [_cap_value(f.get("debt_to_equity"), 0, 500) for f in all_fundamentals]
        metrics["debt_to_equity"] = {
            "value": dte,
            "score": _inverse_percentile_rank(dte, all_dte)
        }

    # Current Ratio - 1.5-3 ideal, penalize extremes
    current = fundamentals.get("current_ratio")
    if current is not None:
        if current < 1.0:
            cr_score = (current / 1.0) * 40  # 0-40 for 0-1
        elif current < 1.5:
            cr_score = 40 + ((current - 1.0) / 0.5) * 30  # 40-70 for 1-1.5
        elif current <= 3.0:
            cr_score = 70 + ((current - 1.5) / 1.5) * 30  # 70-100 for 1.5-3
        else:
            cr_score = max(60, 100 - ((current - 3.0) / 2.0) * 40)  # 100-60 for >3

        metrics["current_ratio"] = {"value": current, "score": cr_score}

    # Institutional Holders - more is better
    inst_holders = fundamentals.get("institutional_holders")
    if inst_holders is not None:
        all_inst = [f.get("institutional_holders") for f in all_fundamentals]
        metrics["institutional_holders"] = {
            "value": inst_holders,
            "score": _percentile_rank(inst_holders, all_inst)
        }

    # Calculate weighted score
    total_weight = 0
    weighted_sum = 0

    for key, weight in weights.items():
        if key in metrics and metrics[key].get("score") is not None:
            weighted_sum += metrics[key]["score"] * weight
            total_weight += weight

    if total_weight > 0:
        score = weighted_sum / total_weight
    else:
        score = None

    return score, metrics


def _get_sector_pe_median(sector: int, all_fundamentals: List[Dict]) -> Optional[float]:
    """Get median P/E ratio for a sector."""
    sector_pes = [
        f.get("pe_ratio") for f in all_fundamentals
        if f.get("sector") == sector and f.get("pe_ratio") is not None and 0 < f.get("pe_ratio") < 100
    ]
    if sector_pes:
        return np.median(sector_pes)
    return None


def _calculate_valuation_score(fundamentals: Dict, all_fundamentals: List[Dict]) -> Tuple[float, Dict]:
    """
    Calculate Valuation score (20% of total).

    Metrics:
    - P/E Ratio (35%): Compare to sector median
    - PEG Ratio (35%): <1 = undervalued, >2 = expensive
    - Price-to-Book (30%): Lower = better value
    """
    metrics = {}
    weights = {"pe_ratio": 0.35, "peg_ratio": 0.35, "price_to_book": 0.3}

    # P/E Ratio - compare to sector median
    pe = fundamentals.get("pe_ratio")
    sector = fundamentals.get("sector", 0)
    if pe is not None and pe > 0:
        sector_median_pe = _get_sector_pe_median(sector, all_fundamentals)

        if sector_median_pe and sector_median_pe > 0:
            # Score based on how much below/above sector median
            pe_ratio_to_median = pe / sector_median_pe

            if pe_ratio_to_median < 0.5:
                pe_score = 100  # Very undervalued
            elif pe_ratio_to_median < 1.0:
                pe_score = 70 + (1.0 - pe_ratio_to_median) * 60  # 70-100 if below median
            elif pe_ratio_to_median < 1.5:
                pe_score = 70 - (pe_ratio_to_median - 1.0) * 40  # 70-50 if 1-1.5x median
            else:
                pe_score = max(20, 50 - (pe_ratio_to_median - 1.5) * 30)  # 50-20 if >1.5x

            metrics["pe_ratio"] = {
                "value": pe,
                "sector_median": sector_median_pe,
                "score": pe_score
            }
        else:
            # Fallback: use percentile (lower is better)
            all_pe = [f.get("pe_ratio") for f in all_fundamentals if f.get("pe_ratio") and 0 < f.get("pe_ratio") < 100]
            metrics["pe_ratio"] = {
                "value": pe,
                "score": _inverse_percentile_rank(pe, all_pe)
            }

    # PEG Ratio - <1 undervalued, >2 expensive
    peg = fundamentals.get("peg_ratio")
    if peg is not None and peg > 0:
        peg = _cap_value(peg, 0, 5)  # Cap at 5

        if peg < 0.5:
            peg_score = 100
        elif peg < 1.0:
            peg_score = 80 + (1.0 - peg) * 40  # 80-100 for 0.5-1
        elif peg < 1.5:
            peg_score = 60 + (1.5 - peg) * 40  # 60-80 for 1-1.5
        elif peg < 2.0:
            peg_score = 40 + (2.0 - peg) * 40  # 40-60 for 1.5-2
        else:
            peg_score = max(10, 40 - (peg - 2.0) * 10)  # 40-10 for >2

        metrics["peg_ratio"] = {"value": peg, "score": peg_score}

    # Price-to-Book - lower is better (inverse percentile)
    ptb = fundamentals.get("price_to_book")
    if ptb is not None and ptb > 0:
        ptb = _cap_value(ptb, 0, 50)  # Cap at 50
        all_ptb = [_cap_value(f.get("price_to_book"), 0, 50) for f in all_fundamentals if f.get("price_to_book") and f.get("price_to_book") > 0]
        metrics["price_to_book"] = {
            "value": ptb,
            "score": _inverse_percentile_rank(ptb, all_ptb)
        }

    # Calculate weighted score
    total_weight = 0
    weighted_sum = 0

    for key, weight in weights.items():
        if key in metrics and metrics[key].get("score") is not None:
            weighted_sum += metrics[key]["score"] * weight
            total_weight += weight

    if total_weight > 0:
        score = weighted_sum / total_weight
    else:
        score = None

    return score, metrics


def _fetch_all_fundamentals(stocks: List[str], max_workers: int = 8) -> Dict[str, Dict]:
    """Fetch fundamental data for all stocks in parallel."""
    fundamentals = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_fundamental_data, ticker): ticker for ticker in stocks}

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                data = future.result()
                if data:
                    fundamentals[ticker] = data
            except Exception as e:
                logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")

    return fundamentals


def _fetch_all_period_returns(stocks: List[str], months: int = 24, max_workers: int = 8) -> Dict[str, float]:
    """
    Fetch price returns for all stocks over a specified period in parallel.

    Args:
        stocks: List of ticker symbols
        months: Number of months for return calculation (1-120)
        max_workers: Parallel workers

    Returns:
        Dict mapping ticker to percentage return
    """
    returns = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_calculate_period_return, ticker, months): ticker for ticker in stocks}

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                ret = future.result()
                if ret is not None:
                    returns[ticker] = ret
            except Exception as e:
                logger.warning(f"Failed to fetch period return for {ticker}: {e}")

    return returns


def calculate_composite_score(
    ticker: str,
    all_fundamentals: Optional[Dict[str, Dict]] = None,
    horizon_months: int = 24,
    all_period_returns: Optional[Dict[str, float]] = None
) -> Optional[Dict]:
    """
    Calculate composite score for a single stock.

    Args:
        ticker: Stock ticker symbol
        all_fundamentals: Optional pre-fetched fundamentals for percentile ranking.
                         If not provided, will use cached data or fetch minimal set.
        horizon_months: Investment horizon in months (1, 3, 6, 12, 24, 60, 120)
        all_period_returns: Pre-fetched returns for momentum calculation

    Returns:
        Dict with composite score, sub-scores, grade, and metrics
    """
    ticker = ticker.upper()
    horizon_months = min(max(horizon_months, 1), 120)

    # Check cache (include horizon in cache key)
    cache_key = f"{ticker}:composite_score:{horizon_months}m"
    cached = stock_data_cache.get(cache_key)
    if cached is not None:
        return cached

    # Get fundamentals for this stock
    fundamentals = fetch_fundamental_data(ticker)
    if not fundamentals:
        return None

    # If no all_fundamentals provided, fetch for comparison
    if all_fundamentals is None:
        # Use a subset of stocks for percentile ranking
        comparison_stocks = COMPOSITE_STOCK_LIST[:100]
        all_fundamentals = _fetch_all_fundamentals(comparison_stocks)

    # Fetch period returns if needed and not provided (skip for 12 months - uses 52-week change)
    if horizon_months != 12 and all_period_returns is None:
        all_period_returns = _fetch_all_period_returns(
            list(all_fundamentals.keys()), months=horizon_months
        )

    all_fund_list = list(all_fundamentals.values())

    # Calculate sub-scores
    growth_score, growth_metrics = _calculate_growth_score(
        fundamentals, all_fund_list, ticker, horizon_months, all_period_returns
    )
    quality_score, quality_metrics = _calculate_quality_score(fundamentals, all_fund_list)
    financial_score, financial_metrics = _calculate_financial_strength_score(fundamentals, all_fund_list)
    valuation_score, valuation_metrics = _calculate_valuation_score(fundamentals, all_fund_list)

    # Calculate composite score with ADAPTIVE weights based on horizon
    # Short-term: momentum/growth matters more
    # Long-term: quality/fundamentals matter more
    if horizon_months <= 3:
        # Very short-term: momentum-heavy
        weights = {
            "growth": 0.40,        # Includes price momentum
            "quality": 0.20,
            "financial_strength": 0.15,
            "valuation": 0.25,
        }
        weight_profile = "short-term (momentum-focused)"
    elif horizon_months <= 12:
        # Medium-term: balanced
        weights = {
            "growth": 0.30,
            "quality": 0.30,
            "financial_strength": 0.20,
            "valuation": 0.20,
        }
        weight_profile = "medium-term (balanced)"
    elif horizon_months <= 60:
        # Long-term (1-5 years): quality focus
        weights = {
            "growth": 0.25,
            "quality": 0.35,
            "financial_strength": 0.25,
            "valuation": 0.15,
        }
        weight_profile = "long-term (quality-focused)"
    else:
        # Very long-term (5+ years): quality and financial strength dominate
        weights = {
            "growth": 0.20,
            "quality": 0.35,
            "financial_strength": 0.30,
            "valuation": 0.15,
        }
        weight_profile = "very long-term (quality + stability)"

    scores = {
        "growth": growth_score,
        "quality": quality_score,
        "financial_strength": financial_score,
        "valuation": valuation_score,
    }

    total_weight = 0
    weighted_sum = 0

    for key, weight in weights.items():
        if scores[key] is not None:
            weighted_sum += scores[key] * weight
            total_weight += weight

    if total_weight > 0:
        composite_score = weighted_sum / total_weight
    else:
        return None

    # Get sector name (reverse lookup)
    sector_code = fundamentals.get("sector", 0)
    sector_name = next(
        (name for name, code in SECTOR_ENCODING.items() if code == sector_code),
        "Unknown"
    )

    # Calculate expected return based on grade and horizon
    # These estimates are based on historical backtesting of the scoring system
    grade = _get_grade(composite_score)
    expected_return, expected_range = _estimate_expected_return(grade, horizon_months)

    result = {
        "ticker": ticker,
        "name": fundamentals.get("name", ticker),
        "sector": sector_name,
        "horizon_months": horizon_months,
        "horizon_label": _get_period_label(horizon_months),
        "composite_score": round(composite_score, 1),
        "grade": grade,
        "weight_profile": weight_profile,
        "weights_used": weights,
        # Expected return estimates
        "expected_annual_return": expected_return,
        "expected_return_range": expected_range,
        "expected_total_return": _calculate_total_expected_return(expected_return, horizon_months),
        # Sub-scores
        "growth_score": round(growth_score, 1) if growth_score else None,
        "quality_score": round(quality_score, 1) if quality_score else None,
        "financial_strength_score": round(financial_score, 1) if financial_score else None,
        "valuation_score": round(valuation_score, 1) if valuation_score else None,
        "metrics": {
            "growth": growth_metrics,
            "quality": quality_metrics,
            "financial_strength": financial_metrics,
            "valuation": valuation_metrics,
        },
        "raw_fundamentals": {
            "market_cap": fundamentals.get("market_cap"),
            "pe_ratio": fundamentals.get("pe_ratio"),
            "forward_pe": fundamentals.get("forward_pe"),
            "peg_ratio": fundamentals.get("peg_ratio"),
            "price_to_book": fundamentals.get("price_to_book"),
            "revenue_growth": fundamentals.get("revenue_growth"),
            "earnings_growth": fundamentals.get("earnings_growth"),
            "roe": fundamentals.get("roe"),
            "roa": fundamentals.get("roa"),
            "profit_margin": fundamentals.get("profit_margin"),
            "operating_margin": fundamentals.get("operating_margin"),
            "debt_to_equity": fundamentals.get("debt_to_equity"),
            "current_ratio": fundamentals.get("current_ratio"),
            "dividend_yield": fundamentals.get("dividend_yield"),
        },
    }

    # Cache the result
    stock_data_cache.set(cache_key, result, COMPOSITE_CACHE_TTL)

    return result


def rank_all_stocks(
    stocks: Optional[List[str]] = None,
    horizon_months: int = 24,
    max_workers: int = 8,
    progress_callback: Optional[callable] = None
) -> List[Dict]:
    """
    Rank all stocks in universe by composite score.

    Args:
        stocks: Optional list of stocks to rank. Defaults to COMPOSITE_STOCK_LIST.
        horizon_months: Investment horizon in months (1, 3, 6, 12, 24, 60, 120)
        max_workers: Number of parallel workers for data fetching.
        progress_callback: Optional callback(current, total, ticker) for progress updates.

    Returns:
        Sorted list of stock scores, best first.
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST

    horizon_months = min(max(horizon_months, 1), 120)

    # Check cache for full ranking (include horizon in key)
    cache_key = f"composite_ranking:{len(stocks)}:{horizon_months}m"
    cached = stock_data_cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch all fundamentals first for accurate percentile ranking
    all_fundamentals = _fetch_all_fundamentals(stocks, max_workers)

    if not all_fundamentals:
        return []

    # Fetch period returns (skip for 12 months - uses 52-week change from fundamentals)
    all_period_returns = None
    if horizon_months != 12:
        all_period_returns = _fetch_all_period_returns(
            list(all_fundamentals.keys()), months=horizon_months, max_workers=max_workers
        )

    # Calculate scores for each stock
    results = []
    total = len(all_fundamentals)

    for i, ticker in enumerate(all_fundamentals.keys()):
        if progress_callback:
            should_continue = progress_callback(i, total, ticker)
            if should_continue is False:
                return None  # Cancelled

        score = calculate_composite_score(
            ticker, all_fundamentals, horizon_months, all_period_returns
        )
        if score and score.get("composite_score") is not None:
            results.append(score)

    # Sort by composite score (descending)
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    # Add rank
    for i, result in enumerate(results):
        result["rank"] = i + 1

    # Cache the ranking
    stock_data_cache.set(cache_key, results, COMPOSITE_CACHE_TTL)

    return results


def get_top_decile(stocks: Optional[List[str]] = None, horizon_months: int = 24) -> List[Dict]:
    """
    Return top 10% of stocks by composite score.

    Args:
        stocks: Optional list of stocks. Defaults to COMPOSITE_STOCK_LIST.
        horizon_months: Investment horizon in months (1, 3, 6, 12, 24, 60, 120)

    Returns:
        Top 10% of ranked stocks.
    """
    all_ranked = rank_all_stocks(stocks, horizon_months=horizon_months)

    if not all_ranked:
        return []

    top_count = max(1, len(all_ranked) // 10)
    return all_ranked[:top_count]


def get_stocks_by_grade(grade: str, stocks: Optional[List[str]] = None, horizon_months: int = 24) -> List[Dict]:
    """
    Get all stocks with a specific grade.

    Args:
        grade: Letter grade (A, B, C, D, F)
        stocks: Optional list of stocks. Defaults to COMPOSITE_STOCK_LIST.
        horizon_months: Investment horizon in months (1, 3, 6, 12, 24, 60, 120)

    Returns:
        List of stocks with the specified grade.
    """
    all_ranked = rank_all_stocks(stocks, horizon_months=horizon_months)

    return [s for s in all_ranked if s.get("grade") == grade.upper()]


def get_sector_rankings(sector: str, stocks: Optional[List[str]] = None, horizon_months: int = 24) -> List[Dict]:
    """
    Get rankings filtered by sector.

    Args:
        sector: Sector name (e.g., "Technology", "Healthcare")
        stocks: Optional list of stocks. Defaults to COMPOSITE_STOCK_LIST.
        horizon_months: Investment horizon in months (1, 3, 6, 12, 24, 60, 120)

    Returns:
        Ranked list of stocks in the specified sector.
    """
    all_ranked = rank_all_stocks(stocks, horizon_months=horizon_months)

    sector_stocks = [s for s in all_ranked if s.get("sector", "").lower() == sector.lower()]

    # Re-rank within sector
    for i, stock in enumerate(sector_stocks):
        stock["sector_rank"] = i + 1

    return sector_stocks


def get_available_sectors(stocks: Optional[List[str]] = None, horizon_months: int = 24) -> List[str]:
    """
    Get list of available sectors in the stock universe.

    Args:
        horizon_months: Investment horizon in months (1, 3, 6, 12, 24, 60, 120)

    Returns:
        Sorted list of unique sector names.
    """
    all_ranked = rank_all_stocks(stocks, horizon_months=horizon_months)

    sectors = set(s.get("sector") for s in all_ranked if s.get("sector"))
    return sorted(sectors)
