"""
Composite Score Backtest

Validates whether the composite scoring system has predictive value by:
1. Quintile Analysis: Compare returns across score quintiles
2. Grade Performance: Compare returns by letter grade
3. Sector-Relative Performance: Do high scores beat their sector?
4. Factor Validation: Test each factor's individual predictive power

Note: This uses current fundamentals with historical returns, which has
look-ahead bias. For production use, you'd want point-in-time fundamentals.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from cache import prediction_cache
from data import fetch_stock_data, fetch_fundamental_data, SECTOR_ENCODING
from composite_score import (
    calculate_composite_score,
    rank_all_stocks,
    COMPOSITE_STOCK_LIST,
)

logger = logging.getLogger(__name__)

# Cache TTL for backtest results (6 hours)
BACKTEST_CACHE_TTL = 21600


def _calculate_returns(ticker: str, periods: List[int] = [30, 90, 180, 365]) -> Dict[str, float]:
    """
    Calculate historical returns for various periods.

    Args:
        ticker: Stock ticker
        periods: List of lookback periods in days

    Returns:
        Dict mapping period to return percentage
    """
    df = fetch_stock_data(ticker, prediction_days=30)  # Gets 3 years of data

    if df is None or len(df) < 30:
        return {}

    returns = {}
    current_price = float(df["close"].iloc[-1])

    for days in periods:
        if len(df) >= days:
            past_price = float(df["close"].iloc[-days])
            if past_price > 0:
                returns[f"{days}d"] = ((current_price - past_price) / past_price) * 100

    return returns


def _calculate_volatility(ticker: str, period: int = 90) -> Optional[float]:
    """Calculate annualized volatility for a stock."""
    df = fetch_stock_data(ticker, prediction_days=30)

    if df is None or len(df) < period:
        return None

    # Calculate daily returns
    prices = df["close"].values[-period:]
    daily_returns = np.diff(prices) / prices[:-1]

    # Annualized volatility
    volatility = np.std(daily_returns) * np.sqrt(252) * 100
    return volatility


def _calculate_sharpe_ratio(returns: float, volatility: float, risk_free_rate: float = 4.0) -> Optional[float]:
    """Calculate Sharpe ratio (annualized)."""
    if volatility is None or volatility == 0:
        return None

    # Annualize the return if it's a 90-day return
    annualized_return = returns * (365 / 90)
    excess_return = annualized_return - risk_free_rate

    return excess_return / volatility


def run_quintile_analysis(
    stocks: Optional[List[str]] = None,
    return_period: str = "90d",
    max_workers: int = 8,
    progress_callback: Optional[callable] = None,
) -> Dict:
    """
    Divide stocks into quintiles by composite score and compare returns.

    This tests the core hypothesis: do higher-scoring stocks outperform?

    Args:
        stocks: List of stocks to analyze (default: COMPOSITE_STOCK_LIST)
        return_period: Return period to analyze ("30d", "90d", "180d", "365d")
        max_workers: Parallel workers for data fetching
        progress_callback: Optional callback(current, total, message)

    Returns:
        Dict with quintile analysis results
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST[:200]  # Use top 200 for speed

    cache_key = f"composite_quintile:{len(stocks)}:{return_period}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Get composite scores
    if progress_callback:
        progress_callback(0, 100, "Calculating composite scores...")

    all_ranked = rank_all_stocks(stocks, max_workers=max_workers)

    if not all_ranked:
        return {
            "error": "Failed to calculate composite scores. Yahoo Finance API may be rate limiting requests. Please wait 5-10 minutes and try again.",
            "rate_limited": True
        }

    # Fetch returns for all stocks
    stock_data = []
    total = len(all_ranked)

    def fetch_returns(stock_info):
        ticker = stock_info["ticker"]
        returns = _calculate_returns(ticker)
        volatility = _calculate_volatility(ticker)
        return {
            **stock_info,
            "returns": returns,
            "volatility": volatility,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_returns, s): s for s in all_ranked}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if progress_callback:
                progress_callback(completed, total, f"Fetching returns ({completed}/{total})")

            try:
                result = future.result()
                if result["returns"].get(return_period) is not None:
                    stock_data.append(result)
            except Exception as e:
                logger.warning(f"Failed to fetch returns: {e}")

    if len(stock_data) < 20:
        return {"error": "Insufficient data for quintile analysis"}

    # Sort by composite score and divide into quintiles
    stock_data.sort(key=lambda x: x["composite_score"], reverse=True)
    quintile_size = len(stock_data) // 5

    quintiles = []
    for i in range(5):
        start = i * quintile_size
        end = start + quintile_size if i < 4 else len(stock_data)
        quintile_stocks = stock_data[start:end]

        returns_list = [s["returns"][return_period] for s in quintile_stocks]
        scores = [s["composite_score"] for s in quintile_stocks]
        volatilities = [s["volatility"] for s in quintile_stocks if s["volatility"]]

        avg_return = np.mean(returns_list)
        avg_score = np.mean(scores)
        avg_volatility = np.mean(volatilities) if volatilities else None

        # Calculate Sharpe for the quintile
        sharpe = None
        if avg_volatility and avg_volatility > 0:
            annualized_return = avg_return * (365 / int(return_period.replace("d", "")))
            sharpe = (annualized_return - 4.0) / avg_volatility

        quintiles.append({
            "quintile": i + 1,
            "label": ["Top 20%", "20-40%", "40-60%", "60-80%", "Bottom 20%"][i],
            "stocks_count": len(quintile_stocks),
            "avg_composite_score": round(avg_score, 1),
            "score_range": f"{min(scores):.0f}-{max(scores):.0f}",
            "avg_return": round(avg_return, 2),
            "median_return": round(np.median(returns_list), 2),
            "best_return": round(max(returns_list), 2),
            "worst_return": round(min(returns_list), 2),
            "avg_volatility": round(avg_volatility, 1) if avg_volatility else None,
            "sharpe_ratio": round(sharpe, 2) if sharpe else None,
            "win_rate": round(sum(1 for r in returns_list if r > 0) / len(returns_list) * 100, 1),
            "top_stocks": [
                {"ticker": s["ticker"], "score": s["composite_score"], "return": round(s["returns"][return_period], 1)}
                for s in quintile_stocks[:3]
            ],
        })

    # Calculate spread (Q1 - Q5)
    spread = quintiles[0]["avg_return"] - quintiles[4]["avg_return"]

    # Determine if the scoring system has predictive value
    monotonic = all(
        quintiles[i]["avg_return"] >= quintiles[i + 1]["avg_return"]
        for i in range(4)
    )

    if spread > 10 and monotonic:
        verdict = "STRONG"
        verdict_detail = "Strong evidence that higher scores predict better returns"
    elif spread > 5:
        verdict = "MODERATE"
        verdict_detail = "Moderate evidence of predictive value"
    elif spread > 0:
        verdict = "WEAK"
        verdict_detail = "Weak but positive relationship between scores and returns"
    else:
        verdict = "NONE"
        verdict_detail = "No clear relationship between scores and returns in this period"

    result = {
        "analysis_type": "quintile",
        "stocks_analyzed": len(stock_data),
        "return_period": return_period,
        "quintiles": quintiles,
        "summary": {
            "top_quintile_return": quintiles[0]["avg_return"],
            "bottom_quintile_return": quintiles[4]["avg_return"],
            "spread": round(spread, 2),
            "is_monotonic": monotonic,
            "verdict": verdict,
            "verdict_detail": verdict_detail,
        },
        "interpretation": [
            f"Top 20% by score returned {quintiles[0]['avg_return']:.1f}% vs {quintiles[4]['avg_return']:.1f}% for bottom 20%",
            f"Spread of {spread:.1f}% between top and bottom quintiles",
            f"Top quintile win rate: {quintiles[0]['win_rate']:.0f}%",
        ],
    }

    prediction_cache.set(cache_key, result, BACKTEST_CACHE_TTL)
    return result


def run_grade_analysis(
    stocks: Optional[List[str]] = None,
    return_period: str = "90d",
    max_workers: int = 8,
) -> Dict:
    """
    Compare performance by letter grade (A/B/C/D/F).

    Args:
        stocks: List of stocks to analyze
        return_period: Return period ("30d", "90d", "180d", "365d")

    Returns:
        Dict with grade-level performance analysis
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST[:200]

    cache_key = f"composite_grade:{len(stocks)}:{return_period}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Get composite scores
    all_ranked = rank_all_stocks(stocks, max_workers=max_workers)

    if not all_ranked:
        return {
            "error": "Failed to calculate composite scores. Yahoo Finance API may be rate limiting requests. Please wait 5-10 minutes and try again.",
            "rate_limited": True
        }

    # Fetch returns
    def fetch_returns(stock_info):
        ticker = stock_info["ticker"]
        returns = _calculate_returns(ticker)
        return {**stock_info, "returns": returns}

    stock_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_returns, s): s for s in all_ranked}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result["returns"].get(return_period) is not None:
                    stock_data.append(result)
            except Exception:
                pass

    # Group by grade
    grades = {"A": [], "B": [], "C": [], "D": [], "F": []}
    for stock in stock_data:
        grade = stock.get("grade", "F")
        if grade in grades:
            grades[grade].append(stock)

    grade_results = []
    for grade in ["A", "B", "C", "D", "F"]:
        stocks_in_grade = grades[grade]
        if not stocks_in_grade:
            continue

        returns_list = [s["returns"][return_period] for s in stocks_in_grade]

        grade_results.append({
            "grade": grade,
            "stocks_count": len(stocks_in_grade),
            "avg_return": round(np.mean(returns_list), 2),
            "median_return": round(np.median(returns_list), 2),
            "std_dev": round(np.std(returns_list), 2),
            "win_rate": round(sum(1 for r in returns_list if r > 0) / len(returns_list) * 100, 1),
            "best_performers": [
                {"ticker": s["ticker"], "return": round(s["returns"][return_period], 1)}
                for s in sorted(stocks_in_grade, key=lambda x: x["returns"][return_period], reverse=True)[:3]
            ],
        })

    # Calculate if grades predict returns
    if len(grade_results) >= 2:
        a_return = next((g["avg_return"] for g in grade_results if g["grade"] == "A"), None)
        worst_return = min(g["avg_return"] for g in grade_results)

        if a_return is not None:
            spread = a_return - worst_return
        else:
            spread = 0
    else:
        spread = 0

    result = {
        "analysis_type": "grade",
        "stocks_analyzed": len(stock_data),
        "return_period": return_period,
        "grades": grade_results,
        "summary": {
            "a_grade_return": next((g["avg_return"] for g in grade_results if g["grade"] == "A"), None),
            "spread": round(spread, 2),
        },
    }

    prediction_cache.set(cache_key, result, BACKTEST_CACHE_TTL)
    return result


def run_factor_analysis(
    stocks: Optional[List[str]] = None,
    return_period: str = "90d",
    max_workers: int = 8,
) -> Dict:
    """
    Test each factor's (Growth, Quality, Financial, Valuation) individual predictive power.

    This helps identify which factors are most valuable.
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST[:150]

    cache_key = f"composite_factor:{len(stocks)}:{return_period}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Get composite scores
    all_ranked = rank_all_stocks(stocks, max_workers=max_workers)

    if not all_ranked:
        return {
            "error": "Failed to calculate composite scores. Yahoo Finance API may be rate limiting requests. Please wait 5-10 minutes and try again.",
            "rate_limited": True
        }

    # Fetch returns
    def fetch_returns(stock_info):
        ticker = stock_info["ticker"]
        returns = _calculate_returns(ticker)
        return {**stock_info, "returns": returns}

    stock_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_returns, s): s for s in all_ranked}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result["returns"].get(return_period) is not None:
                    stock_data.append(result)
            except Exception:
                pass

    if len(stock_data) < 20:
        return {"error": "Insufficient data for factor analysis"}

    factors = ["growth_score", "quality_score", "financial_strength_score", "valuation_score"]
    factor_labels = {
        "growth_score": "Growth",
        "quality_score": "Quality",
        "financial_strength_score": "Financial Strength",
        "valuation_score": "Valuation",
    }

    factor_results = []

    for factor in factors:
        # Filter stocks with this factor score
        valid_stocks = [s for s in stock_data if s.get(factor) is not None]

        if len(valid_stocks) < 20:
            continue

        # Sort by this factor and get top/bottom terciles
        valid_stocks.sort(key=lambda x: x[factor], reverse=True)
        tercile_size = len(valid_stocks) // 3

        top_tercile = valid_stocks[:tercile_size]
        bottom_tercile = valid_stocks[-tercile_size:]

        top_returns = [s["returns"][return_period] for s in top_tercile]
        bottom_returns = [s["returns"][return_period] for s in bottom_tercile]

        top_avg = np.mean(top_returns)
        bottom_avg = np.mean(bottom_returns)
        spread = top_avg - bottom_avg

        # Calculate correlation
        scores = [s[factor] for s in valid_stocks]
        returns_list = [s["returns"][return_period] for s in valid_stocks]
        correlation = np.corrcoef(scores, returns_list)[0, 1]

        factor_results.append({
            "factor": factor_labels[factor],
            "factor_key": factor,
            "stocks_analyzed": len(valid_stocks),
            "top_tercile_return": round(top_avg, 2),
            "bottom_tercile_return": round(bottom_avg, 2),
            "spread": round(spread, 2),
            "correlation": round(correlation, 3),
            "predictive_power": "Strong" if spread > 8 else "Moderate" if spread > 3 else "Weak" if spread > 0 else "None",
        })

    # Rank factors by predictive power
    factor_results.sort(key=lambda x: x["spread"], reverse=True)

    result = {
        "analysis_type": "factor",
        "stocks_analyzed": len(stock_data),
        "return_period": return_period,
        "factors": factor_results,
        "best_factor": factor_results[0]["factor"] if factor_results else None,
        "interpretation": [
            f"Best predictor: {factor_results[0]['factor']} with {factor_results[0]['spread']:.1f}% spread" if factor_results else "N/A",
        ],
    }

    prediction_cache.set(cache_key, result, BACKTEST_CACHE_TTL)
    return result


def run_sector_relative_analysis(
    stocks: Optional[List[str]] = None,
    return_period: str = "90d",
    max_workers: int = 8,
) -> Dict:
    """
    Test if high-scoring stocks outperform their sector average.

    This controls for sector-wide movements.
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST[:200]

    cache_key = f"composite_sector_relative:{len(stocks)}:{return_period}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Get composite scores
    all_ranked = rank_all_stocks(stocks, max_workers=max_workers)

    if not all_ranked:
        return {
            "error": "Failed to calculate composite scores. Yahoo Finance API may be rate limiting requests. Please wait 5-10 minutes and try again.",
            "rate_limited": True
        }

    # Fetch returns
    def fetch_returns(stock_info):
        ticker = stock_info["ticker"]
        returns = _calculate_returns(ticker)
        return {**stock_info, "returns": returns}

    stock_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_returns, s): s for s in all_ranked}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result["returns"].get(return_period) is not None:
                    stock_data.append(result)
            except Exception:
                pass

    # Group by sector
    sectors = {}
    for stock in stock_data:
        sector = stock.get("sector", "Unknown")
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(stock)

    sector_results = []

    for sector, sector_stocks in sectors.items():
        if len(sector_stocks) < 5:
            continue

        # Calculate sector average return
        all_returns = [s["returns"][return_period] for s in sector_stocks]
        sector_avg = np.mean(all_returns)

        # Get top and bottom half by score within sector
        sector_stocks.sort(key=lambda x: x["composite_score"], reverse=True)
        mid = len(sector_stocks) // 2

        top_half = sector_stocks[:mid]
        bottom_half = sector_stocks[mid:]

        top_returns = [s["returns"][return_period] for s in top_half]
        bottom_returns = [s["returns"][return_period] for s in bottom_half]

        top_avg = np.mean(top_returns)
        bottom_avg = np.mean(bottom_returns)

        sector_results.append({
            "sector": sector,
            "stocks_count": len(sector_stocks),
            "sector_avg_return": round(sector_avg, 2),
            "top_half_return": round(top_avg, 2),
            "bottom_half_return": round(bottom_avg, 2),
            "alpha": round(top_avg - sector_avg, 2),  # Excess return vs sector
            "spread": round(top_avg - bottom_avg, 2),
        })

    # Sort by spread
    sector_results.sort(key=lambda x: x["spread"], reverse=True)

    # Calculate overall statistics
    avg_alpha = np.mean([s["alpha"] for s in sector_results]) if sector_results else 0
    avg_spread = np.mean([s["spread"] for s in sector_results]) if sector_results else 0

    result = {
        "analysis_type": "sector_relative",
        "stocks_analyzed": len(stock_data),
        "return_period": return_period,
        "sectors": sector_results,
        "summary": {
            "avg_alpha": round(avg_alpha, 2),
            "avg_spread": round(avg_spread, 2),
            "sectors_where_scoring_works": sum(1 for s in sector_results if s["spread"] > 0),
            "total_sectors": len(sector_results),
        },
    }

    prediction_cache.set(cache_key, result, BACKTEST_CACHE_TTL)
    return result


def run_composite_backtest(
    stocks: Optional[List[str]] = None,
    return_period: str = "90d",
    max_workers: int = 8,
    progress_callback: Optional[callable] = None,
) -> Dict:
    """
    Run comprehensive backtest of the composite scoring system.

    Combines all analysis types into a single report.
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST[:200]

    cache_key = f"composite_backtest_full:{len(stocks)}:{return_period}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Run all analyses
    quintile = run_quintile_analysis(stocks, return_period, max_workers, progress_callback)
    grade = run_grade_analysis(stocks, return_period, max_workers)
    factor = run_factor_analysis(stocks, return_period, max_workers)
    sector = run_sector_relative_analysis(stocks, return_period, max_workers)

    # Overall verdict
    quintile_spread = quintile.get("summary", {}).get("spread", 0)
    grade_a_return = grade.get("summary", {}).get("a_grade_return", 0) or 0

    if quintile_spread > 10 and grade_a_return > 5:
        overall_verdict = "VALIDATED"
        verdict_detail = "Composite scoring system shows strong predictive value in this period"
    elif quintile_spread > 5:
        overall_verdict = "PROMISING"
        verdict_detail = "Moderate evidence of predictive value"
    elif quintile_spread > 0:
        overall_verdict = "INCONCLUSIVE"
        verdict_detail = "Weak relationship between scores and returns"
    else:
        overall_verdict = "FAILED"
        verdict_detail = "Scoring system did not predict returns in this period"

    result = {
        "backtest_type": "composite_score",
        "stocks_analyzed": len(stocks),
        "return_period": return_period,
        "analysis_date": datetime.now().isoformat(),
        "overall_verdict": overall_verdict,
        "verdict_detail": verdict_detail,
        "quintile_analysis": quintile,
        "grade_analysis": grade,
        "factor_analysis": factor,
        "sector_analysis": sector,
        "key_findings": [
            f"Top quintile vs bottom quintile spread: {quintile_spread:.1f}%",
            f"Best predictive factor: {factor.get('best_factor', 'N/A')}",
            f"A-grade stocks average return: {grade_a_return:.1f}%",
        ],
        "caveats": [
            "Uses current fundamentals with historical returns (look-ahead bias)",
            "Past performance does not guarantee future results",
            "Market conditions during test period affect results",
            "For production use, point-in-time fundamentals are needed",
        ],
    }

    prediction_cache.set(cache_key, result, BACKTEST_CACHE_TTL)
    return result
