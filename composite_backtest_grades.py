"""
Walk-Forward Grade Validation Test

This module tests whether the fundamental grading system accurately predicts returns.

IMPORTANT: Uses the SAME scoring logic as the live system (scoring.py)
to ensure backtest results reflect real-world performance.

The test:
1. Goes back to historical periods (10Y, 5Y, 2Y, 1Y ago)
2. At each point, calculates grades using ONLY data available then
3. Tracks actual returns of A/B graded stocks
4. Reports success rate: "Did A-grade stocks deliver expected returns?"

This is the TRUE test of whether the grading system works.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from cache import prediction_cache
from data import fetch_stock_data_extended, fetch_fundamental_data
from config import COMPOSITE_STOCK_LIST
from historical_fundamentals import get_fundamentals_at_date, is_sec_covered
from scoring import calculate_composite_score, get_grade, GRADE_THRESHOLDS

logger = logging.getLogger(__name__)

GRADE_VALIDATION_CACHE_TTL = 21600  # 6 hours


# Expected returns by grade and horizon (cumulative over the period)
# Grade thresholds (from scoring.py): A >= 75, B >= 60, C >= 45, D >= 30, F < 30
# These are conservative targets - "success" = meeting or exceeding
EXPECTED_RETURNS = {
    # Horizon in months: {grade: expected_return %}
    3: {"A": 4, "B": 2, "C": 0, "D": -1, "F": -3},
    6: {"A": 7, "B": 4, "C": 1, "D": -1, "F": -4},
    12: {"A": 15, "B": 10, "C": 5, "D": 0, "F": -5},
    24: {"A": 30, "B": 20, "C": 10, "D": 3, "F": -8},
    60: {"A": 75, "B": 50, "C": 25, "D": 10, "F": -10},   # 5-year horizon
    120: {"A": 150, "B": 100, "C": 50, "D": 20, "F": -15},  # 10-year horizon
}


@dataclass
class GradeValidationResult:
    """Result of validating a single graded pick."""
    ticker: str
    grade: str
    score: float
    grade_date: str
    expected_return: float
    actual_return: float
    met_expectation: bool
    exceeded_benchmark: bool  # vs SPY
    benchmark_return: float
    # Individual factor scores for correlation analysis
    # Matches composite_score.py factors: growth, quality, financial_strength, valuation
    growth_score: float = 0
    quality_score: float = 0
    financial_score: float = 0
    valuation_score: float = 0


def _get_historical_prices(ticker: str, years: int = 10) -> Optional[Dict[str, float]]:
    """Get historical closing prices indexed by date string."""
    df = fetch_stock_data_extended(ticker, years=years)

    if df is None or len(df) < 100:
        return None

    prices = {}
    for _, row in df.iterrows():
        date_str = row['timestamp'].strftime('%Y-%m-%d')
        prices[date_str] = float(row['close'])

    return prices


def _calculate_point_in_time_score(
    ticker: str,
    prices: Dict[str, float],
    as_of_date: str,
    fundamentals: Optional[Dict],
    horizon_months: int = 12,
    use_historical: bool = False,
) -> Optional[Tuple[float, str, Dict[str, float]]]:
    """
    Calculate composite score and grade using only data available at as_of_date.

    IMPORTANT: Uses the EXACT same scoring logic as the live system (scoring.py)
    to ensure backtest results are representative of real-world performance.

    Returns:
        Tuple of (score, grade, factor_scores) or None if insufficient data
        factor_scores contains: growth_score, quality_score, financial_score, valuation_score
    """
    valid_dates = sorted([d for d in prices.keys() if d <= as_of_date])

    if len(valid_dates) < 252:  # Need 1 year of history
        return None

    # Calculate price momentum (252-day return) as decimal
    current_price = prices[valid_dates[-1]]
    year_ago_price = prices[valid_dates[-252]]
    price_momentum = (current_price - year_ago_price) / year_ago_price  # As decimal, not percentage

    # Use the SHARED scoring module (same as live system)
    result = calculate_composite_score(
        fundamentals=fundamentals or {},
        price_momentum=price_momentum,
        horizon_months=horizon_months,
    )

    if result is None:
        return None

    factor_scores = {
        "growth_score": result.get("growth_score") or 0,
        "quality_score": result.get("quality_score") or 0,
        "financial_score": result.get("financial_score") or 0,
        "valuation_score": result.get("valuation_score") or 0,
        "sentiment_score": result.get("sentiment_score") or 0,  # Short-term only
    }

    return (result["composite_score"], result["grade"], factor_scores)


def run_walk_forward_grade_validation(
    stocks: Optional[List[str]] = None,
    horizon_months: int = 12,
    test_periods: Optional[List[int]] = None,  # Years ago to start testing
    max_workers: int = 4,
    progress_callback: Optional[callable] = None,
    use_historical_fundamentals: bool = True,
) -> Dict:
    """
    Run walk-forward validation of the grading system.

    This tests: "When we graded stocks historically, did they deliver expected returns?"

    Args:
        stocks: Universe of stocks to test
        horizon_months: Holding period to test (3, 6, 12, or 24 months)
        test_periods: Years ago to start testing (default: [10, 5, 2, 1])
        max_workers: Parallel workers
        progress_callback: Optional progress callback
        use_historical_fundamentals: If True, use SEC EDGAR historical data
                                     (avoids look-ahead bias but slower)

    Returns:
        Dict with validation results by period
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST[:200]  # Use top 200 for speed

    if test_periods is None:
        test_periods = [10, 5, 2, 1]  # 10Y, 5Y, 2Y, 1Y ago

    # Validate horizon
    if horizon_months not in EXPECTED_RETURNS:
        horizon_months = 12

    hist_flag = "hist" if use_historical_fundamentals else "yf"
    cache_key = f"grade_validation_v4:{len(stocks)}:{horizon_months}m:{'-'.join(map(str, test_periods))}:{hist_flag}"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    if progress_callback:
        progress_callback(0, 100, "Fetching historical data...")

    # Fetch all historical data (need max years + horizon)
    max_years = max(test_periods) + (horizon_months // 12) + 1

    all_prices = {}
    all_fundamentals = {}  # Only used when not using historical fundamentals
    sec_covered_tickers = set()

    def fetch_data(ticker):
        prices = _get_historical_prices(ticker, years=max_years)
        # When using historical fundamentals, we fetch SEC data per test date
        # For now, just check if ticker is SEC-covered
        if use_historical_fundamentals:
            is_covered = is_sec_covered(ticker)
            fundamentals = None  # Will be fetched per test date
        else:
            is_covered = False
            fundamentals = fetch_fundamental_data(ticker)
        if prices and len(prices) > 252:
            return ticker, prices, fundamentals, is_covered
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_data, ticker): ticker for ticker in stocks}
        completed = 0
        total = len(stocks)
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                all_prices[result[0]] = result[1]
                all_fundamentals[result[0]] = result[2]
                if result[3]:  # is_sec_covered
                    sec_covered_tickers.add(result[0])
            if progress_callback:
                pct = int((completed / total) * 30)
                progress_callback(pct, 100, f"Fetching data ({completed}/{total})...")

    # Get SPY prices for benchmark
    spy_prices = _get_historical_prices("SPY", years=max_years)
    if not spy_prices:
        return {"error": "Could not fetch SPY benchmark data"}

    if progress_callback:
        progress_callback(30, 100, "Running walk-forward validation...")

    # Results by period
    period_results = {}
    # Collect ALL validation results for factor analysis
    all_validation_results_for_factors = []

    for period_idx, years_ago in enumerate(test_periods):
        if progress_callback:
            pct = 30 + int((period_idx / len(test_periods)) * 60)
            progress_callback(pct, 100, f"Testing {years_ago}Y ago period...")

        # Calculate start and end dates for this test period
        end_date = datetime.now() - timedelta(days=horizon_months * 30)  # Need room for horizon
        start_date = end_date - timedelta(days=years_ago * 365)

        # Create quarterly test points within this period
        test_points = []
        current = start_date
        while current < end_date:
            test_points.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=90)  # Quarterly

        if len(test_points) < 2:
            continue

        # Track results for this period
        a_grade_results = []
        b_grade_results = []
        all_grade_results = {"A": [], "B": [], "C": [], "D": [], "F": []}

        for test_date in test_points:
            # Find nearest valid date in price data
            valid_test_date = None
            for offset in range(7):
                check_date = (datetime.strptime(test_date, '%Y-%m-%d') + timedelta(days=offset)).strftime('%Y-%m-%d')
                if check_date in spy_prices:
                    valid_test_date = check_date
                    break

            if not valid_test_date:
                continue

            # Calculate horizon end date
            horizon_end = (datetime.strptime(valid_test_date, '%Y-%m-%d') +
                          timedelta(days=horizon_months * 30)).strftime('%Y-%m-%d')

            # Score all stocks as of test_date
            for ticker, prices in all_prices.items():
                if valid_test_date not in prices or horizon_end not in prices:
                    continue

                # Get fundamentals for this test date
                if use_historical_fundamentals and ticker in sec_covered_tickers:
                    # Use historical SEC data - point-in-time, no look-ahead bias
                    historical_price = prices.get(valid_test_date)
                    fundamentals = get_fundamentals_at_date(
                        ticker, valid_test_date, historical_price
                    )
                else:
                    # Use yfinance data (current fundamentals) or None for ETFs
                    fundamentals = all_fundamentals.get(ticker)

                result = _calculate_point_in_time_score(
                    ticker, prices, valid_test_date,
                    fundamentals, horizon_months,
                    use_historical=use_historical_fundamentals
                )

                if result is None:
                    continue

                score, grade, factor_scores = result

                # Calculate actual return
                start_price = prices[valid_test_date]
                end_price = prices[horizon_end]
                actual_return = ((end_price - start_price) / start_price) * 100

                # Calculate benchmark return
                spy_start = spy_prices.get(valid_test_date)
                spy_end = spy_prices.get(horizon_end)
                benchmark_return = ((spy_end - spy_start) / spy_start) * 100 if spy_start and spy_end else 0

                # Get expected return for this grade
                expected_return = EXPECTED_RETURNS[horizon_months].get(grade, 0)

                # Did it meet expectation?
                met_expectation = actual_return >= expected_return
                exceeded_benchmark = actual_return > benchmark_return

                validation_result = GradeValidationResult(
                    ticker=ticker,
                    grade=grade,
                    score=score,
                    grade_date=valid_test_date,
                    expected_return=expected_return,
                    actual_return=actual_return,
                    met_expectation=met_expectation,
                    exceeded_benchmark=exceeded_benchmark,
                    benchmark_return=benchmark_return,
                    growth_score=factor_scores["growth_score"],
                    quality_score=factor_scores["quality_score"],
                    financial_score=factor_scores["financial_score"],
                    valuation_score=factor_scores["valuation_score"],
                )

                all_grade_results[grade].append(validation_result)
                all_validation_results_for_factors.append(validation_result)

                if grade == "A":
                    a_grade_results.append(validation_result)
                elif grade == "B":
                    b_grade_results.append(validation_result)

        # Calculate summary statistics for this period
        def summarize_grade(results: List[GradeValidationResult]) -> Dict:
            if not results:
                return {
                    "count": 0,
                    "success_rate": 0,
                    "beat_benchmark_rate": 0,
                    "avg_return": 0,
                    "avg_expected": 0,
                }

            return {
                "count": len(results),
                "success_rate": round(sum(1 for r in results if r.met_expectation) / len(results) * 100, 1),
                "beat_benchmark_rate": round(sum(1 for r in results if r.exceeded_benchmark) / len(results) * 100, 1),
                "avg_return": round(np.mean([r.actual_return for r in results]), 2),
                "avg_expected": round(np.mean([r.expected_return for r in results]), 2),
                "best_picks": [
                    {"ticker": r.ticker, "return": round(r.actual_return, 1), "date": r.grade_date}
                    for r in sorted(results, key=lambda x: x.actual_return, reverse=True)[:3]
                ],
                "worst_picks": [
                    {"ticker": r.ticker, "return": round(r.actual_return, 1), "date": r.grade_date}
                    for r in sorted(results, key=lambda x: x.actual_return)[:3]
                ],
            }

        period_results[f"{years_ago}Y"] = {
            "period_start": start_date.strftime('%Y-%m-%d'),
            "period_end": end_date.strftime('%Y-%m-%d'),
            "test_points": len(test_points),
            "horizon_months": horizon_months,
            "grades": {
                grade: summarize_grade(results)
                for grade, results in all_grade_results.items()
            },
            "a_and_b_combined": summarize_grade(a_grade_results + b_grade_results),
        }

    if progress_callback:
        progress_callback(95, 100, "Generating summary...")

    # Compute factor correlations with actual returns
    factor_analysis = {}
    if len(all_validation_results_for_factors) > 10:
        all_validation_results = all_validation_results_for_factors
        returns = np.array([r.actual_return for r in all_validation_results])

        for factor_name in ["growth_score", "quality_score", "financial_score", "valuation_score"]:
            factor_values = np.array([getattr(r, factor_name) for r in all_validation_results])

            # Calculate correlation
            if np.std(factor_values) > 0 and np.std(returns) > 0:
                correlation = np.corrcoef(factor_values, returns)[0, 1]
            else:
                correlation = 0

            # Calculate average return for high vs low factor scores
            median_score = np.median(factor_values)
            high_factor = [r.actual_return for r, f in zip(all_validation_results, factor_values) if f >= median_score]
            low_factor = [r.actual_return for r, f in zip(all_validation_results, factor_values) if f < median_score]

            high_avg = np.mean(high_factor) if high_factor else 0
            low_avg = np.mean(low_factor) if low_factor else 0
            spread = high_avg - low_avg

            # Determine predictive power
            if abs(correlation) >= 0.15:
                power = "STRONG" if correlation > 0 else "INVERSE"
            elif abs(correlation) >= 0.08:
                power = "MODERATE" if correlation > 0 else "WEAK INVERSE"
            else:
                power = "WEAK"

            factor_analysis[factor_name.replace("_score", "")] = {
                "correlation": round(correlation, 3),
                "high_score_avg_return": round(high_avg, 2),
                "low_score_avg_return": round(low_avg, 2),
                "spread": round(spread, 2),
                "predictive_power": power,
            }

    # Overall summary

    # Determine if grading system is validated
    avg_a_success = np.mean([
        period_results[p]["grades"]["A"]["success_rate"]
        for p in period_results if period_results[p]["grades"]["A"]["count"] > 0
    ]) if any(period_results[p]["grades"]["A"]["count"] > 0 for p in period_results) else 0

    avg_b_success = np.mean([
        period_results[p]["grades"]["B"]["success_rate"]
        for p in period_results if period_results[p]["grades"]["B"]["count"] > 0
    ]) if any(period_results[p]["grades"]["B"]["count"] > 0 for p in period_results) else 0

    avg_ab_success = np.mean([
        period_results[p]["a_and_b_combined"]["success_rate"]
        for p in period_results if period_results[p]["a_and_b_combined"]["count"] > 0
    ]) if any(period_results[p]["a_and_b_combined"]["count"] > 0 for p in period_results) else 0

    if avg_ab_success >= 70:
        verdict = "VALIDATED"
        verdict_detail = f"A/B grade picks delivered expected returns {avg_ab_success:.0f}% of the time"
    elif avg_ab_success >= 55:
        verdict = "PARTIALLY VALIDATED"
        verdict_detail = f"A/B grade picks delivered expected returns {avg_ab_success:.0f}% of the time - moderate reliability"
    elif avg_ab_success >= 40:
        verdict = "WEAK"
        verdict_detail = f"A/B grade picks only delivered {avg_ab_success:.0f}% of the time - needs improvement"
    else:
        verdict = "NOT VALIDATED"
        verdict_detail = f"A/B grade picks only delivered {avg_ab_success:.0f}% of the time - grading system unreliable"

    # Generate factor insights for interpretation
    factor_insights = []
    if factor_analysis:
        best_factor = max(factor_analysis.items(), key=lambda x: x[1]["correlation"])
        worst_factor = min(factor_analysis.items(), key=lambda x: x[1]["correlation"])
        factor_insights = [
            f"Best predictor: {best_factor[0]} (r={best_factor[1]['correlation']}, spread={best_factor[1]['spread']}%)",
            f"Worst predictor: {worst_factor[0]} (r={worst_factor[1]['correlation']}, spread={worst_factor[1]['spread']}%)",
        ]

    # Note about fundamental data source
    fundamentals_source = (
        "SEC EDGAR (point-in-time, no look-ahead bias)"
        if use_historical_fundamentals
        else "yfinance (current fundamentals - may have look-ahead bias)"
    )
    sec_covered_count = len(sec_covered_tickers) if use_historical_fundamentals else 0

    result = {
        "test_type": "walk_forward_grade_validation",
        "horizon_months": horizon_months,
        "expected_returns": EXPECTED_RETURNS[horizon_months],
        "stocks_tested": len(all_prices),
        "sec_covered_stocks": sec_covered_count,
        "fundamentals_source": fundamentals_source,
        "use_historical_fundamentals": use_historical_fundamentals,
        "periods_tested": list(period_results.keys()),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "summary": {
            "avg_a_grade_success_rate": round(avg_a_success, 1),
            "avg_b_grade_success_rate": round(avg_b_success, 1),
            "avg_ab_combined_success_rate": round(avg_ab_success, 1),
        },
        "factor_analysis": factor_analysis,
        "period_results": period_results,
        "interpretation": [
            f"Tested {len(all_prices)} stocks across {len(period_results)} time periods",
            f"Fundamentals source: {fundamentals_source}",
            f"SEC-covered stocks: {sec_covered_count}/{len(all_prices)}",
            f"Horizon: {horizon_months} months",
            f"A-grade expected return: {EXPECTED_RETURNS[horizon_months]['A']}%",
            f"B-grade expected return: {EXPECTED_RETURNS[horizon_months]['B']}%",
            f"A-grade picks met expectations {avg_a_success:.0f}% of the time",
            f"B-grade picks met expectations {avg_b_success:.0f}% of the time",
            f"Overall A/B success rate: {avg_ab_success:.0f}%",
        ] + factor_insights,
    }

    if progress_callback:
        progress_callback(100, 100, "Validation complete")

    prediction_cache.set(cache_key, result, GRADE_VALIDATION_CACHE_TTL)
    return result
