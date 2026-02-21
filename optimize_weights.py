#!/usr/bin/env python3
"""
Weight Optimization for Horizon-Aware Scoring

This script:
1. Runs validation tests across all horizons and historical periods
2. Collects factor correlations with actual returns
3. Uses grid search to find optimal weights for each horizon
4. Outputs optimized weights and expected improvement

Usage:
    python optimize_weights.py
    python optimize_weights.py --quick     # Quick test with fewer stocks
    python optimize_weights.py --yfinance  # Use yfinance (faster, has look-ahead bias)
"""

import argparse
import time
import itertools
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from cache import prediction_cache
from data import fetch_stock_data_extended, fetch_fundamental_data
from historical_fundamentals import get_fundamentals_at_date, is_sec_covered
from scoring import (
    calculate_composite_score, get_grade, GRADE_THRESHOLDS,
    score_growth, score_quality, score_financial_strength, score_valuation,
    score_sentiment_contrarian
)

# S&P 500 stocks (comprehensive list)
SP500_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO",
    "ACN", "ABT", "BAC", "CRM", "DHR", "PFE", "CMCSA", "VZ", "ADBE", "NKE",
    "TXN", "NFLX", "WFC", "PM", "ORCL", "AMD", "INTC", "UPS", "RTX", "NEE",
    "HON", "COP", "BMY", "QCOM", "LOW", "UNP", "MS", "SPGI", "INTU", "IBM",
    "ELV", "BA", "AMGN", "GE", "CAT", "DE", "SBUX", "GS", "ISRG", "PLD",
    "GILD", "MDT", "AXP", "BLK", "LMT", "SYK", "BKNG", "ADI", "MDLZ", "CVS",
    "TMUS", "CB", "REGN", "AMT", "SCHW", "TJX", "LRCX", "ADP", "MMC", "MO",
    "CI", "EOG", "SLB", "ZTS", "VRTX", "SO", "DUK", "CME", "SNPS", "CDNS",
    "BDX", "CL", "ITW", "BSX", "EQIX", "FI", "WM", "ETN", "APD", "AON",
    "NOC", "SHW", "ICE", "KLAC", "PNC", "HCA", "CSX", "EMR", "PGR", "MPC",
    "ORLY", "GD", "MCO", "FCX", "MCK", "USB", "AZO", "MAR", "CTAS", "MCHP",
    "NSC", "ROP", "AJG", "MSI", "SRE", "AFL", "PSX", "TT", "APH", "AEP",
    "CMG", "PCAR", "F", "GM", "TRV", "KMB", "WELL", "ADSK", "PAYX", "PSA",
    "DXCM", "PANW", "MNST", "CPRT", "PYPL", "MELI", "FTNT", "IDXX", "ODFL",
    "FAST", "CSGP", "AEE", "KHC", "VRSK", "DLTR", "CDW", "GIS", "BKR", "EW",
    "CTSH", "BIIB", "KDP", "ON", "TEAM", "ALGN", "WBD", "ENPH", "ANSS", "ZS",
    "DDOG", "CRWD", "CEG", "TTD", "FANG", "GEHC",
    "MSCI", "MPWR", "KEYS", "TDG", "FICO", "TRGP", "VLTO", "DECK", "AXON",
    "EXC", "D", "PEG", "ED", "XEL", "ES", "AWK", "WEC", "DTE", "AES",
    "HES", "DVN", "OXY", "HAL", "MRO", "VLO",
    "A", "ILMN", "IQV", "TECH", "WAT", "MTD", "PKI", "HOLX", "DGX", "LH",
    "STZ", "TAP", "AIG", "ALL", "MET", "PRU", "HIG", "CINF", "GL", "AFG", "WRB",
    "FIS", "GPN", "JKHY", "PAYC", "PCTY", "WEX", "BR",
    "CARR", "JCI", "LII", "GNRC", "HUBB", "EME", "PWR", "FSLR",
    "POOL", "WSO", "SNA", "SWK", "GWW", "NDSN",
    "DOV", "IR", "PH", "ROK", "AME", "ZBRA", "GRMN", "TER", "ENTG",
    "J", "TTEK", "FTV", "BAH", "LDOS", "KBR", "CACI", "BWXT",
    "LKQ", "APTV", "BWA", "MGA", "LEA", "GNTX", "VC",
    "ROL", "CHRW", "XPO", "EXPD", "JBHT", "SAIA",
    "RCL", "CCL", "NCLH", "HLT", "H", "WH",
    "EXPE", "ABNB", "UBER", "LYFT", "DASH",
    "DRI", "YUM", "QSR", "DPZ",
    "LIN", "ECL", "PPG", "NUE", "STLD", "RS", "CLF", "AA",
    "ALB", "MP",
    "CCI", "SPG", "O", "DLR", "AVB",
    "EQR", "VTR", "ARE", "BXP", "KIM", "REG", "FRT", "NNN",
    "WPC", "STAG", "REXR", "FR",
    "C", "TFC", "COF",
    "BK", "STT", "NTRS", "DFS", "SYF", "ALLY",
    "NDAQ", "CBOE", "MKTX", "VIRT", "IBKR", "RJF",
    "SF", "EVR", "LAZ", "HLI", "PJT", "JEF",
    "BX", "KKR", "APO", "CG", "ARES",
    "BEN", "TROW", "IVZ",
    "MRNA", "INCY", "EXEL",
    "PODD", "TFX",
    "STE", "BAX",
    "UHS", "THC", "ACHC", "ENSG",
    "WBA", "ABC", "CAH", "HSIC",
    "DIS", "T", "CHTR",
    "EA", "TTWO",
    "MTCH", "PINS", "SNAP", "ZM", "SPOT",
    "PARA", "LYV",
    "NXST", "NWS", "NYT",
    "ROST", "DG", "TGT", "BBY", "ULTA", "LULU", "GPS", "ANF",
    "W", "CVNA",
    "RH", "WSM",
    "AN", "PAG", "LAD", "GPI",
    "BC", "PII", "HOG", "THO", "WGO",
    "HAS", "MAT", "ELY",
    "MGM", "WYNN", "LVS", "CZR", "PENN", "DKNG",
    "EL", "HSY", "HRL", "CPB", "SJM", "MKC",
    "TSN", "CAG", "POST",
    "CLX", "CHD", "NWL",
    "KR", "ACI", "SFM", "CASY",
    "FIZZ", "CELH",
    "PXD", "MTDR",
    "OVV", "EQT", "AR", "SWN", "CTRA",
    "WMB", "KMI", "OKE", "LNG", "ET", "EPD", "MPLX", "PAA",
    "DK", "PBF",
    "ETR", "FE", "PPL", "CMS", "CNP",
    "EVRG", "NI", "PNW", "LNT",
    "WTRG",
]

# Deduplicate
SP500_STOCKS = list(dict.fromkeys(SP500_STOCKS))

# All horizons to test
HORIZONS = [3, 6, 12, 24, 60, 120]

# Expected returns by grade and horizon
EXPECTED_RETURNS = {
    3: {"A": 4, "B": 2, "C": 0, "D": -1, "F": -3},
    6: {"A": 7, "B": 4, "C": 1, "D": -1, "F": -4},
    12: {"A": 15, "B": 10, "C": 5, "D": 0, "F": -5},
    24: {"A": 30, "B": 20, "C": 10, "D": 3, "F": -8},
    60: {"A": 75, "B": 50, "C": 25, "D": 10, "F": -10},
    120: {"A": 150, "B": 100, "C": 50, "D": 20, "F": -15},
}


@dataclass
class ValidationSample:
    """A single validation sample with all factor scores and actual return."""
    ticker: str
    test_date: str
    horizon_months: int
    actual_return: float
    benchmark_return: float
    # Individual factor scores
    growth_score: float
    quality_score: float
    financial_score: float
    valuation_score: float
    sentiment_score: float
    # Raw fundamentals for re-scoring with different weights
    fundamentals: Dict
    price_momentum: float


def _get_historical_prices(ticker: str, years: int = 12) -> Optional[Dict[str, float]]:
    """Get historical closing prices indexed by date string."""
    df = fetch_stock_data_extended(ticker, years=years)

    if df is None or len(df) < 100:
        return None

    prices = {}
    for _, row in df.iterrows():
        date_str = row['timestamp'].strftime('%Y-%m-%d')
        prices[date_str] = float(row['close'])

    return prices


def collect_validation_samples(
    stocks: List[str],
    horizons: List[int],
    test_periods: List[int],  # Years ago
    max_workers: int = 8,
    use_historical_fundamentals: bool = True,
    progress_callback: Optional[callable] = None,
) -> Dict[int, List[ValidationSample]]:
    """
    Collect validation samples for all horizons.

    Returns:
        Dict mapping horizon_months to list of ValidationSample
    """
    print(f"Collecting validation samples for {len(stocks)} stocks...")
    print(f"Horizons: {horizons}")
    print(f"Test periods: {test_periods} years ago")

    # Need max_years + max_horizon data
    max_years = max(test_periods) + (max(horizons) // 12) + 2

    # Fetch all price data and fundamentals
    all_prices = {}
    all_fundamentals = {}
    sec_covered = set()

    def fetch_data(ticker):
        prices = _get_historical_prices(ticker, years=max_years)
        if use_historical_fundamentals:
            is_covered = is_sec_covered(ticker)
            fundamentals = None
        else:
            is_covered = False
            fundamentals = fetch_fundamental_data(ticker)
        if prices and len(prices) > 252:
            return ticker, prices, fundamentals, is_covered
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_data, t): t for t in stocks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                all_prices[result[0]] = result[1]
                all_fundamentals[result[0]] = result[2]
                if result[3]:
                    sec_covered.add(result[0])
            if completed % 50 == 0:
                print(f"  Fetched {completed}/{len(stocks)} stocks...")

    print(f"  Loaded {len(all_prices)} stocks with sufficient history")
    print(f"  SEC EDGAR covered: {len(sec_covered)}")

    # Get SPY benchmark
    spy_prices = _get_historical_prices("SPY", years=max_years)
    if not spy_prices:
        raise ValueError("Could not fetch SPY benchmark data")

    # Collect samples for each horizon
    samples_by_horizon = defaultdict(list)

    for horizon_months in horizons:
        print(f"\nCollecting samples for {horizon_months}M horizon...")

        for years_ago in test_periods:
            # Calculate date range
            end_date = datetime.now() - timedelta(days=horizon_months * 30)
            start_date = end_date - timedelta(days=years_ago * 365)

            # Quarterly test points
            current = start_date
            test_dates = []
            while current < end_date:
                test_dates.append(current.strftime('%Y-%m-%d'))
                current += timedelta(days=90)

            for test_date in test_dates:
                # Find nearest valid date
                valid_date = None
                for offset in range(7):
                    check = (datetime.strptime(test_date, '%Y-%m-%d') +
                             timedelta(days=offset)).strftime('%Y-%m-%d')
                    if check in spy_prices:
                        valid_date = check
                        break

                if not valid_date:
                    continue

                # Horizon end date
                horizon_end = (datetime.strptime(valid_date, '%Y-%m-%d') +
                              timedelta(days=horizon_months * 30)).strftime('%Y-%m-%d')

                for ticker, prices in all_prices.items():
                    if valid_date not in prices or horizon_end not in prices:
                        continue

                    # Get fundamentals
                    if use_historical_fundamentals and ticker in sec_covered:
                        hist_price = prices.get(valid_date)
                        fundamentals = get_fundamentals_at_date(ticker, valid_date, hist_price)
                    else:
                        fundamentals = all_fundamentals.get(ticker)

                    if not fundamentals:
                        continue

                    # Calculate price momentum
                    valid_dates = sorted([d for d in prices.keys() if d <= valid_date])
                    if len(valid_dates) < 252:
                        continue

                    current_price = prices[valid_dates[-1]]
                    year_ago_price = prices[valid_dates[-252]]
                    price_momentum = (current_price - year_ago_price) / year_ago_price

                    # Calculate individual factor scores
                    growth_score, _ = score_growth(
                        fundamentals.get("revenue_growth"),
                        fundamentals.get("earnings_growth"),
                        price_momentum
                    )

                    quality_score, _ = score_quality(
                        fundamentals.get("roe"),
                        fundamentals.get("roa"),
                        fundamentals.get("profit_margin"),
                        fundamentals.get("operating_margin")
                    )

                    financial_score, _ = score_financial_strength(
                        fundamentals.get("debt_to_equity"),
                        fundamentals.get("current_ratio")
                    )

                    valuation_score, _ = score_valuation(
                        fundamentals.get("pe_ratio"),
                        fundamentals.get("price_to_book"),
                        fundamentals.get("peg_ratio"),
                        fundamentals.get("sector")
                    )

                    # Sentiment only for short-term
                    sentiment_score = None
                    if horizon_months <= 12:
                        sentiment_score, _ = score_sentiment_contrarian(
                            fundamentals.get("short_percent"),
                            fundamentals.get("analyst_sentiment"),
                            fundamentals.get("earnings_surprise"),
                            quality_score
                        )

                    # Calculate returns
                    start_price = prices[valid_date]
                    end_price = prices[horizon_end]
                    actual_return = ((end_price - start_price) / start_price) * 100

                    spy_start = spy_prices.get(valid_date)
                    spy_end = spy_prices.get(horizon_end)
                    benchmark_return = ((spy_end - spy_start) / spy_start) * 100 if spy_start and spy_end else 0

                    sample = ValidationSample(
                        ticker=ticker,
                        test_date=valid_date,
                        horizon_months=horizon_months,
                        actual_return=actual_return,
                        benchmark_return=benchmark_return,
                        growth_score=growth_score or 50,
                        quality_score=quality_score or 50,
                        financial_score=financial_score or 50,
                        valuation_score=valuation_score or 50,
                        sentiment_score=sentiment_score or 50,
                        fundamentals=fundamentals,
                        price_momentum=price_momentum,
                    )

                    samples_by_horizon[horizon_months].append(sample)

        print(f"  Collected {len(samples_by_horizon[horizon_months])} samples for {horizon_months}M")

    return dict(samples_by_horizon)


def calculate_grade_with_weights(
    sample: ValidationSample,
    weights: Dict[str, float],
) -> Tuple[float, str]:
    """Calculate composite score and grade using given weights."""
    scores = {
        "growth": sample.growth_score,
        "quality": sample.quality_score,
        "financial": sample.financial_score,
        "valuation": sample.valuation_score,
        "sentiment": sample.sentiment_score if sample.horizon_months <= 12 else None,
    }

    total_weight = 0
    weighted_sum = 0

    for factor, weight in weights.items():
        if scores.get(factor) is not None and weight > 0:
            weighted_sum += scores[factor] * weight
            total_weight += weight

    if total_weight == 0:
        return 50, "C"

    composite = weighted_sum / total_weight
    grade = get_grade(composite)

    return composite, grade


def evaluate_weights(
    samples: List[ValidationSample],
    weights: Dict[str, float],
    horizon_months: int,
) -> Dict:
    """
    Evaluate a set of weights on validation samples.

    Returns metrics including success rate, factor correlations, etc.
    """
    expected_returns = EXPECTED_RETURNS.get(horizon_months, EXPECTED_RETURNS[12])

    results_by_grade = defaultdict(list)
    all_scores = []
    all_returns = []

    for sample in samples:
        score, grade = calculate_grade_with_weights(sample, weights)
        expected = expected_returns.get(grade, 0)
        met_expectation = sample.actual_return >= expected

        results_by_grade[grade].append({
            "actual_return": sample.actual_return,
            "expected_return": expected,
            "met_expectation": met_expectation,
            "beat_benchmark": sample.actual_return > sample.benchmark_return,
        })

        all_scores.append(score)
        all_returns.append(sample.actual_return)

    # Calculate success rates
    def grade_metrics(results):
        if not results:
            return {"count": 0, "success_rate": 0, "avg_return": 0}
        return {
            "count": len(results),
            "success_rate": sum(r["met_expectation"] for r in results) / len(results) * 100,
            "avg_return": np.mean([r["actual_return"] for r in results]),
            "beat_benchmark_rate": sum(r["beat_benchmark"] for r in results) / len(results) * 100,
        }

    a_metrics = grade_metrics(results_by_grade["A"])
    b_metrics = grade_metrics(results_by_grade["B"])
    ab_combined = grade_metrics(results_by_grade["A"] + results_by_grade["B"])

    # Score-return correlation
    if len(all_scores) > 10:
        correlation = np.corrcoef(all_scores, all_returns)[0, 1]
    else:
        correlation = 0

    return {
        "a_grade": a_metrics,
        "b_grade": b_metrics,
        "ab_combined": ab_combined,
        "score_return_correlation": correlation,
        "total_samples": len(samples),
    }


def optimize_weights_for_horizon(
    samples: List[ValidationSample],
    horizon_months: int,
    include_sentiment: bool = True,
) -> Dict:
    """
    Find optimal weights for a given horizon using grid search.

    Returns the best weights and their performance.
    """
    print(f"\nOptimizing weights for {horizon_months}M horizon...")
    print(f"  Samples: {len(samples)}")

    # Define weight search space
    # Weights must sum to 1.0, each factor 0-0.6
    weight_options = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    factors = ["valuation", "quality", "financial", "growth"]
    if include_sentiment and horizon_months <= 12:
        factors.append("sentiment")

    best_weights = None
    best_score = -float('inf')
    best_metrics = None

    # Generate all valid weight combinations (sum to ~1.0)
    tested = 0
    valid_combos = []

    for combo in itertools.product(weight_options, repeat=len(factors)):
        total = sum(combo)
        # Allow weights that sum close to 1.0 (0.95 - 1.05)
        if 0.95 <= total <= 1.05:
            # Normalize to exactly 1.0
            weights = {f: c/total for f, c in zip(factors, combo)}
            valid_combos.append(weights)

    print(f"  Testing {len(valid_combos)} weight combinations...")

    for weights in valid_combos:
        tested += 1
        if tested % 500 == 0:
            print(f"    Tested {tested}/{len(valid_combos)}...")

        metrics = evaluate_weights(samples, weights, horizon_months)

        # Scoring: prioritize A/B combined success rate, with correlation bonus
        ab_success = metrics["ab_combined"]["success_rate"]
        correlation = metrics["score_return_correlation"]

        # Composite optimization score
        opt_score = ab_success + (correlation * 10)  # Bonus for positive correlation

        if opt_score > best_score:
            best_score = opt_score
            best_weights = weights
            best_metrics = metrics

    print(f"  Best weights found:")
    for f, w in sorted(best_weights.items(), key=lambda x: -x[1]):
        print(f"    {f}: {w*100:.0f}%")
    print(f"  A/B Success Rate: {best_metrics['ab_combined']['success_rate']:.1f}%")
    print(f"  Score-Return Correlation: {best_metrics['score_return_correlation']:.3f}")

    return {
        "horizon_months": horizon_months,
        "best_weights": best_weights,
        "metrics": best_metrics,
        "optimization_score": best_score,
    }


def analyze_factor_correlations(
    samples: List[ValidationSample],
    horizon_months: int,
) -> Dict:
    """Analyze correlation of each factor with returns."""
    returns = np.array([s.actual_return for s in samples])

    factors = {
        "valuation": np.array([s.valuation_score for s in samples]),
        "quality": np.array([s.quality_score for s in samples]),
        "financial": np.array([s.financial_score for s in samples]),
        "growth": np.array([s.growth_score for s in samples]),
    }

    if horizon_months <= 12:
        factors["sentiment"] = np.array([s.sentiment_score for s in samples])

    correlations = {}
    for name, values in factors.items():
        if np.std(values) > 0:
            corr = np.corrcoef(values, returns)[0, 1]
        else:
            corr = 0

        # High vs low analysis
        median = np.median(values)
        high_returns = [r for v, r in zip(values, returns) if v >= median]
        low_returns = [r for v, r in zip(values, returns) if v < median]

        correlations[name] = {
            "correlation": round(corr, 4),
            "high_avg_return": round(np.mean(high_returns), 2) if high_returns else 0,
            "low_avg_return": round(np.mean(low_returns), 2) if low_returns else 0,
            "spread": round(np.mean(high_returns) - np.mean(low_returns), 2) if high_returns and low_returns else 0,
        }

    return correlations


def get_current_weights(horizon_months: int) -> Dict[str, float]:
    """Get current weights from scoring.py for comparison."""
    from scoring import get_weights
    return get_weights(horizon_months)


def run_optimization(
    stocks: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    test_periods: Optional[List[int]] = None,
    max_workers: int = 8,
    use_historical_fundamentals: bool = True,
) -> Dict:
    """
    Run the full optimization pipeline.
    """
    if stocks is None:
        stocks = SP500_STOCKS
    if horizons is None:
        horizons = HORIZONS
    if test_periods is None:
        test_periods = [10, 5, 2, 1]

    print("=" * 80)
    print("WEIGHT OPTIMIZATION FOR HORIZON-AWARE SCORING")
    print("=" * 80)
    print(f"Stocks: {len(stocks)}")
    print(f"Horizons: {horizons}")
    print(f"Test periods: {test_periods} years ago")
    print(f"Fundamentals: {'SEC EDGAR' if use_historical_fundamentals else 'yfinance'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Collect all validation samples
    samples_by_horizon = collect_validation_samples(
        stocks=stocks,
        horizons=horizons,
        test_periods=test_periods,
        max_workers=max_workers,
        use_historical_fundamentals=use_historical_fundamentals,
    )

    # Analyze and optimize each horizon
    results = {}

    for horizon in horizons:
        samples = samples_by_horizon.get(horizon, [])
        if len(samples) < 50:
            print(f"\nSkipping {horizon}M - insufficient samples ({len(samples)})")
            continue

        # Analyze factor correlations
        correlations = analyze_factor_correlations(samples, horizon)

        print(f"\n{'='*60}")
        print(f"HORIZON: {horizon} MONTHS")
        print(f"{'='*60}")
        print(f"Samples: {len(samples)}")
        print(f"\nFactor Correlations with Returns:")
        for factor, data in sorted(correlations.items(), key=lambda x: -x[1]["correlation"]):
            print(f"  {factor:12s}: r={data['correlation']:+.4f}, spread={data['spread']:+.1f}%")

        # Get current weights
        current_weights = get_current_weights(horizon)
        current_metrics = evaluate_weights(samples, current_weights, horizon)

        print(f"\nCurrent Weights Performance:")
        print(f"  A/B Success Rate: {current_metrics['ab_combined']['success_rate']:.1f}%")
        print(f"  Score-Return Correlation: {current_metrics['score_return_correlation']:.3f}")

        # Optimize
        include_sentiment = horizon <= 12
        optimization = optimize_weights_for_horizon(samples, horizon, include_sentiment)

        # Calculate improvement
        improvement = (optimization["metrics"]["ab_combined"]["success_rate"] -
                      current_metrics["ab_combined"]["success_rate"])

        print(f"\n  IMPROVEMENT: {improvement:+.1f}% success rate")

        results[horizon] = {
            "samples": len(samples),
            "factor_correlations": correlations,
            "current_weights": current_weights,
            "current_metrics": current_metrics,
            "optimized_weights": optimization["best_weights"],
            "optimized_metrics": optimization["metrics"],
            "improvement": improvement,
        }

    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    print(f"\n{'Horizon':<10} {'Current':<12} {'Optimized':<12} {'Change':<10}")
    print("-" * 50)

    for horizon in horizons:
        if horizon not in results:
            continue
        r = results[horizon]
        current = r["current_metrics"]["ab_combined"]["success_rate"]
        optimized = r["optimized_metrics"]["ab_combined"]["success_rate"]
        change = optimized - current
        print(f"{horizon}M{'':<7} {current:<12.1f} {optimized:<12.1f} {change:+.1f}%")

    print("\n" + "-" * 80)
    print("OPTIMIZED WEIGHTS BY HORIZON:")
    print("-" * 80)

    for horizon in horizons:
        if horizon not in results:
            continue
        weights = results[horizon]["optimized_weights"]
        print(f"\n{horizon}M:")
        for f, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {f}: {w*100:.0f}%")

    # Generate code for scoring.py
    print("\n" + "=" * 80)
    print("SUGGESTED CODE UPDATE FOR scoring.py:")
    print("=" * 80)
    print("""
def get_weights(horizon_months: int) -> Dict[str, float]:
    \"\"\"
    Get factor weights based on investment horizon.
    OPTIMIZED WEIGHTS (generated by optimize_weights.py)
    \"\"\"
""")

    prev_weights = None
    for horizon in sorted(horizons):
        if horizon not in results:
            continue
        weights = results[horizon]["optimized_weights"]

        # Format weights
        weights_str = "{\n"
        for f in ["valuation", "sentiment", "quality", "financial", "growth"]:
            if f in weights:
                weights_str += f'            "{f}": {weights[f]:.2f},\n'
        weights_str += "        }"

        if horizon == horizons[0]:
            print(f"    if horizon_months <= {horizon}:")
        elif horizon == horizons[-1]:
            print(f"    else:")
        else:
            print(f"    elif horizon_months <= {horizon}:")

        print(f"        return {weights_str}")

    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize scoring weights by horizon")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer stocks (100)",
    )
    parser.add_argument(
        "--yfinance",
        action="store_true",
        help="Use yfinance (faster, has look-ahead bias)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    stocks = SP500_STOCKS[:100] if args.quick else SP500_STOCKS
    use_historical = not args.yfinance

    results = run_optimization(
        stocks=stocks,
        horizons=HORIZONS,
        test_periods=[10, 5, 2, 1],
        max_workers=args.workers,
        use_historical_fundamentals=use_historical,
    )
