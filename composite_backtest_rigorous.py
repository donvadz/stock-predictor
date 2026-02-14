"""
Rigorous Composite Score Backtest

A more realistic backtest that simulates actual portfolio performance:
1. Walk-forward simulation with quarterly rebalancing
2. Portfolio tracking with real returns
3. Benchmark comparison (vs SPY)
4. Monte Carlo significance testing
5. Risk-adjusted metrics (Sharpe, Sortino, Max Drawdown)

Key improvement: Uses historical price data to calculate momentum-based
scores at each point in time, avoiding look-ahead bias for that component.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import random
from scipy import stats

from cache import prediction_cache, stock_data_cache
from data import fetch_stock_data, fetch_stock_data_extended, fetch_fundamental_data, SECTOR_ENCODING
from config import COMPOSITE_STOCK_LIST

logger = logging.getLogger(__name__)

BACKTEST_CACHE_TTL = 21600  # 6 hours


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio at a point in time."""
    date: str
    portfolio_value: float
    benchmark_value: float
    holdings: List[Dict]
    cash: float


def _get_historical_prices(ticker: str, years: int = 3) -> Optional[Dict[str, float]]:
    """
    Get historical closing prices indexed by date string.

    Args:
        ticker: Stock ticker symbol
        years: Number of years of history (1-10)

    Returns:
        Dict mapping date string to closing price
    """
    if years <= 3:
        df = fetch_stock_data(ticker, prediction_days=30)  # Gets 3 years
    else:
        df = fetch_stock_data_extended(ticker, years=years)

    if df is None or len(df) < 100:
        return None

    prices = {}
    for _, row in df.iterrows():
        date_str = row['timestamp'].strftime('%Y-%m-%d')
        prices[date_str] = float(row['close'])

    return prices


def _calculate_momentum_score(prices: Dict[str, float], as_of_date: str, lookback_days: int = 252) -> Optional[float]:
    """
    Calculate momentum score using only data available at as_of_date.
    This avoids look-ahead bias.
    """
    # Get all dates before as_of_date
    valid_dates = sorted([d for d in prices.keys() if d <= as_of_date])

    if len(valid_dates) < lookback_days:
        return None

    current_price = prices[valid_dates[-1]]
    past_price = prices[valid_dates[-lookback_days]]

    if past_price <= 0:
        return None

    return ((current_price - past_price) / past_price) * 100


def _calculate_volatility_score(prices: Dict[str, float], as_of_date: str, lookback_days: int = 63) -> Optional[float]:
    """
    Calculate volatility using only data available at as_of_date.
    Lower volatility = higher score (inverse).
    """
    valid_dates = sorted([d for d in prices.keys() if d <= as_of_date])

    if len(valid_dates) < lookback_days:
        return None

    recent_prices = [prices[d] for d in valid_dates[-lookback_days:]]
    returns = np.diff(recent_prices) / np.array(recent_prices[:-1])

    volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized

    # Convert to score (lower vol = higher score)
    # Vol of 20% = score 80, Vol of 40% = score 60, etc.
    score = max(0, min(100, 100 - volatility))

    return score


def _calculate_trend_score(prices: Dict[str, float], as_of_date: str) -> Optional[float]:
    """
    Calculate trend score based on moving averages.
    Price > SMA50 > SMA200 = bullish
    """
    valid_dates = sorted([d for d in prices.keys() if d <= as_of_date])

    if len(valid_dates) < 200:
        return None

    recent_prices = [prices[d] for d in valid_dates[-200:]]
    current_price = recent_prices[-1]
    sma50 = np.mean(recent_prices[-50:])
    sma200 = np.mean(recent_prices)

    score = 50  # Neutral

    if current_price > sma50:
        score += 20
    else:
        score -= 20

    if sma50 > sma200:
        score += 20
    else:
        score -= 20

    if current_price > sma200:
        score += 10
    else:
        score -= 10

    return max(0, min(100, score))


def _calculate_point_in_time_score(
    ticker: str,
    prices: Dict[str, float],
    as_of_date: str,
    fundamentals: Optional[Dict] = None
) -> Optional[float]:
    """
    Calculate a composite score using only data available at as_of_date.

    This is a simplified scoring that avoids look-ahead bias by using:
    - Historical momentum (calculable from past prices)
    - Historical volatility (calculable from past prices)
    - Trend strength (calculable from past prices)
    - Current fundamentals (with caveat - these may have changed)

    The fundamental component introduces some bias, but it's much less
    than using fundamentals to predict past returns.
    """
    # Technical scores (no look-ahead bias)
    momentum = _calculate_momentum_score(prices, as_of_date, 252)
    volatility = _calculate_volatility_score(prices, as_of_date, 63)
    trend = _calculate_trend_score(prices, as_of_date)

    # Fundamental score (some bias, but stable over time)
    fundamental_score = 50  # Default neutral
    if fundamentals:
        scores = []

        # ROE - quality indicator
        roe = fundamentals.get('roe')
        if roe is not None:
            if 0.15 <= roe <= 0.25:
                scores.append(80)
            elif 0.10 <= roe <= 0.30:
                scores.append(60)
            elif roe > 0:
                scores.append(40)
            else:
                scores.append(20)

        # Profit margin
        pm = fundamentals.get('profit_margin')
        if pm is not None:
            if pm > 0.20:
                scores.append(80)
            elif pm > 0.10:
                scores.append(60)
            elif pm > 0:
                scores.append(40)
            else:
                scores.append(20)

        # Debt to equity (lower is better)
        dte = fundamentals.get('debt_to_equity')
        if dte is not None:
            if dte < 50:
                scores.append(80)
            elif dte < 100:
                scores.append(60)
            elif dte < 200:
                scores.append(40)
            else:
                scores.append(20)

        if scores:
            fundamental_score = np.mean(scores)

    # Combine scores
    components = []
    weights = []

    if momentum is not None:
        # Normalize momentum to 0-100 scale
        norm_momentum = max(0, min(100, 50 + momentum))  # -50% to +50% -> 0 to 100
        components.append(norm_momentum)
        weights.append(0.30)  # 30% weight

    if volatility is not None:
        components.append(volatility)
        weights.append(0.20)  # 20% weight

    if trend is not None:
        components.append(trend)
        weights.append(0.25)  # 25% weight

    components.append(fundamental_score)
    weights.append(0.25)  # 25% weight

    if not components:
        return None

    # Weighted average
    total_weight = sum(weights)
    weighted_sum = sum(c * w for c, w in zip(components, weights))

    return weighted_sum / total_weight


def run_portfolio_simulation(
    stocks: Optional[List[str]] = None,
    initial_capital: float = 100000,
    top_n: int = 20,
    rebalance_months: int = 3,
    simulation_years: int = 2,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_workers: int = 8,
    progress_callback: Optional[callable] = None,
) -> Dict:
    """
    Run a walk-forward portfolio simulation.

    Args:
        stocks: Universe of stocks to choose from
        initial_capital: Starting capital
        top_n: Number of top-ranked stocks to hold
        rebalance_months: Months between rebalancing
        simulation_years: Number of years to simulate (1-10)
        start_date: Start date (YYYY-MM-DD), defaults based on simulation_years
        end_date: End date (YYYY-MM-DD), defaults to today
        max_workers: Parallel workers
        progress_callback: Optional progress callback

    Returns:
        Dict with simulation results
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST[:150]  # Use top 150 for speed

    simulation_years = min(max(simulation_years, 1), 10)  # Clamp to 1-10 years

    cache_key = f"portfolio_sim:{len(stocks)}:{top_n}:{rebalance_months}:{simulation_years}y"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Set date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=simulation_years * 365)).strftime('%Y-%m-%d')

    # Fetch all price data (need extra year for initial momentum calculation)
    data_years = min(simulation_years + 1, 10)

    if progress_callback:
        progress_callback(0, 100, f"Fetching {data_years} years of price data...")

    all_prices = {}
    all_fundamentals = {}

    def fetch_data(ticker):
        prices = _get_historical_prices(ticker, years=data_years)
        fundamentals = fetch_fundamental_data(ticker)
        return ticker, prices, fundamentals

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_data, t): t for t in stocks}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            if progress_callback:
                progress_callback(completed, len(stocks) * 2, f"Fetching data ({completed}/{len(stocks)})")

            try:
                ticker, prices, fundamentals = future.result()
                if prices and len(prices) > 200:
                    all_prices[ticker] = prices
                    all_fundamentals[ticker] = fundamentals
            except Exception as e:
                logger.warning(f"Failed to fetch {futures[future]}: {e}")

    if len(all_prices) < top_n:
        return {
            "error": f"Insufficient data. Only {len(all_prices)} stocks have enough history. This may be due to Yahoo Finance rate limiting. Please wait 5-10 minutes and try again.",
            "rate_limited": len(all_prices) == 0
        }

    # Get SPY for benchmark
    spy_prices = _get_historical_prices("SPY")
    if not spy_prices:
        return {"error": "Could not fetch SPY data for benchmark"}

    # Generate rebalancing dates
    rebalance_dates = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    while current < end:
        date_str = current.strftime('%Y-%m-%d')
        # Find nearest trading day
        for offset in range(7):
            check_date = (current + timedelta(days=offset)).strftime('%Y-%m-%d')
            if check_date in spy_prices:
                rebalance_dates.append(check_date)
                break
        current += timedelta(days=rebalance_months * 30)

    if len(rebalance_dates) < 2:
        return {"error": "Insufficient date range for simulation"}

    # Run simulation
    portfolio_value = initial_capital
    benchmark_value = initial_capital
    holdings = {}  # ticker -> shares
    cash = initial_capital

    snapshots = []
    all_trades = []
    period_returns = []
    benchmark_returns = []

    spy_start_price = spy_prices.get(rebalance_dates[0])

    for i, rebal_date in enumerate(rebalance_dates[:-1]):
        if progress_callback:
            progress_callback(
                len(stocks) + i,
                len(stocks) + len(rebalance_dates),
                f"Simulating period {i+1}/{len(rebalance_dates)-1}"
            )

        next_date = rebalance_dates[i + 1]

        # Score all stocks as of rebal_date
        scores = []
        for ticker, prices in all_prices.items():
            if rebal_date not in prices or next_date not in prices:
                continue

            score = _calculate_point_in_time_score(
                ticker, prices, rebal_date, all_fundamentals.get(ticker)
            )

            if score is not None:
                scores.append({
                    'ticker': ticker,
                    'score': score,
                    'price_at_rebal': prices[rebal_date],
                    'price_at_next': prices[next_date],
                })

        if len(scores) < top_n:
            continue

        # Rank and select top N
        scores.sort(key=lambda x: x['score'], reverse=True)
        selected = scores[:top_n]

        # Calculate portfolio value before rebalancing
        if holdings:
            portfolio_value = cash
            for ticker, shares in holdings.items():
                if ticker in all_prices and rebal_date in all_prices[ticker]:
                    portfolio_value += shares * all_prices[ticker][rebal_date]

        # Rebalance: equal weight across top N
        position_size = portfolio_value / top_n
        new_holdings = {}

        for stock in selected:
            ticker = stock['ticker']
            price = stock['price_at_rebal']
            shares = position_size / price
            new_holdings[ticker] = shares

            all_trades.append({
                'date': rebal_date,
                'ticker': ticker,
                'action': 'buy',
                'shares': shares,
                'price': price,
                'score': stock['score'],
            })

        holdings = new_holdings
        cash = 0

        # Calculate end-of-period value
        end_portfolio_value = 0
        for ticker, shares in holdings.items():
            if ticker in all_prices and next_date in all_prices[ticker]:
                end_portfolio_value += shares * all_prices[ticker][next_date]

        # Calculate benchmark value
        if spy_start_price and next_date in spy_prices:
            benchmark_value = initial_capital * (spy_prices[next_date] / spy_start_price)

        # Record returns
        period_return = ((end_portfolio_value - portfolio_value) / portfolio_value) * 100
        period_returns.append(period_return)

        if i > 0:
            prev_benchmark = initial_capital * (spy_prices[rebal_date] / spy_start_price)
            bench_return = ((benchmark_value - prev_benchmark) / prev_benchmark) * 100
            benchmark_returns.append(bench_return)

        snapshots.append(PortfolioSnapshot(
            date=next_date,
            portfolio_value=end_portfolio_value,
            benchmark_value=benchmark_value,
            holdings=[{'ticker': t, 'shares': s} for t, s in holdings.items()],
            cash=0,
        ))

        portfolio_value = end_portfolio_value

    if not snapshots:
        return {"error": "Simulation produced no results"}

    # Calculate final metrics
    final_portfolio = snapshots[-1].portfolio_value
    final_benchmark = snapshots[-1].benchmark_value

    total_return = ((final_portfolio - initial_capital) / initial_capital) * 100
    benchmark_return = ((final_benchmark - initial_capital) / initial_capital) * 100
    alpha = total_return - benchmark_return

    # Risk metrics
    if period_returns:
        avg_period_return = np.mean(period_returns)
        std_period_return = np.std(period_returns)

        # Annualize (assuming quarterly rebalancing)
        periods_per_year = 12 / rebalance_months
        annualized_return = avg_period_return * periods_per_year
        annualized_vol = std_period_return * np.sqrt(periods_per_year)

        sharpe = (annualized_return - 4) / annualized_vol if annualized_vol > 0 else 0

        # Win rate
        win_rate = sum(1 for r in period_returns if r > 0) / len(period_returns) * 100

        # Max drawdown
        peak = initial_capital
        max_dd = 0
        for snap in snapshots:
            if snap.portfolio_value > peak:
                peak = snap.portfolio_value
            dd = (peak - snap.portfolio_value) / peak * 100
            max_dd = max(max_dd, dd)
    else:
        avg_period_return = 0
        sharpe = 0
        win_rate = 0
        max_dd = 0
        annualized_return = 0
        annualized_vol = 0

    # Determine verdict
    if alpha > 10 and sharpe > 0.5:
        verdict = "STRONG ALPHA"
        verdict_detail = "Strategy significantly outperformed benchmark with good risk-adjusted returns"
    elif alpha > 5:
        verdict = "POSITIVE ALPHA"
        verdict_detail = "Strategy outperformed benchmark"
    elif alpha > 0:
        verdict = "MARGINAL ALPHA"
        verdict_detail = "Strategy slightly outperformed benchmark"
    elif alpha > -5:
        verdict = "NEUTRAL"
        verdict_detail = "Strategy performed similarly to benchmark"
    else:
        verdict = "UNDERPERFORMED"
        verdict_detail = "Strategy underperformed the benchmark"

    result = {
        "simulation_type": "walk_forward_portfolio",
        "simulation_years": simulation_years,
        "period": f"{start_date} to {end_date}",
        "stocks_in_universe": len(all_prices),
        "positions_held": top_n,
        "rebalance_frequency": f"Every {rebalance_months} months",
        "num_periods": len(period_returns),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "performance": {
            "initial_capital": initial_capital,
            "final_portfolio_value": round(final_portfolio, 2),
            "final_benchmark_value": round(final_benchmark, 2),
            "total_return": round(total_return, 2),
            "benchmark_return": round(benchmark_return, 2),
            "alpha": round(alpha, 2),
            "annualized_return": round(annualized_return, 2),
            "annualized_volatility": round(annualized_vol, 2),
        },
        "risk_metrics": {
            "sharpe_ratio": round(sharpe, 2),
            "win_rate": round(win_rate, 1),
            "max_drawdown": round(max_dd, 2),
            "avg_period_return": round(avg_period_return, 2),
        },
        "equity_curve": [
            {"date": s.date, "portfolio": round(s.portfolio_value, 2), "benchmark": round(s.benchmark_value, 2)}
            for s in snapshots
        ],
        "recent_holdings": [
            {"ticker": h['ticker'], "shares": round(h['shares'], 2)}
            for h in snapshots[-1].holdings[:10]
        ] if snapshots else [],
        "methodology": {
            "scoring": "Point-in-time momentum + volatility + trend + fundamentals",
            "rebalancing": f"Equal-weight top {top_n} stocks every {rebalance_months} months",
            "benchmark": "SPY (S&P 500 ETF)",
            "bias_mitigation": "Uses historical prices only; fundamentals have some forward bias",
        },
    }

    prediction_cache.set(cache_key, result, BACKTEST_CACHE_TTL)
    return result


def _calculate_technical_only_score(
    ticker: str,
    prices: Dict[str, float],
    as_of_date: str
) -> Optional[float]:
    """
    Calculate score using ONLY price-based metrics (no fundamentals).

    This avoids forward-looking bias entirely since all inputs
    are derived from historical prices available at as_of_date.
    """
    momentum = _calculate_momentum_score(prices, as_of_date, 252)
    volatility = _calculate_volatility_score(prices, as_of_date, 63)
    trend = _calculate_trend_score(prices, as_of_date)

    components = []
    weights = []

    if momentum is not None:
        norm_momentum = max(0, min(100, 50 + momentum))
        components.append(norm_momentum)
        weights.append(0.40)  # Increased weight (was 30% of 75%)

    if volatility is not None:
        components.append(volatility)
        weights.append(0.27)  # Increased weight (was 20% of 75%)

    if trend is not None:
        components.append(trend)
        weights.append(0.33)  # Increased weight (was 25% of 75%)

    if not components:
        return None

    total_weight = sum(weights)
    weighted_sum = sum(c * w for c, w in zip(components, weights))

    return weighted_sum / total_weight


def run_monte_carlo_significance(
    stocks: Optional[List[str]] = None,
    num_simulations: int = 500,
    top_n: int = 20,
    return_period_days: int = 90,
    data_years: int = 3,
    max_workers: int = 8,
    progress_callback: Optional[callable] = None,
) -> Dict:
    """
    Monte Carlo test for statistical significance using multiple historical periods.

    This test avoids forward-looking bias by:
    1. Testing across multiple historical time windows (not just the most recent)
    2. Using only price-based technical scores (no fundamentals that could leak future info)
    3. Comparing strategy vs random selection at each historical point

    Compares the actual strategy returns to randomly selected portfolios.
    If the strategy outperforms >95% of random portfolios, it's significant.

    Args:
        stocks: Universe of stocks
        num_simulations: Number of random portfolios per time window
        top_n: Number of stocks in each portfolio
        return_period_days: Days to measure returns (also used as window spacing)
        data_years: Years of historical data to test across (1-10)
        max_workers: Parallel workers
        progress_callback: Progress callback

    Returns:
        Dict with significance test results
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST[:150]  # Match portfolio simulation universe

    data_years = min(max(data_years, 1), 10)  # Clamp to 1-10 years

    cache_key = f"monte_carlo_v2:{len(stocks)}:{num_simulations}:{top_n}:{return_period_days}:{data_years}y"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    if progress_callback:
        progress_callback(0, 100, "Fetching historical data...")

    # Fetch all price data upfront
    all_prices = {}

    def fetch_prices(ticker):
        prices = _get_historical_prices(ticker, years=data_years)
        if prices and len(prices) > return_period_days * 2:
            return ticker, prices
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_prices, stocks))
        for r in results:
            if r is not None:
                all_prices[r[0]] = r[1]

    if len(all_prices) < top_n * 2:
        return {"error": f"Insufficient data. Only {len(all_prices)} stocks available."}

    if progress_callback:
        progress_callback(20, 100, "Identifying test windows...")

    # Find common date range across all stocks
    all_date_sets = [set(prices.keys()) for prices in all_prices.values()]
    common_dates = sorted(set.intersection(*all_date_sets))

    if len(common_dates) < return_period_days * 3:
        return {"error": f"Insufficient common dates. Only {len(common_dates)} days overlap."}

    # Create test windows: every return_period_days, going back from most recent
    # Each window: score stocks at window_start, measure returns to window_end
    test_windows = []
    window_spacing = return_period_days  # Non-overlapping windows

    # Start from the earliest possible window and move forward
    # This ensures we have enough history for scoring at each window start
    min_history_needed = 252  # Need 1 year of history for momentum calculation

    i = min_history_needed
    while i + return_period_days < len(common_dates):
        window_start_idx = i
        window_end_idx = i + return_period_days
        test_windows.append({
            'start_date': common_dates[window_start_idx],
            'end_date': common_dates[window_end_idx],
            'start_idx': window_start_idx,
            'end_idx': window_end_idx,
        })
        i += window_spacing

    if len(test_windows) < 4:
        return {"error": f"Insufficient test windows. Only {len(test_windows)} windows available."}

    if progress_callback:
        progress_callback(30, 100, f"Testing {len(test_windows)} historical windows...")

    # Run the test across all windows
    all_strategy_returns = []
    all_random_returns = []
    window_results = []

    for w_idx, window in enumerate(test_windows):
        if progress_callback:
            pct = 30 + (w_idx / len(test_windows)) * 60
            progress_callback(pct, 100, f"Testing window {w_idx + 1}/{len(test_windows)}...")

        # Score all stocks as of window start using ONLY technical indicators
        # This is truly point-in-time: we only use price data available at start_date
        stock_scores = []
        for ticker, prices in all_prices.items():
            dates = sorted(prices.keys())

            # Ensure we have the exact dates for this window
            if window['start_date'] not in prices or window['end_date'] not in prices:
                continue

            start_price = prices[window['start_date']]
            end_price = prices[window['end_date']]

            if start_price <= 0:
                continue

            # Calculate return over the window
            ret = ((end_price - start_price) / start_price) * 100

            # Calculate score using ONLY data available at window start (no fundamentals!)
            score = _calculate_technical_only_score(ticker, prices, window['start_date'])

            if score is not None:
                stock_scores.append({
                    'ticker': ticker,
                    'score': score,
                    'return': ret,
                })

        if len(stock_scores) < top_n * 2:
            continue  # Skip this window if insufficient data

        # Strategy: pick top N by score
        stock_scores.sort(key=lambda x: x['score'], reverse=True)
        strategy_stocks = stock_scores[:top_n]
        strategy_return = np.mean([s['return'] for s in strategy_stocks])
        all_strategy_returns.append(strategy_return)

        # Random: sample multiple random portfolios for this window
        window_random_returns = []
        sims_per_window = max(50, num_simulations // len(test_windows))
        for _ in range(sims_per_window):
            random_selection = random.sample(stock_scores, top_n)
            random_return = np.mean([s['return'] for s in random_selection])
            window_random_returns.append(random_return)
            all_random_returns.append(random_return)

        window_results.append({
            'period': f"{window['start_date']} to {window['end_date']}",
            'strategy_return': round(strategy_return, 2),
            'random_mean': round(np.mean(window_random_returns), 2),
            'beat_random': strategy_return > np.mean(window_random_returns),
            'top_picks': [s['ticker'] for s in strategy_stocks[:3]],
        })

    if len(all_strategy_returns) < 4:
        return {"error": f"Insufficient valid windows. Only {len(all_strategy_returns)} completed."}

    if progress_callback:
        progress_callback(95, 100, "Calculating statistics...")

    # Aggregate results across all windows
    avg_strategy_return = np.mean(all_strategy_returns)
    avg_random_return = np.mean(all_random_returns)

    # Calculate how often strategy beat random
    windows_beat = sum(1 for w in window_results if w['beat_random'])
    win_rate = windows_beat / len(window_results) * 100

    # Calculate percentile: what % of random portfolios did strategy beat?
    percentile = sum(1 for r in all_random_returns if avg_strategy_return > r) / len(all_random_returns) * 100

    # Calculate p-value using t-test for paired comparison
    # Compare each window's strategy return to that window's random mean
    strategy_vs_random_diffs = [
        w['strategy_return'] - w['random_mean'] for w in window_results
    ]

    # One-sample t-test: is the mean difference > 0?
    t_stat, p_value_ttest = stats.ttest_1samp(strategy_vs_random_diffs, 0)
    p_value = p_value_ttest / 2 if t_stat > 0 else 1 - p_value_ttest / 2  # One-tailed

    # Determine significance
    if p_value < 0.01:
        significance = "HIGHLY SIGNIFICANT"
        significance_detail = "Strategy outperforms random selection with >99% confidence"
    elif p_value < 0.05:
        significance = "SIGNIFICANT"
        significance_detail = "Strategy outperforms random selection with >95% confidence"
    elif p_value < 0.10:
        significance = "MARGINALLY SIGNIFICANT"
        significance_detail = "Weak evidence of strategy outperformance"
    else:
        significance = "NOT SIGNIFICANT"
        significance_detail = "Cannot distinguish from random stock selection"

    result = {
        "test_type": "monte_carlo_significance_v2",
        "methodology": "Multi-period point-in-time test (no forward-looking bias)",
        "stocks_analyzed": len(all_prices),
        "portfolio_size": top_n,
        "test_windows": len(window_results),
        "return_period": f"{return_period_days} days per window",
        "total_simulations": len(all_random_returns),
        "significance": significance,
        "significance_detail": significance_detail,
        "results": {
            "strategy_avg_return": round(avg_strategy_return, 2),
            "random_avg_return": round(avg_random_return, 2),
            "strategy_std": round(np.std(all_strategy_returns), 2),
            "windows_beat_random": f"{windows_beat}/{len(window_results)}",
            "win_rate": round(win_rate, 1),
            "percentile": round(percentile, 1),
            "p_value": round(p_value, 4),
            "t_statistic": round(t_stat, 3),
        },
        "distribution": {
            "strategy_min": round(min(all_strategy_returns), 2),
            "strategy_max": round(max(all_strategy_returns), 2),
            "random_5th_percentile": round(np.percentile(all_random_returns, 5), 2),
            "random_median": round(np.median(all_random_returns), 2),
            "random_95th_percentile": round(np.percentile(all_random_returns, 95), 2),
        },
        "window_details": window_results[-5:],  # Show last 5 windows
        "interpretation": [
            f"Tested {len(window_results)} historical {return_period_days}-day periods",
            f"Strategy avg return: {avg_strategy_return:.1f}% vs random avg: {avg_random_return:.1f}%",
            f"Strategy beat random in {windows_beat}/{len(window_results)} windows ({win_rate:.0f}%)",
            f"P-value: {p_value:.3f} (t-test, lower = more significant, <0.05 is threshold)",
            "Note: Uses only technical indicators (no fundamental forward-bias)",
        ],
    }

    prediction_cache.set(cache_key, result, BACKTEST_CACHE_TTL)
    return result


def run_rigorous_backtest(
    stocks: Optional[List[str]] = None,
    simulation_years: int = 2,
    max_workers: int = 8,
    progress_callback: Optional[callable] = None,
) -> Dict:
    """
    Run comprehensive rigorous backtest combining all methods.

    Args:
        stocks: Universe of stocks
        simulation_years: Years to simulate (1-10)
        max_workers: Parallel workers
        progress_callback: Progress callback
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST[:150]

    simulation_years = min(max(simulation_years, 1), 10)  # Clamp to 1-10 years

    cache_key = f"rigorous_backtest:{len(stocks)}:{simulation_years}y"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    # Run portfolio simulation
    portfolio_result = run_portfolio_simulation(
        stocks=stocks,
        top_n=20,
        rebalance_months=3,
        simulation_years=simulation_years,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )

    # Run Monte Carlo test (use simulation_years for data fetch)
    monte_carlo_result = run_monte_carlo_significance(
        stocks=stocks,
        num_simulations=500,
        top_n=20,
        return_period_days=90 * simulation_years,  # Scale return period with years
        data_years=simulation_years + 1,
        max_workers=max_workers,
    )

    # Overall assessment
    portfolio_alpha = portfolio_result.get('performance', {}).get('alpha', 0)
    p_value = monte_carlo_result.get('results', {}).get('p_value', 1)

    if portfolio_alpha > 5 and p_value < 0.05:
        overall_verdict = "VALIDATED"
        overall_detail = "Strategy shows significant alpha with statistical backing"
    elif portfolio_alpha > 0 and p_value < 0.10:
        overall_verdict = "PROMISING"
        overall_detail = "Strategy shows positive alpha with some statistical support"
    elif portfolio_alpha > 0:
        overall_verdict = "INCONCLUSIVE"
        overall_detail = "Positive returns but not statistically significant"
    else:
        overall_verdict = "NOT VALIDATED"
        overall_detail = "Strategy does not show reliable outperformance"

    result = {
        "backtest_type": "rigorous_comprehensive",
        "simulation_years": simulation_years,
        "overall_verdict": overall_verdict,
        "overall_detail": overall_detail,
        "portfolio_simulation": portfolio_result,
        "statistical_significance": monte_carlo_result,
        "key_metrics": {
            "alpha_vs_spy": portfolio_result.get('performance', {}).get('alpha'),
            "sharpe_ratio": portfolio_result.get('risk_metrics', {}).get('sharpe_ratio'),
            "statistical_significance": monte_carlo_result.get('significance'),
            "p_value": monte_carlo_result.get('results', {}).get('p_value'),
        },
    }

    prediction_cache.set(cache_key, result, BACKTEST_CACHE_TTL)
    return result
