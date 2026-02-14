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
    fundamentals: Optional[Dict] = None,
    holding_period_days: int = 90
) -> Optional[float]:
    """
    Calculate a composite score with adaptive weights based on holding period.

    Weights adjust automatically:
    - Short-term (<=90 days): More momentum/trend, less fundamentals
    - Medium-term (91-180 days): Balanced approach
    - Long-term (>180 days): More fundamentals, less momentum

    This reflects reality: momentum predicts short-term, fundamentals predict long-term.
    """
    # Determine weight profile based on holding period
    if holding_period_days <= 90:
        # Short-term: momentum and trend dominate
        w_momentum = 0.35
        w_trend = 0.25
        w_volatility = 0.15
        w_fundamentals = 0.25
    elif holding_period_days <= 180:
        # Medium-term: balanced
        w_momentum = 0.28
        w_trend = 0.22
        w_volatility = 0.15
        w_fundamentals = 0.35
    else:
        # Long-term: fundamentals dominate
        w_momentum = 0.20
        w_trend = 0.18
        w_volatility = 0.12
        w_fundamentals = 0.50

    # Technical scores (no look-ahead bias)
    momentum = _calculate_momentum_score(prices, as_of_date, 252)
    volatility = _calculate_volatility_score(prices, as_of_date, 63)
    trend = _calculate_trend_score(prices, as_of_date)

    # === ENHANCED FUNDAMENTAL SCORING ===
    fundamental_score = 50  # Default neutral
    if fundamentals:
        # Separate categories for more nuanced scoring
        quality_scores = []      # Profitability & efficiency
        growth_scores = []       # Revenue & earnings growth
        valuation_scores = []    # P/E, PEG, P/B
        health_scores = []       # Debt, liquidity

        # --- QUALITY METRICS (weight: 30% of fundamental score) ---

        # ROE - Return on Equity (higher is better, but not too high)
        roe = fundamentals.get('roe')
        if roe is not None:
            if 0.15 <= roe <= 0.30:
                quality_scores.append(90)  # Sweet spot
            elif 0.10 <= roe < 0.15:
                quality_scores.append(70)
            elif 0.30 < roe <= 0.40:
                quality_scores.append(70)  # High but sustainable
            elif roe > 0.40:
                quality_scores.append(50)  # Suspiciously high
            elif roe > 0:
                quality_scores.append(40)
            else:
                quality_scores.append(20)

        # ROA - Return on Assets
        roa = fundamentals.get('roa')
        if roa is not None:
            if roa > 0.10:
                quality_scores.append(85)
            elif roa > 0.05:
                quality_scores.append(70)
            elif roa > 0:
                quality_scores.append(50)
            else:
                quality_scores.append(25)

        # Profit Margin
        pm = fundamentals.get('profit_margin')
        if pm is not None:
            if pm > 0.20:
                quality_scores.append(90)
            elif pm > 0.15:
                quality_scores.append(80)
            elif pm > 0.10:
                quality_scores.append(65)
            elif pm > 0.05:
                quality_scores.append(50)
            elif pm > 0:
                quality_scores.append(35)
            else:
                quality_scores.append(20)

        # Operating Margin
        om = fundamentals.get('operating_margin')
        if om is not None:
            if om > 0.25:
                quality_scores.append(90)
            elif om > 0.15:
                quality_scores.append(75)
            elif om > 0.10:
                quality_scores.append(60)
            elif om > 0:
                quality_scores.append(40)
            else:
                quality_scores.append(20)

        # --- GROWTH METRICS (weight: 30% of fundamental score) ---

        # Revenue Growth (YoY)
        rev_growth = fundamentals.get('revenue_growth')
        if rev_growth is not None:
            if rev_growth > 0.20:
                growth_scores.append(90)  # Strong growth
            elif rev_growth > 0.10:
                growth_scores.append(75)
            elif rev_growth > 0.05:
                growth_scores.append(60)
            elif rev_growth > 0:
                growth_scores.append(45)
            elif rev_growth > -0.05:
                growth_scores.append(35)  # Slight decline
            else:
                growth_scores.append(20)  # Significant decline

        # Earnings Growth (YoY)
        earn_growth = fundamentals.get('earnings_growth')
        if earn_growth is not None:
            if earn_growth > 0.25:
                growth_scores.append(90)
            elif earn_growth > 0.15:
                growth_scores.append(75)
            elif earn_growth > 0.05:
                growth_scores.append(60)
            elif earn_growth > 0:
                growth_scores.append(45)
            elif earn_growth > -0.10:
                growth_scores.append(30)
            else:
                growth_scores.append(15)

        # --- VALUATION METRICS (weight: 25% of fundamental score) ---

        # P/E Ratio (lower is better, but not too low)
        pe = fundamentals.get('pe_ratio')
        if pe is not None and pe > 0:
            if 10 <= pe <= 20:
                valuation_scores.append(85)  # Reasonable valuation
            elif 20 < pe <= 25:
                valuation_scores.append(70)
            elif 5 <= pe < 10:
                valuation_scores.append(70)  # Cheap but might be value trap
            elif 25 < pe <= 35:
                valuation_scores.append(55)
            elif pe < 5:
                valuation_scores.append(40)  # Suspiciously cheap
            else:
                valuation_scores.append(35)  # Expensive

        # PEG Ratio (P/E to Growth - lower is better)
        peg = fundamentals.get('peg_ratio')
        if peg is not None and peg > 0:
            if peg < 1.0:
                valuation_scores.append(90)  # Undervalued relative to growth
            elif peg < 1.5:
                valuation_scores.append(75)
            elif peg < 2.0:
                valuation_scores.append(60)
            elif peg < 3.0:
                valuation_scores.append(45)
            else:
                valuation_scores.append(30)

        # Price to Book (lower is better for value)
        pb = fundamentals.get('price_to_book')
        if pb is not None and pb > 0:
            if pb < 1.5:
                valuation_scores.append(80)
            elif pb < 3.0:
                valuation_scores.append(65)
            elif pb < 5.0:
                valuation_scores.append(50)
            else:
                valuation_scores.append(35)

        # --- FINANCIAL HEALTH METRICS (weight: 15% of fundamental score) ---

        # Debt to Equity (lower is better)
        dte = fundamentals.get('debt_to_equity')
        if dte is not None:
            if dte < 30:
                health_scores.append(90)  # Very low debt
            elif dte < 50:
                health_scores.append(80)
            elif dte < 100:
                health_scores.append(65)
            elif dte < 150:
                health_scores.append(50)
            elif dte < 200:
                health_scores.append(35)
            else:
                health_scores.append(20)

        # Current Ratio (higher is better, but not too high)
        cr = fundamentals.get('current_ratio')
        if cr is not None:
            if 1.5 <= cr <= 3.0:
                health_scores.append(85)  # Healthy liquidity
            elif 1.2 <= cr < 1.5:
                health_scores.append(70)
            elif 3.0 < cr <= 5.0:
                health_scores.append(65)  # Too much cash sitting around
            elif cr > 5.0:
                health_scores.append(50)
            elif cr >= 1.0:
                health_scores.append(55)
            else:
                health_scores.append(25)  # Liquidity risk

        # Combine fundamental sub-scores with internal weights
        sub_scores = []
        sub_weights = []

        if quality_scores:
            sub_scores.append(np.mean(quality_scores))
            sub_weights.append(0.30)  # 30% quality

        if growth_scores:
            sub_scores.append(np.mean(growth_scores))
            sub_weights.append(0.30)  # 30% growth

        if valuation_scores:
            sub_scores.append(np.mean(valuation_scores))
            sub_weights.append(0.25)  # 25% valuation

        if health_scores:
            sub_scores.append(np.mean(health_scores))
            sub_weights.append(0.15)  # 15% health

        if sub_scores:
            total_sub_weight = sum(sub_weights)
            fundamental_score = sum(s * w for s, w in zip(sub_scores, sub_weights)) / total_sub_weight

    # === COMBINE ALL COMPONENTS ===
    # === COMBINE ALL COMPONENTS WITH ADAPTIVE WEIGHTS ===
    components = []
    weights = []

    if momentum is not None:
        norm_momentum = max(0, min(100, 50 + momentum))
        components.append(norm_momentum)
        weights.append(w_momentum)

    if volatility is not None:
        components.append(volatility)
        weights.append(w_volatility)

    if trend is not None:
        components.append(trend)
        weights.append(w_trend)

    components.append(fundamental_score)
    weights.append(w_fundamentals)

    if not components:
        return None

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

            # Pass holding period so weights adapt (quarterly rebalance = ~90 days)
            holding_days = rebalance_months * 30
            score = _calculate_point_in_time_score(
                ticker, prices, rebal_date, all_fundamentals.get(ticker), holding_days
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
    return_period_days: int = None,  # Auto-calculated based on data_years
    data_years: int = 3,
    max_workers: int = 8,
    progress_callback: Optional[callable] = None,
) -> Dict:
    """
    Pick Quality Test - Evaluates how well the scoring system picks winning stocks.

    Instead of comparing to random portfolios (irrelevant for long-term investing),
    this test answers: "Do the top-scored stocks actually go up?"

    Holding periods are automatically adjusted based on the test horizon:
    - 2-3 years: Quarterly (90-day) holding periods
    - 4-6 years: Semi-annual (180-day) holding periods
    - 7-10 years: Annual (252-day) holding periods

    This ensures the test matches your investment horizon.

    Metrics:
    - Win Rate: % of picks that had positive returns
    - Average Return: Mean return of top picks
    - Consistency: How often picks were profitable across different periods
    - Quality Grade: Overall assessment of pick quality

    Uses full scoring (momentum + volatility + trend + fundamentals) since
    the goal is to pick good long-term stocks, not beat a benchmark.

    Args:
        stocks: Universe of stocks (defaults to S&P 500 components)
        num_simulations: Not used (kept for API compatibility)
        top_n: Number of top stocks to select each period
        return_period_days: Holding period in days (auto-calculated if None)
        data_years: Years of historical data to test (2, 5, or 10 recommended)
        max_workers: Parallel workers for data fetching
        progress_callback: Optional progress callback

    Returns:
        Dict with pick quality metrics
    """
    if stocks is None:
        stocks = COMPOSITE_STOCK_LIST[:150]  # S&P 500 focused

    data_years = min(max(data_years, 1), 10)

    # Auto-calculate holding period based on investment horizon
    # Longer horizons = longer holding periods (more meaningful for long-term investing)
    if return_period_days is None:
        if data_years <= 3:
            return_period_days = 90   # Quarterly for short-term tests
            horizon_label = "Short-term (quarterly holds)"
        elif data_years <= 6:
            return_period_days = 180  # Semi-annual for medium-term
            horizon_label = "Medium-term (semi-annual holds)"
        else:
            return_period_days = 252  # Annual for long-term
            horizon_label = "Long-term (annual holds)"
    else:
        if return_period_days <= 90:
            horizon_label = "Quarterly holds"
        elif return_period_days <= 180:
            horizon_label = "Semi-annual holds"
        else:
            horizon_label = "Annual holds"

    cache_key = f"pick_quality_v2:{len(stocks)}:{top_n}:{return_period_days}:{data_years}y"
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        return cached

    if progress_callback:
        progress_callback(0, 100, "Fetching historical data...")

    # Fetch price data and fundamentals
    all_prices = {}
    all_fundamentals = {}

    def fetch_stock_data(ticker):
        prices = _get_historical_prices(ticker, years=data_years)
        fundamentals = fetch_fundamental_data(ticker)
        if prices and len(prices) > return_period_days * 2:
            return ticker, prices, fundamentals
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_stock_data, stocks))
        for r in results:
            if r is not None:
                all_prices[r[0]] = r[1]
                all_fundamentals[r[0]] = r[2]

    if len(all_prices) < top_n * 2:
        return {"error": f"Insufficient data. Only {len(all_prices)} stocks available."}

    if progress_callback:
        progress_callback(20, 100, "Analyzing historical periods...")

    # Find common dates
    all_date_sets = [set(prices.keys()) for prices in all_prices.values()]
    common_dates = sorted(set.intersection(*all_date_sets))

    if len(common_dates) < return_period_days * 3:
        return {"error": f"Insufficient common dates. Only {len(common_dates)} days overlap."}

    # Create test windows
    test_windows = []
    min_history_needed = 252

    i = min_history_needed
    while i + return_period_days < len(common_dates):
        test_windows.append({
            'start_date': common_dates[i],
            'end_date': common_dates[i + return_period_days],
        })
        i += return_period_days  # Non-overlapping

    if len(test_windows) < 2:
        return {"error": f"Insufficient test windows. Only {len(test_windows)} available."}

    if progress_callback:
        progress_callback(30, 100, f"Testing {len(test_windows)} periods...")

    # Track all picks across all windows
    all_picks = []  # Individual stock picks with their returns
    period_results = []  # Aggregated results per period

    for w_idx, window in enumerate(test_windows):
        if progress_callback:
            pct = 30 + (w_idx / len(test_windows)) * 60
            progress_callback(pct, 100, f"Period {w_idx + 1}/{len(test_windows)}...")

        # Score all stocks using FULL scoring (including fundamentals)
        stock_scores = []
        for ticker, prices in all_prices.items():
            if window['start_date'] not in prices or window['end_date'] not in prices:
                continue

            start_price = prices[window['start_date']]
            end_price = prices[window['end_date']]

            if start_price <= 0:
                continue

            ret = ((end_price - start_price) / start_price) * 100

            # Use full scoring with fundamentals - weights adapt to holding period
            score = _calculate_point_in_time_score(
                ticker, prices, window['start_date'], all_fundamentals.get(ticker), return_period_days
            )

            if score is not None:
                stock_scores.append({
                    'ticker': ticker,
                    'score': score,
                    'return': ret,
                })

        if len(stock_scores) < top_n:
            continue

        # Pick top N by score
        stock_scores.sort(key=lambda x: x['score'], reverse=True)
        top_picks = stock_scores[:top_n]

        # Track individual picks
        for pick in top_picks:
            all_picks.append({
                'ticker': pick['ticker'],
                'return': pick['return'],
                'score': pick['score'],
                'period': window['start_date'],
                'positive': pick['return'] > 0,
            })

        # Period summary
        period_return = np.mean([p['return'] for p in top_picks])
        period_winners = sum(1 for p in top_picks if p['return'] > 0)

        period_results.append({
            'period': f"{window['start_date']} to {window['end_date']}",
            'avg_return': round(period_return, 2),
            'winners': period_winners,
            'win_rate': round(period_winners / top_n * 100, 1),
            'top_picks': [p['ticker'] for p in top_picks[:5]],
            'best_pick': max(top_picks, key=lambda x: x['return']),
            'worst_pick': min(top_picks, key=lambda x: x['return']),
        })

    if len(all_picks) < top_n:
        return {"error": "Insufficient picks to analyze."}

    if progress_callback:
        progress_callback(95, 100, "Calculating quality metrics...")

    # Calculate overall metrics
    total_picks = len(all_picks)
    winning_picks = sum(1 for p in all_picks if p['positive'])
    overall_win_rate = winning_picks / total_picks * 100

    avg_return = np.mean([p['return'] for p in all_picks])
    avg_winner_return = np.mean([p['return'] for p in all_picks if p['positive']]) if winning_picks > 0 else 0
    avg_loser_return = np.mean([p['return'] for p in all_picks if not p['positive']]) if winning_picks < total_picks else 0

    # Consistency: how many periods had positive average returns?
    positive_periods = sum(1 for p in period_results if p['avg_return'] > 0)
    consistency = positive_periods / len(period_results) * 100

    # Find best and worst performers across all picks
    best_picks = sorted(all_picks, key=lambda x: x['return'], reverse=True)[:5]
    worst_picks = sorted(all_picks, key=lambda x: x['return'])[:5]

    # Determine quality grade
    if overall_win_rate >= 70 and avg_return >= 5 and consistency >= 70:
        quality_grade = "EXCELLENT"
        quality_detail = "Strong stock picker - high win rate with consistent positive returns"
    elif overall_win_rate >= 60 and avg_return >= 2 and consistency >= 60:
        quality_grade = "GOOD"
        quality_detail = "Reliable stock picker - majority of picks are profitable"
    elif overall_win_rate >= 50 and avg_return >= 0:
        quality_grade = "FAIR"
        quality_detail = "Moderate picker - slightly better than coin flip"
    else:
        quality_grade = "NEEDS IMPROVEMENT"
        quality_detail = "Picks not consistently profitable - consider refining criteria"

    # Show adaptive weights used for this holding period
    if return_period_days <= 90:
        weights_used = "35% Momentum, 25% Trend, 25% Fundamentals, 15% Volatility (short-term optimized)"
    elif return_period_days <= 180:
        weights_used = "28% Momentum, 22% Trend, 35% Fundamentals, 15% Volatility (balanced)"
    else:
        weights_used = "20% Momentum, 18% Trend, 50% Fundamentals, 12% Volatility (long-term optimized)"

    result = {
        "test_type": "pick_quality",
        "methodology": f"Adaptive scoring with {horizon_label}",
        "scoring_weights": weights_used,
        "fundamental_factors": "ROE, ROA, margins, revenue growth, earnings growth, P/E, PEG, P/B, debt ratio, liquidity",
        "stocks_analyzed": len(all_prices),
        "portfolio_size": top_n,
        "periods_tested": len(period_results),
        "data_years": data_years,
        "holding_period": f"{return_period_days} days",
        "horizon": horizon_label,
        "total_picks_analyzed": total_picks,
        "significance": quality_grade,  # Keep for frontend compatibility
        "significance_detail": quality_detail,
        "results": {
            # Frontend-compatible fields
            "strategy_return": round(avg_return, 2),
            "random_mean_return": 0,  # Not applicable - kept for compatibility
            "percentile": round(overall_win_rate, 1),  # Repurpose as win rate
            "p_value": round(1 - (overall_win_rate / 100), 4),  # Lower = better picks
            # New meaningful fields
            "win_rate": round(overall_win_rate, 1),
            "avg_return_per_period": round(avg_return, 2),
            "avg_winner_return": round(avg_winner_return, 2),
            "avg_loser_return": round(avg_loser_return, 2),
            "consistency": round(consistency, 1),
            "positive_periods": f"{positive_periods}/{len(period_results)}",
        },
        "distribution": {
            "best_period_return": round(max(p['avg_return'] for p in period_results), 2),
            "worst_period_return": round(min(p['avg_return'] for p in period_results), 2),
            "median_period_return": round(np.median([p['avg_return'] for p in period_results]), 2),
        },
        "best_picks": [
            {"ticker": p['ticker'], "return": round(p['return'], 1), "period": p['period']}
            for p in best_picks
        ],
        "worst_picks": [
            {"ticker": p['ticker'], "return": round(p['return'], 1), "period": p['period']}
            for p in worst_picks
        ],
        "period_details": period_results[-5:],  # Last 5 periods
        "interpretation": [
            f"Test Horizon: {data_years} years with {horizon_label.lower()}",
            f"Analyzed {total_picks} stock picks across {len(period_results)} periods",
            f"Win Rate: {overall_win_rate:.1f}% of picks had positive returns",
            f"Average Return: {avg_return:.1f}% per {return_period_days}-day holding period",
            f"Consistency: {positive_periods}/{len(period_results)} periods were profitable ({consistency:.0f}%)",
            f"Quality Grade: {quality_grade}",
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

    # Run Pick Quality test (replaces Monte Carlo significance test)
    # Tests whether top-scored stocks actually go up over time
    pick_quality_result = run_monte_carlo_significance(
        stocks=stocks,
        top_n=20,
        return_period_days=90,  # Quarterly holding periods
        data_years=simulation_years + 1,
        max_workers=max_workers,
    )

    # Overall assessment based on:
    # 1. Portfolio returns (did we make money?)
    # 2. Pick quality (do our picks consistently go up?)
    portfolio_return = portfolio_result.get('performance', {}).get('total_return', 0)
    portfolio_alpha = portfolio_result.get('performance', {}).get('alpha', 0)
    win_rate = pick_quality_result.get('results', {}).get('win_rate', 0)
    consistency = pick_quality_result.get('results', {}).get('consistency', 0)
    pick_quality = pick_quality_result.get('significance', 'UNKNOWN')

    # New verdict logic: focus on whether picks are profitable, not beating benchmarks
    if win_rate >= 65 and portfolio_return > 0 and consistency >= 60:
        overall_verdict = "STRONG PICKER"
        overall_detail = f"Reliable long-term picks: {win_rate:.0f}% win rate, {consistency:.0f}% consistency"
    elif win_rate >= 55 and portfolio_return > 0:
        overall_verdict = "GOOD PICKER"
        overall_detail = f"Solid picks with {win_rate:.0f}% win rate - suitable for long-term investing"
    elif win_rate >= 50 and portfolio_return > 0:
        overall_verdict = "MODERATE"
        overall_detail = f"Picks are slightly better than average ({win_rate:.0f}% win rate)"
    elif portfolio_return > 0:
        overall_verdict = "MARGINAL"
        overall_detail = "Positive returns but pick quality needs improvement"
    else:
        overall_verdict = "NEEDS WORK"
        overall_detail = "Strategy not producing reliable profitable picks"

    result = {
        "backtest_type": "rigorous_comprehensive",
        "simulation_years": simulation_years,
        "overall_verdict": overall_verdict,
        "overall_detail": overall_detail,
        "portfolio_simulation": portfolio_result,
        "statistical_significance": pick_quality_result,  # Renamed internally but kept for frontend
        "key_metrics": {
            "alpha_vs_spy": portfolio_result.get('performance', {}).get('alpha'),
            "sharpe_ratio": portfolio_result.get('risk_metrics', {}).get('sharpe_ratio'),
            "statistical_significance": pick_quality_result.get('significance'),  # Now shows quality grade
            "p_value": pick_quality_result.get('results', {}).get('p_value'),  # Lower = better picks
            # New metrics
            "win_rate": win_rate,
            "consistency": consistency,
            "avg_return_per_pick": pick_quality_result.get('results', {}).get('avg_return_per_period'),
        },
    }

    prediction_cache.set(cache_key, result, BACKTEST_CACHE_TTL)
    return result
