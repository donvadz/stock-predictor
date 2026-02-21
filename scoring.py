"""
Unified Scoring Module for Stock Ranking

This module contains the SINGLE source of truth for scoring logic.
Both the live system (composite_score.py) and backtest (composite_backtest_grades.py)
import from here to ensure identical behavior.

Methodology: OPTIMIZED HORIZON-AWARE SCORING (Feb 2026)

Optimization Process:
- Tested 443 S&P 500 stocks with SEC EDGAR point-in-time fundamentals
- 85K+ validation samples across 10Y, 5Y, 2Y, 1Y historical periods
- Grid search over 20K+ weight combinations per horizon
- No look-ahead bias (used actual fundamentals from filing dates)

Key Findings from SEC EDGAR Backtesting:
1. Valuation: POSITIVE correlation (+0.04 to +0.08) - value investing works
2. Quality: NEGATIVE correlation (-0.04 to -0.07) - already priced in
3. Sentiment: Effective for short-term (contrarian, squeeze plays)
4. Financial: Increasingly important for longer horizons
5. Growth: Slightly negative - momentum doesn't persist reliably

OPTIMIZED Weight Profiles by Horizon:
- 1-3M:  val=21%, sentiment=53%, quality=0%, financial=11%, growth=16%
- 4-6M:  val=26%, sentiment=53%, quality=0%, financial=21%, growth=0%
- 7-12M: val=30%, sentiment=50%, quality=0%, financial=20%, growth=0%
- 24M:   val=43%, sentiment=0%,  quality=0%, financial=48%, growth=10%
- 60M+:  val=42%, sentiment=0%,  quality=0%, financial=47%, growth=11%

Validation Results (A/B Combined Success Rate):
- 3M:  57.4% -> 61.2% (+3.7%)
- 6M:  62.8% -> 66.1% (+3.3%)
- 12M: 52.7% -> 57.4% (+4.7%)
- 24M: 56.2% -> 59.0% (+2.7%)
- 60M: 64.4% -> 69.3% (+4.9%)

Grade Philosophy:
- Grade reflects expected ALPHA, not just quality
- A: Undervalued + strong financials = expected to beat market
- C: Fairly priced = market returns
- F: Overvalued or weak financials = avoid

Grade Thresholds (consistent everywhere):
- A: >= 75 (Excellent - undervalued, strong financials, strong buy)
- B: >= 60 (Good - slight discount to fair value)
- C: >= 45 (Average - fairly priced, market returns)
- D: >= 30 (Below average - overpriced or weak)
- F: < 30 (Poor - avoid)
"""

from typing import Dict, Optional, Tuple


# =============================================================================
# GRADE THRESHOLDS - Single source of truth
# =============================================================================

GRADE_THRESHOLDS = [
    (75, "A"),  # Excellent
    (60, "B"),  # Good
    (45, "C"),  # Average
    (30, "D"),  # Below average
    (0, "F"),   # Poor
]


# =============================================================================
# SECTOR-SPECIFIC P/E THRESHOLDS
# =============================================================================
# Each sector has different valuation norms. Tech companies trade at higher P/E
# ratios than utilities or financials. Using relative thresholds prevents
# penalizing growth sectors for their naturally higher valuations.
#
# Sector codes from data.py SECTOR_ENCODING:
# 1=Technology, 2=Healthcare, 3=Financial, 4=Consumer Cyclical,
# 5=Communication, 6=Industrials, 7=Consumer Defensive, 8=Energy,
# 9=Utilities, 10=Real Estate, 11=Basic Materials, 0=Unknown/ETF

SECTOR_PE_THRESHOLDS = {
    1: {"low": 18, "median": 28, "high": 45},   # Technology - higher growth justifies higher P/E
    2: {"low": 15, "median": 22, "high": 35},   # Healthcare - variable due to R&D
    3: {"low": 8, "median": 12, "high": 18},    # Financial Services - low P/E sector
    4: {"low": 12, "median": 18, "high": 28},   # Consumer Cyclical - moderate
    5: {"low": 15, "median": 22, "high": 35},   # Communication Services - tech-adjacent
    6: {"low": 12, "median": 18, "high": 28},   # Industrials - moderate
    7: {"low": 15, "median": 20, "high": 28},   # Consumer Defensive - stable, low growth
    8: {"low": 8, "median": 12, "high": 20},    # Energy - cyclical, low P/E
    9: {"low": 14, "median": 18, "high": 24},   # Utilities - stable, regulated
    10: {"low": 20, "median": 35, "high": 50},  # Real Estate - uses different metrics (FFO)
    11: {"low": 8, "median": 14, "high": 22},   # Basic Materials - cyclical
    0: {"low": 12, "median": 18, "high": 30},   # Unknown/ETF - use market average
}


def get_grade(score: float) -> str:
    """Convert numeric score (0-100) to letter grade."""
    if score is None:
        return "F"
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


# =============================================================================
# FACTOR WEIGHTS BY HORIZON - Single source of truth
# =============================================================================

def get_weights(horizon_months: int) -> Dict[str, float]:
    """
    Get factor weights based on investment horizon.

    OPTIMIZED WEIGHTS (Feb 2026) - Based on SEC EDGAR backtesting:
    - Tested on 443 S&P 500 stocks with 85K+ validation samples
    - Uses point-in-time fundamentals (no look-ahead bias)
    - Tested across 10Y, 5Y, 2Y, 1Y historical periods

    Key findings from optimization:
    - Valuation: POSITIVE correlation (+0.04 to +0.08) - value investing works!
    - Quality: NEGATIVE correlation (-0.04 to -0.07) - already priced in, weight=0
    - Sentiment: Works for short-term (mean reversion, squeeze plays)
    - Financial: Increasingly important for longer horizons (+0.01 to +0.03)
    - Growth: Slightly negative correlation - momentum doesn't persist

    Improvement over previous weights:
    - 3M:  57.4% -> 61.2% (+3.7%)
    - 6M:  62.8% -> 66.1% (+3.3%)
    - 12M: 52.7% -> 57.4% (+4.7%)
    - 24M: 56.2% -> 59.0% (+2.7%)
    - 60M: 64.4% -> 69.3% (+4.9%)
    """
    if horizon_months <= 3:
        # Short-term: sentiment dominates (contrarian/squeeze plays)
        # Valuation and growth have some predictive value
        return {
            "valuation": 0.21,
            "sentiment": 0.53,
            "quality": 0.00,   # Negative correlation - remove
            "financial": 0.11,
            "growth": 0.16,    # Some short-term momentum value
        }
    elif horizon_months <= 6:
        # Medium-term: sentiment + valuation + financial
        # Growth loses predictive value
        return {
            "valuation": 0.26,
            "sentiment": 0.53,
            "quality": 0.00,
            "financial": 0.21,
            "growth": 0.00,
        }
    elif horizon_months <= 12:
        # 1 year: sentiment still valuable, fundamentals increase
        return {
            "valuation": 0.30,
            "sentiment": 0.50,
            "quality": 0.00,
            "financial": 0.20,
            "growth": 0.00,
        }
    elif horizon_months <= 24:
        # 2 years: pure fundamentals, no sentiment
        # Financial strength becomes dominant
        return {
            "valuation": 0.43,
            "sentiment": 0.00,
            "quality": 0.00,
            "financial": 0.48,
            "growth": 0.10,    # Slight growth factor
        }
    else:
        # 5+ years: financial strength + valuation (69.3% success rate)
        return {
            "valuation": 0.42,
            "sentiment": 0.00,
            "quality": 0.00,
            "financial": 0.47,
            "growth": 0.11,
        }


# =============================================================================
# ABSOLUTE SCORING FUNCTIONS
# =============================================================================

def score_growth(
    revenue_growth: Optional[float],
    earnings_growth: Optional[float],
    price_momentum: Optional[float],
) -> Tuple[Optional[float], Dict]:
    """
    Calculate Growth score using absolute thresholds.

    Weights: revenue (40%), earnings (40%), momentum (20%)

    Args:
        revenue_growth: YoY revenue growth as decimal (0.15 = 15%)
        earnings_growth: YoY earnings growth as decimal
        price_momentum: Price return as decimal (0.20 = 20%)

    Returns:
        Tuple of (score 0-100, metrics dict)
    """
    metrics = {}
    components = []
    weights = []

    # Revenue Growth (40%)
    # Thresholds: >30% = excellent, 15-30% = good, 5-15% = average, <5% = poor
    if revenue_growth is not None:
        if revenue_growth >= 0.30:
            rev_score = 90 + min(10, (revenue_growth - 0.30) * 50)
        elif revenue_growth >= 0.15:
            rev_score = 70 + (revenue_growth - 0.15) / 0.15 * 20
        elif revenue_growth >= 0.05:
            rev_score = 50 + (revenue_growth - 0.05) / 0.10 * 20
        elif revenue_growth >= 0:
            rev_score = 30 + revenue_growth / 0.05 * 20
        else:
            rev_score = max(0, 30 + revenue_growth * 100)  # Negative growth

        metrics["revenue_growth"] = {"value": revenue_growth, "score": rev_score}
        components.append(rev_score)
        weights.append(0.4)

    # Earnings Growth (40%)
    # Similar thresholds but allow for more volatility
    if earnings_growth is not None:
        if earnings_growth >= 0.50:
            earn_score = 90 + min(10, (earnings_growth - 0.50) * 20)
        elif earnings_growth >= 0.20:
            earn_score = 70 + (earnings_growth - 0.20) / 0.30 * 20
        elif earnings_growth >= 0.05:
            earn_score = 50 + (earnings_growth - 0.05) / 0.15 * 20
        elif earnings_growth >= 0:
            earn_score = 30 + earnings_growth / 0.05 * 20
        else:
            earn_score = max(0, 30 + earnings_growth * 50)

        metrics["earnings_growth"] = {"value": earnings_growth, "score": earn_score}
        components.append(earn_score)
        weights.append(0.4)

    # Price Momentum (20%)
    # Positive momentum is good, but extreme momentum may indicate overvaluation
    if price_momentum is not None:
        if 0.10 <= price_momentum <= 0.40:
            mom_score = 70 + (price_momentum - 0.10) / 0.30 * 30  # Sweet spot
        elif price_momentum > 0.40:
            mom_score = max(50, 100 - (price_momentum - 0.40) * 50)  # Too hot
        elif price_momentum >= 0:
            mom_score = 50 + price_momentum / 0.10 * 20
        else:
            mom_score = max(0, 50 + price_momentum * 100)  # Negative momentum

        metrics["price_momentum"] = {"value": price_momentum, "score": mom_score}
        components.append(mom_score)
        weights.append(0.2)

    if not components:
        return None, metrics

    score = sum(c * w for c, w in zip(components, weights)) / sum(weights)
    return score, metrics


def score_quality(
    roe: Optional[float],
    roa: Optional[float],
    profit_margin: Optional[float],
    operating_margin: Optional[float],
) -> Tuple[Optional[float], Dict]:
    """
    Calculate Quality score using absolute thresholds.

    Weights: ROE (30%), ROA (20%), profit margin (25%), operating margin (25%)

    Args:
        roe: Return on equity as decimal (0.20 = 20%)
        roa: Return on assets as decimal
        profit_margin: Net profit margin as decimal
        operating_margin: Operating margin as decimal

    Returns:
        Tuple of (score 0-100, metrics dict)
    """
    metrics = {}
    components = []
    weights = []

    # ROE (30%) - Ideal range 15-25%, penalize extremes
    if roe is not None:
        if roe < 0:
            roe_score = 0
        elif roe <= 0.10:
            roe_score = roe / 0.10 * 50  # 0-50 for 0-10%
        elif roe <= 0.15:
            roe_score = 50 + (roe - 0.10) / 0.05 * 20  # 50-70 for 10-15%
        elif roe <= 0.25:
            roe_score = 70 + (roe - 0.15) / 0.10 * 30  # 70-100 for 15-25%
        elif roe <= 0.40:
            roe_score = 100 - (roe - 0.25) / 0.15 * 20  # 100-80 for 25-40%
        else:
            roe_score = max(60, 80 - (roe - 0.40) * 50)  # Penalize >40%

        metrics["roe"] = {"value": roe, "score": roe_score}
        components.append(roe_score)
        weights.append(0.3)

    # ROA (20%) - Higher is better, 10%+ is excellent
    if roa is not None:
        if roa < 0:
            roa_score = 0
        elif roa <= 0.05:
            roa_score = roa / 0.05 * 50  # 0-50 for 0-5%
        elif roa <= 0.10:
            roa_score = 50 + (roa - 0.05) / 0.05 * 30  # 50-80 for 5-10%
        elif roa <= 0.20:
            roa_score = 80 + (roa - 0.10) / 0.10 * 20  # 80-100 for 10-20%
        else:
            roa_score = 100

        metrics["roa"] = {"value": roa, "score": roa_score}
        components.append(roa_score)
        weights.append(0.2)

    # Profit Margin (25%) - Higher is better
    if profit_margin is not None:
        if profit_margin < 0:
            pm_score = 0
        elif profit_margin <= 0.05:
            pm_score = profit_margin / 0.05 * 40
        elif profit_margin <= 0.10:
            pm_score = 40 + (profit_margin - 0.05) / 0.05 * 20
        elif profit_margin <= 0.20:
            pm_score = 60 + (profit_margin - 0.10) / 0.10 * 25
        elif profit_margin <= 0.30:
            pm_score = 85 + (profit_margin - 0.20) / 0.10 * 15
        else:
            pm_score = 100

        metrics["profit_margin"] = {"value": profit_margin, "score": pm_score}
        components.append(pm_score)
        weights.append(0.25)

    # Operating Margin (25%) - Higher is better
    if operating_margin is not None:
        if operating_margin < 0:
            om_score = 0
        elif operating_margin <= 0.10:
            om_score = operating_margin / 0.10 * 50
        elif operating_margin <= 0.20:
            om_score = 50 + (operating_margin - 0.10) / 0.10 * 25
        elif operating_margin <= 0.30:
            om_score = 75 + (operating_margin - 0.20) / 0.10 * 15
        else:
            om_score = min(100, 90 + (operating_margin - 0.30) * 50)

        metrics["operating_margin"] = {"value": operating_margin, "score": om_score}
        components.append(om_score)
        weights.append(0.25)

    if not components:
        return None, metrics

    score = sum(c * w for c, w in zip(components, weights)) / sum(weights)
    return score, metrics


def score_financial_strength(
    debt_to_equity: Optional[float],
    current_ratio: Optional[float],
) -> Tuple[Optional[float], Dict]:
    """
    Calculate Financial Strength score using absolute thresholds.

    Weights: debt-to-equity (60%), current ratio (40%)

    Args:
        debt_to_equity: D/E ratio as percentage (50 = 50% or 0.5x)
        current_ratio: Current assets / current liabilities

    Returns:
        Tuple of (score 0-100, metrics dict)
    """
    metrics = {}
    components = []
    weights = []

    # Debt-to-Equity (60%) - Lower is better
    # Note: D/E comes as percentage (50 = 50%), not decimal
    if debt_to_equity is not None and debt_to_equity >= 0:
        if debt_to_equity <= 30:
            dte_score = 90 + (30 - debt_to_equity) / 30 * 10  # 90-100 for 0-30%
        elif debt_to_equity <= 50:
            dte_score = 75 + (50 - debt_to_equity) / 20 * 15  # 75-90 for 30-50%
        elif debt_to_equity <= 100:
            dte_score = 50 + (100 - debt_to_equity) / 50 * 25  # 50-75 for 50-100%
        elif debt_to_equity <= 200:
            dte_score = 25 + (200 - debt_to_equity) / 100 * 25  # 25-50 for 100-200%
        else:
            dte_score = max(0, 25 - (debt_to_equity - 200) / 100 * 25)  # 0-25 for >200%

        metrics["debt_to_equity"] = {"value": debt_to_equity, "score": dte_score}
        components.append(dte_score)
        weights.append(0.6)

    # Current Ratio (40%) - 1.5-3.0 is ideal
    if current_ratio is not None and current_ratio > 0:
        if current_ratio < 1.0:
            cr_score = current_ratio * 40  # 0-40 for 0-1
        elif current_ratio < 1.5:
            cr_score = 40 + (current_ratio - 1.0) / 0.5 * 30  # 40-70 for 1-1.5
        elif current_ratio <= 3.0:
            cr_score = 70 + (current_ratio - 1.5) / 1.5 * 30  # 70-100 for 1.5-3
        else:
            cr_score = max(60, 100 - (current_ratio - 3.0) * 10)  # Slight penalty >3

        metrics["current_ratio"] = {"value": current_ratio, "score": cr_score}
        components.append(cr_score)
        weights.append(0.4)

    if not components:
        return None, metrics

    score = sum(c * w for c, w in zip(components, weights)) / sum(weights)
    return score, metrics


def score_valuation(
    pe_ratio: Optional[float],
    price_to_book: Optional[float],
    peg_ratio: Optional[float] = None,
    sector: Optional[int] = None,
) -> Tuple[Optional[float], Dict]:
    """
    Calculate Valuation score using sector-relative thresholds.

    Weights: P/E (40%), P/B (35%), PEG (25% if available)

    Args:
        pe_ratio: Price to earnings ratio
        price_to_book: Price to book ratio
        peg_ratio: PEG ratio (optional)
        sector: Sector code (0-11) for sector-relative P/E scoring

    Returns:
        Tuple of (score 0-100, metrics dict)
    """
    metrics = {}
    components = []
    weights = []

    # P/E Ratio (40%) - Use sector-relative thresholds if sector provided
    if pe_ratio is not None and pe_ratio > 0:
        # Get sector-specific thresholds (default to market average if unknown)
        thresholds = SECTOR_PE_THRESHOLDS.get(sector, SECTOR_PE_THRESHOLDS[0])
        low = thresholds["low"]
        median = thresholds["median"]
        high = thresholds["high"]

        # Score based on where P/E falls relative to sector norms
        # Below low threshold = excellent (90-100)
        # low to median = good (70-90)
        # median to high = average (40-70)
        # above high = poor (0-40)
        if pe_ratio <= low * 0.5:
            pe_score = 100  # Extremely cheap for sector
        elif pe_ratio <= low:
            pe_score = 90 + (low - pe_ratio) / (low * 0.5) * 10  # 90-100
        elif pe_ratio <= median:
            pe_score = 70 + (median - pe_ratio) / (median - low) * 20  # 70-90
        elif pe_ratio <= high:
            pe_score = 40 + (high - pe_ratio) / (high - median) * 30  # 40-70
        elif pe_ratio <= high * 1.5:
            pe_score = 15 + (high * 1.5 - pe_ratio) / (high * 0.5) * 25  # 15-40
        else:
            pe_score = max(0, 15 - (pe_ratio - high * 1.5) / high * 15)  # 0-15

        metrics["pe_ratio"] = {"value": pe_ratio, "score": pe_score, "sector": sector}
        components.append(pe_score)
        weights.append(0.40)

    # Price-to-Book (35%) - Lower is better
    if price_to_book is not None and price_to_book > 0:
        if price_to_book <= 1.0:
            ptb_score = 100  # Trading below book value
        elif price_to_book <= 2.0:
            ptb_score = 80 + (2.0 - price_to_book) * 20  # 80-100
        elif price_to_book <= 3.0:
            ptb_score = 60 + (3.0 - price_to_book) * 20  # 60-80
        elif price_to_book <= 5.0:
            ptb_score = 40 + (5.0 - price_to_book) / 2 * 20  # 40-60
        elif price_to_book <= 10.0:
            ptb_score = 20 + (10.0 - price_to_book) / 5 * 20  # 20-40
        else:
            ptb_score = max(0, 20 - (price_to_book - 10) / 10 * 20)  # 0-20

        metrics["price_to_book"] = {"value": price_to_book, "score": ptb_score}
        components.append(ptb_score)
        weights.append(0.35)

    # PEG Ratio (25%) - <1 is undervalued, >2 is expensive
    if peg_ratio is not None and peg_ratio > 0:
        if peg_ratio <= 0.5:
            peg_score = 100
        elif peg_ratio <= 1.0:
            peg_score = 80 + (1.0 - peg_ratio) / 0.5 * 20  # 80-100
        elif peg_ratio <= 1.5:
            peg_score = 60 + (1.5 - peg_ratio) / 0.5 * 20  # 60-80
        elif peg_ratio <= 2.0:
            peg_score = 40 + (2.0 - peg_ratio) / 0.5 * 20  # 40-60
        elif peg_ratio <= 3.0:
            peg_score = 20 + (3.0 - peg_ratio) / 1.0 * 20  # 20-40
        else:
            peg_score = max(0, 20 - (peg_ratio - 3.0) / 2 * 20)  # 0-20

        metrics["peg_ratio"] = {"value": peg_ratio, "score": peg_score}
        components.append(peg_score)
        weights.append(0.25)

    if not components:
        return None, metrics

    score = sum(c * w for c, w in zip(components, weights)) / sum(weights)
    return score, metrics


def score_sentiment_contrarian(
    short_percent: Optional[float],
    analyst_sentiment: Optional[float],
    earnings_surprise: Optional[float],
    quality_score: Optional[float] = None,
) -> Tuple[Optional[float], Dict]:
    """
    Calculate Sentiment/Contrarian score for short-term horizons.

    This factor captures mean-reversion and momentum signals that work
    best in the 1-6 month timeframe:
    - High short interest + quality = squeeze potential
    - Extreme analyst sentiment = contrarian opportunity
    - Earnings surprise = post-earnings drift momentum

    Weights: Short Interest (40%), Analyst Sentiment (30%), Earnings Surprise (30%)

    Args:
        short_percent: Short interest as float of float (0.15 = 15%)
        analyst_sentiment: Analyst sentiment from -1 (bearish) to +1 (bullish)
        earnings_surprise: Last earnings surprise as decimal (0.10 = 10% beat)
        quality_score: Quality score (0-100) for squeeze potential calculation

    Returns:
        Tuple of (score 0-100, metrics dict)
    """
    metrics = {}
    components = []
    weights = []

    # Short Interest (40%) - Contrarian signal
    # High short interest on quality stocks = squeeze potential
    # High short interest on poor stocks = legitimate concern
    if short_percent is not None:
        # Ensure short_percent is a decimal (some sources report as percentage)
        if short_percent > 1:
            short_percent = short_percent / 100

        # Base score: moderate short interest (5-15%) can be contrarian opportunity
        if short_percent <= 0.03:
            # Very low short interest - not much contrarian opportunity
            short_score = 50
        elif short_percent <= 0.10:
            # Moderate short interest - potential opportunity
            short_score = 50 + (short_percent - 0.03) / 0.07 * 30  # 50-80
        elif short_percent <= 0.20:
            # High short interest - high squeeze potential but risky
            if quality_score is not None and quality_score >= 60:
                # Quality stock with high short = squeeze potential
                short_score = 80 + (short_percent - 0.10) / 0.10 * 20  # 80-100
            else:
                # Low quality with high short = legitimate concern
                short_score = 40 - (short_percent - 0.10) / 0.10 * 20  # 40-20
        else:
            # Extremely high short interest (>20%)
            if quality_score is not None and quality_score >= 70:
                short_score = 95  # Prime squeeze candidate
            else:
                short_score = 20  # Serious concerns

        metrics["short_percent"] = {"value": short_percent, "score": short_score}
        components.append(short_score)
        weights.append(0.40)

    # Analyst Sentiment (30%) - Contrarian signal
    # Extreme sentiment (very bullish or very bearish) often precedes reversal
    if analyst_sentiment is not None:
        # Moderate sentiment is neutral, extremes are contrarian
        if analyst_sentiment >= 0.8:
            # Extremely bullish consensus - contrarian bearish
            sent_score = 30  # Too much optimism
        elif analyst_sentiment >= 0.5:
            # Bullish but not extreme - positive but cautious
            sent_score = 50 + (0.8 - analyst_sentiment) / 0.3 * 20  # 50-70
        elif analyst_sentiment >= 0:
            # Neutral to mildly bullish - good
            sent_score = 70 + analyst_sentiment / 0.5 * 15  # 70-85
        elif analyst_sentiment >= -0.5:
            # Mildly bearish - contrarian opportunity
            sent_score = 70 + abs(analyst_sentiment) / 0.5 * 15  # 70-85
        elif analyst_sentiment >= -0.8:
            # Bearish - stronger contrarian signal
            sent_score = 85 + (abs(analyst_sentiment) - 0.5) / 0.3 * 10  # 85-95
        else:
            # Extremely bearish consensus - strong contrarian buy
            sent_score = 95

        metrics["analyst_sentiment"] = {"value": analyst_sentiment, "score": sent_score}
        components.append(sent_score)
        weights.append(0.30)

    # Earnings Surprise (30%) - Post-earnings drift momentum
    # Stocks tend to drift in the direction of earnings surprise
    if earnings_surprise is not None:
        if earnings_surprise >= 0.20:
            # Strong beat (>20%) - high momentum score
            earn_score = 90 + min(10, (earnings_surprise - 0.20) * 50)  # 90-100
        elif earnings_surprise >= 0.10:
            # Good beat (10-20%)
            earn_score = 75 + (earnings_surprise - 0.10) / 0.10 * 15  # 75-90
        elif earnings_surprise >= 0:
            # Small beat (0-10%)
            earn_score = 55 + earnings_surprise / 0.10 * 20  # 55-75
        elif earnings_surprise >= -0.10:
            # Small miss (0-10%)
            earn_score = 35 + (earnings_surprise + 0.10) / 0.10 * 20  # 35-55
        elif earnings_surprise >= -0.20:
            # Big miss (10-20%)
            earn_score = 15 + (earnings_surprise + 0.20) / 0.10 * 20  # 15-35
        else:
            # Major miss (>20%)
            earn_score = max(0, 15 + (earnings_surprise + 0.20) * 50)  # 0-15

        metrics["earnings_surprise"] = {"value": earnings_surprise, "score": earn_score}
        components.append(earn_score)
        weights.append(0.30)

    if not components:
        return None, metrics

    score = sum(c * w for c, w in zip(components, weights)) / sum(weights)
    return score, metrics


def calculate_composite_score(
    fundamentals: Dict,
    price_momentum: Optional[float] = None,
    horizon_months: int = 12,
) -> Optional[Dict]:
    """
    Calculate complete composite score for a stock.

    This is the MAIN scoring function used by both live and backtest systems.

    Args:
        fundamentals: Dict with fundamental metrics (roe, roa, pe_ratio, etc.)
        price_momentum: Price return over period as decimal (optional)
        horizon_months: Investment horizon for weight selection

    Returns:
        Dict with composite score, sub-scores, grade, and all metrics
    """
    if not fundamentals:
        return None

    # Get sector for sector-relative valuation scoring
    sector = fundamentals.get("sector")

    # Calculate sub-scores
    growth_score, growth_metrics = score_growth(
        revenue_growth=fundamentals.get("revenue_growth"),
        earnings_growth=fundamentals.get("earnings_growth"),
        price_momentum=price_momentum,
    )

    quality_score, quality_metrics = score_quality(
        roe=fundamentals.get("roe"),
        roa=fundamentals.get("roa"),
        profit_margin=fundamentals.get("profit_margin"),
        operating_margin=fundamentals.get("operating_margin"),
    )

    financial_score, financial_metrics = score_financial_strength(
        debt_to_equity=fundamentals.get("debt_to_equity"),
        current_ratio=fundamentals.get("current_ratio"),
    )

    # Use sector-relative valuation scoring
    valuation_score, valuation_metrics = score_valuation(
        pe_ratio=fundamentals.get("pe_ratio"),
        price_to_book=fundamentals.get("price_to_book"),
        peg_ratio=fundamentals.get("peg_ratio"),
        sector=sector,
    )

    # Calculate sentiment score for short-term horizons (<=12 months)
    sentiment_score = None
    sentiment_metrics = {}
    if horizon_months <= 12:
        sentiment_score, sentiment_metrics = score_sentiment_contrarian(
            short_percent=fundamentals.get("short_percent"),
            analyst_sentiment=fundamentals.get("analyst_sentiment"),
            earnings_surprise=fundamentals.get("earnings_surprise"),
            quality_score=quality_score,
        )

    # Get weights for this horizon
    weights = get_weights(horizon_months)

    # Calculate weighted composite
    scores = {
        "growth": growth_score,
        "quality": quality_score,
        "financial": financial_score,
        "valuation": valuation_score,
        "sentiment": sentiment_score,
    }

    total_weight = 0
    weighted_sum = 0

    for factor, weight in weights.items():
        if scores.get(factor) is not None and weight > 0:
            weighted_sum += scores[factor] * weight
            total_weight += weight

    if total_weight == 0:
        return None

    composite = weighted_sum / total_weight
    grade = get_grade(composite)

    return {
        "composite_score": round(composite, 1),
        "grade": grade,
        "growth_score": round(growth_score, 1) if growth_score else None,
        "quality_score": round(quality_score, 1) if quality_score else None,
        "financial_score": round(financial_score, 1) if financial_score else None,
        "valuation_score": round(valuation_score, 1) if valuation_score else None,
        "sentiment_score": round(sentiment_score, 1) if sentiment_score else None,
        "weights": weights,
        "horizon_months": horizon_months,
        "sector": sector,
        "metrics": {
            "growth": growth_metrics,
            "quality": quality_metrics,
            "financial": financial_metrics,
            "valuation": valuation_metrics,
            "sentiment": sentiment_metrics,
        },
    }
