# /improve-model

Analyze and improve the stock grading model based on validation results.

## Usage

```
/improve-model [horizon]
```

## What This Does

1. Runs validation on S&P 500 stocks to get factor correlations
2. Analyzes which factors predict returns best
3. Suggests weight adjustments based on correlations
4. Updates `composite_backtest_grades.py` and `composite_score.py`

## Key Files

- `composite_backtest_grades.py` - Walk-forward grading (momentum, trend, fundamental, volatility)
- `composite_score.py` - Live scoring (growth, quality, financial_strength, valuation)
- `run_sp500_validation.py` - Validation script

## Factor Mapping

| Backtest Factor | Live Score Factor |
|-----------------|-------------------|
| momentum | growth |
| fundamental | quality + financial + valuation |
| volatility | (no direct mapping) |
| trend | (no direct mapping) |

## Current Validated Correlations (Feb 2026)

| Horizon | Best Factor | r | Action |
|---------|-------------|------|--------|
| 3M | fundamental | +0.028 | Increase fundamental weight |
| 6M | fundamental | +0.042 | Increase fundamental weight |
| 12M | volatility | +0.313 | Invert volatility scoring |
| 24M | fundamental | +0.063 | Maximize fundamental weight |
| 60M | fundamental | +0.028 | Maximize fundamental weight |

## Improvement Process

1. Run validation: `python run_sp500_validation.py`
2. Check factor correlations in output
3. Adjust weights in `_calculate_point_in_time_score()` to match correlations
4. Bump cache version (e.g., `grade_validation_v2` → `v3`)
5. Re-run validation to verify improvement

## Weight Guidelines

- Factors with **positive correlation** → increase weight
- Factors with **negative correlation** → minimize weight (0.05)
- Factor with **highest |r|** → give highest weight
- Sum of weights must equal 1.0
