# /validate-grades

Run the walk-forward grade validation test on S&P 500 stocks.

## Usage

```
/validate-grades [horizon]
```

- `horizon` (optional): 3, 6, 12, 24, 60, or 120 months. Default: all horizons.

## What This Does

1. Tests the grading system (`composite_backtest_grades.py`) against 457 S&P 500 stocks
2. Uses walk-forward validation across 10Y, 5Y, 2Y, 1Y historical periods
3. Reports success rates for A-grade and B-grade picks
4. Analyzes factor correlations (momentum, trend, fundamental, volatility)

## How to Run

Run the full validation script:

```bash
python run_sp500_validation.py
```

Or for a single horizon, use Python directly:

```python
from composite_backtest_grades import run_walk_forward_grade_validation

result = run_walk_forward_grade_validation(
    stocks=SP500_STOCKS,  # from run_sp500_validation.py
    horizon_months=12,    # 3, 6, 12, 24, 60, or 120
    test_periods=[10, 5, 2, 1],
    max_workers=8,
)
print(f"Verdict: {result['verdict']}")
print(f"A+B Success: {result['summary']['avg_ab_combined_success_rate']}%")
```

## Interpreting Results

- **VALIDATED**: 70%+ success rate - grading system works well
- **PARTIALLY VALIDATED**: 55-70% - moderate predictive power
- **WEAK**: 40-55% - marginal predictive value
- **NOT VALIDATED**: <40% - grading system unreliable

## Current Baseline (Feb 2026)

| Horizon | A+B Success | Best Factor |
|---------|-------------|-------------|
| 3M | 53.0% | fundamental |
| 6M | 56.2% | fundamental |
| 12M | 57.8% | volatility |
| 24M | 56.0% | fundamental |
| 60M | 62.5% | fundamental |
