import { useState, useEffect, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Estimate backtest time in seconds based on test periods
function getEstimatedSeconds(periods) {
  const p = Number(periods) || 30
  // Roughly 9 seconds per period for 61 stocks (measured: 10 periods = 90 sec)
  return Math.round(p * 9)
}

// Format seconds as "X:XX"
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

function ScreenerBacktest() {
  const [days, setDays] = useState('5')
  const [minConfidence, setMinConfidence] = useState('75')
  const [testPeriods, setTestPeriods] = useState('30')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const timerRef = useRef(null)

  // Timer effect
  useEffect(() => {
    if (loading) {
      setElapsedSeconds(0)
      timerRef.current = setInterval(() => {
        setElapsedSeconds(s => s + 1)
      }, 1000)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [loading])

  const handleSubmit = async (e) => {
    e.preventDefault()

    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const confidence = Number(minConfidence) / 100
      const response = await fetch(
        `${API_BASE}/screener-backtest?days=${days}&min_confidence=${confidence}&test_periods=${testPeriods}`
      )

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Backtest failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const getWinRateColor = (rate) => {
    if (rate >= 60) return 'accuracy-great'
    if (rate >= 55) return 'accuracy-good'
    if (rate >= 50) return 'accuracy-ok'
    return 'accuracy-poor'
  }

  const getReturnColor = (ret) => {
    if (ret > 0) return 'accuracy-good'
    return 'accuracy-poor'
  }

  return (
    <div className="card backtest-card">
      <h2>Screener Backtest</h2>
      <p className="card-description">
        Comprehensive backtest of the screener strategy with confidence buckets, risk metrics,
        regime analysis, and market-relative validation. Uses non-overlapping periods and long-only trades.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="screener-days">Days Ahead</label>
            <input
              id="screener-days"
              type="number"
              min="1"
              max="30"
              value={days}
              onChange={(e) => setDays(e.target.value)}
              disabled={loading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="screener-confidence">Min Confidence (%)</label>
            <input
              id="screener-confidence"
              type="number"
              min="50"
              max="95"
              value={minConfidence}
              onChange={(e) => setMinConfidence(e.target.value)}
              disabled={loading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="screener-periods">Test Periods</label>
            <input
              id="screener-periods"
              type="number"
              min="10"
              max="100"
              value={testPeriods}
              onChange={(e) => setTestPeriods(e.target.value)}
              disabled={loading}
            />
          </div>
        </div>

        <button type="submit" disabled={loading}>
          {loading && <span className="spinner"></span>}
          {loading ? 'Running Backtest...' : 'Run Screener Backtest'}
        </button>
      </form>

      {loading && (
        <div className="loading-text">
          <p>Testing screener on ~61 stocks over {testPeriods} non-overlapping periods...</p>
          <p className="loading-timer">
            ⏱ Elapsed: {formatTime(elapsedSeconds)}
            {' '} • {' '}
            Est. remaining: ~{formatTime(Math.max(0, getEstimatedSeconds(testPeriods) - elapsedSeconds))}
          </p>
        </div>
      )}

      {error && (
        <div className="error">{error}</div>
      )}

      {result && (
        <div className="backtest-results">
          <h3>Screener Strategy Results</h3>
          <p className="backtest-subtitle">
            {result.summary.stocks_tested} stocks tested, {result.summary.total_picks} screener picks over {result.test_periods} periods
          </p>

          {/* Summary Metrics */}
          <div className="backtest-metrics">
            <div className={`metric-card ${getWinRateColor(result.summary.win_rate)}`}>
              <div className="metric-value">{result.summary.win_rate}%</div>
              <div className="metric-label">Win Rate</div>
              <div className="metric-detail">
                {result.summary.winning_picks}/{result.summary.total_picks} picks went up
              </div>
            </div>

            <div className={`metric-card ${getReturnColor(result.summary.avg_pick_return)}`}>
              <div className="metric-value">{result.summary.avg_pick_return > 0 ? '+' : ''}{result.summary.avg_pick_return}%</div>
              <div className="metric-label">Avg Return Per Pick</div>
              <div className="metric-detail">
                vs {result.summary.avg_baseline_return > 0 ? '+' : ''}{result.summary.avg_baseline_return}% market avg
              </div>
            </div>

            <div className={`metric-card ${getReturnColor(result.summary.alpha_vs_baseline)}`}>
              <div className="metric-value">{result.summary.alpha_vs_baseline > 0 ? '+' : ''}{result.summary.alpha_vs_baseline}%</div>
              <div className="metric-label">Alpha Per Trade</div>
              <div className="metric-detail">excess return vs baseline</div>
            </div>

            {result.risk_metrics && (
              <div className={`metric-card ${result.risk_metrics.worst_single_trade > -10 ? 'accuracy-ok' : 'accuracy-poor'}`}>
                <div className="metric-value">{result.risk_metrics.worst_single_trade}%</div>
                <div className="metric-label">Worst Trade</div>
                <div className="metric-detail">max single loss</div>
              </div>
            )}
          </div>

          {/* 1. CONFIDENCE BUCKETS */}
          {result.confidence_buckets && result.confidence_buckets.length > 0 && (
            <div className="analysis-section">
              <h4>Confidence Bucket Analysis</h4>
              <p className="section-note">Does higher confidence correlate with better outcomes?</p>
              <div className="predictions-table">
                <table>
                  <thead>
                    <tr>
                      <th>Confidence</th>
                      <th>Trades</th>
                      <th>Win Rate</th>
                      <th>Avg Return</th>
                      <th>Worst Trade</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.confidence_buckets.map((bucket, idx) => (
                      <tr key={idx}>
                        <td><strong>{bucket.bucket}</strong></td>
                        <td>{bucket.trades}</td>
                        <td className={bucket.win_rate >= 55 ? 'change-up' : bucket.win_rate >= 50 ? '' : 'change-down'}>
                          {bucket.win_rate}%
                        </td>
                        <td className={bucket.avg_return >= 0 ? 'change-up' : 'change-down'}>
                          {bucket.avg_return >= 0 ? '+' : ''}{bucket.avg_return}%
                        </td>
                        <td className="change-down">{bucket.worst_trade}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 2. RISK METRICS */}
          {result.risk_metrics && (
            <div className="analysis-section">
              <h4>Risk Metrics</h4>
              <p className="section-note">Drawdown and loss streak analysis for risk management.</p>
              <div className="risk-grid">
                <div className="risk-item">
                  <span className="risk-label">Worst Single Trade:</span>
                  <span className="risk-value change-down">{result.risk_metrics.worst_single_trade}%</span>
                </div>
                <div className="risk-item">
                  <span className="risk-label">Worst Loss Streak:</span>
                  <span className="risk-value">
                    {result.risk_metrics.worst_streak_length} trades ({result.risk_metrics.worst_streak_loss}%)
                  </span>
                </div>
                <div className="risk-item">
                  <span className="risk-label">Max Drawdown:</span>
                  <span className="risk-value change-down">{result.risk_metrics.max_drawdown}%</span>
                </div>
              </div>
            </div>
          )}

          {/* 3. REGIME ANALYSIS */}
          {result.regime_analysis && (
            <div className="analysis-section">
              <h4>Regime Segmentation</h4>
              <p className="section-note">How does the screener perform in different market conditions?</p>

              {result.regime_analysis.volatility && result.regime_analysis.volatility.length > 0 && (
                <div className="regime-subsection">
                  <h5>By Volatility</h5>
                  <div className="predictions-table">
                    <table>
                      <thead>
                        <tr>
                          <th>Regime</th>
                          <th>Trades</th>
                          <th>Win Rate</th>
                          <th>Avg Return</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.regime_analysis.volatility.map((regime, idx) => (
                          <tr key={idx}>
                            <td><strong>{regime.regime}</strong></td>
                            <td>{regime.trades}</td>
                            <td className={regime.win_rate >= 55 ? 'change-up' : regime.win_rate >= 50 ? '' : 'change-down'}>
                              {regime.win_rate}%
                            </td>
                            <td className={regime.avg_return >= 0 ? 'change-up' : 'change-down'}>
                              {regime.avg_return >= 0 ? '+' : ''}{regime.avg_return}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {result.regime_analysis.market_trend && result.regime_analysis.market_trend.length > 0 && (
                <div className="regime-subsection">
                  <h5>By Market Trend (SPY)</h5>
                  <div className="predictions-table">
                    <table>
                      <thead>
                        <tr>
                          <th>Regime</th>
                          <th>Trades</th>
                          <th>Win Rate</th>
                          <th>Avg Return</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.regime_analysis.market_trend.map((regime, idx) => (
                          <tr key={idx}>
                            <td><strong>{regime.regime}</strong></td>
                            <td>{regime.trades}</td>
                            <td className={regime.win_rate >= 55 ? 'change-up' : regime.win_rate >= 50 ? '' : 'change-down'}>
                              {regime.win_rate}%
                            </td>
                            <td className={regime.avg_return >= 0 ? 'change-up' : 'change-down'}>
                              {regime.avg_return >= 0 ? '+' : ''}{regime.avg_return}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 4. MARKET COMPARISON */}
          {result.market_comparison && (
            <div className="analysis-section">
              <h4>Market-Relative Validation (vs SPY)</h4>
              <p className="section-note">Did screener picks beat the market over the same periods?</p>
              <div className="market-grid">
                <div className="market-item">
                  <span className="market-label">Picks with SPY benchmark:</span>
                  <span className="market-value">{result.market_comparison.picks_with_benchmark}</span>
                </div>
                <div className="market-item">
                  <span className="market-label">Beat SPY:</span>
                  <span className={`market-value ${result.market_comparison.beat_market_pct >= 50 ? 'change-up' : 'change-down'}`}>
                    {result.market_comparison.beat_market_count} ({result.market_comparison.beat_market_pct}%)
                  </span>
                </div>
                <div className="market-item">
                  <span className="market-label">Avg Alpha vs SPY:</span>
                  <span className={`market-value ${result.market_comparison.avg_alpha >= 0 ? 'change-up' : 'change-down'}`}>
                    {result.market_comparison.avg_alpha >= 0 ? '+' : ''}{result.market_comparison.avg_alpha}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* 5. NO-TRADE FREQUENCY */}
          {result.no_trade_analysis && (
            <div className="analysis-section">
              <h4>Trade Frequency Analysis</h4>
              <p className="section-note">How often does the screener find opportunities?</p>
              <div className="trade-freq-grid">
                <div className="trade-freq-item">
                  <span className="trade-freq-label">Total test periods:</span>
                  <span className="trade-freq-value">{result.no_trade_analysis.total_periods}</span>
                </div>
                <div className="trade-freq-item">
                  <span className="trade-freq-label">Periods with picks:</span>
                  <span className="trade-freq-value">{result.no_trade_analysis.periods_with_picks}</span>
                </div>
                <div className="trade-freq-item">
                  <span className="trade-freq-label">No-trade periods:</span>
                  <span className="trade-freq-value">
                    {result.no_trade_analysis.periods_without_picks} ({result.no_trade_analysis.no_trade_pct}%)
                  </span>
                </div>
                <div className="trade-freq-item">
                  <span className="trade-freq-label">Avg picks per period:</span>
                  <span className="trade-freq-value">{result.no_trade_analysis.avg_picks_per_period}</span>
                </div>
              </div>
            </div>
          )}

          {/* INTERPRETATION */}
          <div className="backtest-interpretation">
            <h4>Key Takeaways</h4>
            {result.summary.win_rate >= 55 ? (
              <p className="interpretation-good">
                {result.summary.win_rate >= 60 ? '✓ Strong:' : '✓ Positive:'} Screener picks went up {result.summary.win_rate}% of the time (above 50% random).
              </p>
            ) : result.summary.win_rate >= 50 ? (
              <p className="interpretation-ok">
                ~ Marginal: Win rate of {result.summary.win_rate}% is near random chance.
              </p>
            ) : (
              <p className="interpretation-poor">
                ✗ Warning: Win rate below 50% suggests the screener underperformed random selection.
              </p>
            )}

            {result.summary.alpha_vs_baseline > 0 ? (
              <p className="interpretation-good">
                ✓ Alpha: Screener picks averaged +{result.summary.alpha_vs_baseline}% excess return per trade vs baseline.
              </p>
            ) : (
              <p className="interpretation-ok">
                ~ No alpha: Screener picks underperformed baseline by {Math.abs(result.summary.alpha_vs_baseline)}% per trade.
              </p>
            )}

            {result.confidence_buckets && result.confidence_buckets.length >= 2 && (
              result.confidence_buckets[result.confidence_buckets.length - 1].win_rate >
              result.confidence_buckets[0].win_rate ? (
                <p className="interpretation-good">
                  ✓ Confidence calibration works: Higher confidence buckets have better win rates.
                </p>
              ) : (
                <p className="interpretation-ok">
                  ~ Confidence not well-calibrated: Higher confidence doesn't consistently mean better outcomes.
                </p>
              )
            )}

            {result.risk_metrics && result.risk_metrics.max_drawdown > 20 && (
              <p className="interpretation-poor">
                ⚠ Risk warning: Max drawdown of {result.risk_metrics.max_drawdown}% - size positions carefully.
              </p>
            )}

            <p className="interpretation-note">
              Note: Uses non-overlapping {result.days}-day periods. Long-only (no shorting). Returns are per-trade averages.
            </p>
          </div>

          {/* TOP STOCKS */}
          {result.top_stocks && result.top_stocks.length > 0 && (
            <div className="recent-predictions">
              <h4>Top Performing Stocks (when picked)</h4>
              <div className="predictions-table">
                <table>
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th>Avg Return</th>
                      <th>Times Picked</th>
                      <th>Win Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.top_stocks.map((stock, idx) => (
                      <tr key={idx}>
                        <td><strong>{stock.ticker}</strong></td>
                        <td className={stock.avg_return >= 0 ? 'change-up' : 'change-down'}>
                          {stock.avg_return >= 0 ? '+' : ''}{stock.avg_return}%
                        </td>
                        <td>{stock.num_picks}x</td>
                        <td>{stock.win_rate}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* BEST/WORST PICKS */}
          <div className="picks-columns">
            {result.best_picks && result.best_picks.length > 0 && (
              <div className="recent-predictions picks-column">
                <h4>Best Picks</h4>
                <div className="predictions-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Return</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.best_picks.map((pick, idx) => (
                        <tr key={idx}>
                          <td><strong>{pick.ticker}</strong></td>
                          <td className="change-up">+{(pick.actual_return * 100).toFixed(1)}%</td>
                          <td>{(pick.confidence * 100).toFixed(0)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {result.worst_picks && result.worst_picks.length > 0 && (
              <div className="recent-predictions picks-column">
                <h4>Worst Picks</h4>
                <div className="predictions-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Return</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.worst_picks.map((pick, idx) => (
                        <tr key={idx}>
                          <td><strong>{pick.ticker}</strong></td>
                          <td className="change-down">{(pick.actual_return * 100).toFixed(1)}%</td>
                          <td>{(pick.confidence * 100).toFixed(0)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default ScreenerBacktest
