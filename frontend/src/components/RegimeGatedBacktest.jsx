import { useState } from 'react'
import useJob from '../hooks/useJob'

// Format seconds as "X:XX"
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

function RegimeGatedBacktest() {
  const [days, setDays] = useState('5')

  const {
    status,
    progress,
    progressMessage,
    result,
    error,
    elapsedSeconds,
    isLoading,
    isCancelled,
    startJob,
    cancelJob,
  } = useJob('regime-gated-backtest')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (isLoading) return

    try {
      await startJob('regime-gated-backtest', {
        days: Number(days),
      })
    } catch (err) {
      console.error('Failed to start regime-gated backtest:', err)
    }
  }

  const handleCancel = async () => {
    await cancelJob()
  }

  const getValueColor = (value, isPositive = true) => {
    if (isPositive) {
      return value > 0 ? 'change-up' : value < 0 ? 'change-down' : ''
    } else {
      return value < 0 ? 'change-up' : value > 0 ? 'change-down' : ''
    }
  }

  const getWinRateColor = (rate) => {
    if (rate >= 60) return 'change-up'
    if (rate >= 50) return ''
    return 'change-down'
  }

  return (
    <div className="card backtest-card">
      <h2>Regime-Gated Backtest</h2>
      <p className="card-description">
        Test market regime gate strategy: suppress trades or raise confidence threshold
        when SPY trailing return is below -1% (bear market signal). Compares three modes
        to evaluate risk reduction.
      </p>

      <form onSubmit={handleSubmit} className={isLoading ? 'form-disabled' : ''}>
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="regime-days">Days Ahead</label>
            <input
              id="regime-days"
              type="number"
              min="1"
              max="30"
              value={days}
              onChange={(e) => setDays(e.target.value)}
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <label>Settings (Fixed)</label>
            <div className="settings-info">
              Normal: ≥75% conf<br/>
              Bear: ≥80% or suppress
            </div>
          </div>
        </div>

        <div className="button-row">
          <button type="submit" disabled={isLoading}>
            {isLoading && <span className="spinner"></span>}
            {isLoading ? 'Running Backtest...' : 'Run Regime-Gated Backtest'}
          </button>
          {isLoading && (
            <button
              type="button"
              className="cancel-button"
              onClick={handleCancel}
            >
              <span className="cancel-icon">✕</span>
              Cancel
            </button>
          )}
        </div>
      </form>

      {isLoading && (
        <div className="job-progress-container">
          <div className="job-progress-header">
            <span className="job-progress-text">{progressMessage || 'Testing screener with regime gate...'}</span>
            <span className="job-progress-percent">{progress}%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-bar-fill"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <div className="job-timer">
            <span>Elapsed: {formatTime(elapsedSeconds)}</span>
          </div>
          <p className="loading-note">
            Tip: After running once, requests load instantly from cache.
          </p>
        </div>
      )}

      {isCancelled && (
        <div className="cancelled-message">
          <span className="cancelled-icon">⚠️</span>
          <span>Backtest was cancelled. Click "Run Regime-Gated Backtest" to start again.</span>
        </div>
      )}

      {error && !isCancelled && (
        <div className="error">{error}</div>
      )}

      {result && (
        <div className="backtest-results">
          <h3>
            Regime Gate Comparison
            {result._cached && <span className="cache-badge">From Cache</span>}
          </h3>
          <p className="backtest-subtitle">
            {result.summary.stocks_tested} stocks tested over {result.summary.total_periods} periods
            ({result.summary.bear_periods} bear, {result.summary.normal_periods} normal)
          </p>

          {/* Regime Info */}
          <div className="regime-info-box">
            <div className="regime-info-item">
              <span className="regime-label">Bear Trigger:</span>
              <span className="regime-value">{result.summary.bear_threshold}</span>
            </div>
            <div className="regime-info-item">
              <span className="regime-label">Normal Confidence:</span>
              <span className="regime-value">{result.summary.normal_confidence}</span>
            </div>
            <div className="regime-info-item">
              <span className="regime-label">Bear Confidence:</span>
              <span className="regime-value">{result.summary.bear_confidence}</span>
            </div>
            <div className="regime-info-item">
              <span className="regime-label">Bear Periods:</span>
              <span className="regime-value">{result.period_breakdown.bear_periods_pct}%</span>
            </div>
          </div>

          {/* COMPARISON TABLE */}
          <div className="analysis-section">
            <h4>Mode Comparison</h4>
            <p className="section-note">
              Same stocks, same periods, different regime rules applied.
            </p>
            <div className="comparison-table-wrapper">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>No Gate (Baseline)</th>
                    <th>Bear: Suppressed</th>
                    <th>Bear: 80% Conf</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Total Trades</strong></td>
                    <td>{result.comparison.no_gate.total_trades}</td>
                    <td>
                      {result.comparison.suppressed.total_trades}
                      {result.comparison.suppressed.trade_reduction > 0 && (
                        <span className="delta-badge negative">
                          -{result.comparison.suppressed.trade_reduction}%
                        </span>
                      )}
                    </td>
                    <td>
                      {result.comparison.high_confidence.total_trades}
                      {result.comparison.high_confidence.trade_reduction > 0 && (
                        <span className="delta-badge negative">
                          -{result.comparison.high_confidence.trade_reduction}%
                        </span>
                      )}
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Win Rate</strong></td>
                    <td className={getWinRateColor(result.comparison.no_gate.win_rate)}>
                      {result.comparison.no_gate.win_rate}%
                    </td>
                    <td className={getWinRateColor(result.comparison.suppressed.win_rate)}>
                      {result.comparison.suppressed.win_rate}%
                      {result.comparison.suppressed.win_rate > result.comparison.no_gate.win_rate && (
                        <span className="delta-badge positive">
                          +{(result.comparison.suppressed.win_rate - result.comparison.no_gate.win_rate).toFixed(1)}
                        </span>
                      )}
                    </td>
                    <td className={getWinRateColor(result.comparison.high_confidence.win_rate)}>
                      {result.comparison.high_confidence.win_rate}%
                      {result.comparison.high_confidence.win_rate > result.comparison.no_gate.win_rate && (
                        <span className="delta-badge positive">
                          +{(result.comparison.high_confidence.win_rate - result.comparison.no_gate.win_rate).toFixed(1)}
                        </span>
                      )}
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Avg Return</strong></td>
                    <td className={getValueColor(result.comparison.no_gate.avg_return)}>
                      {result.comparison.no_gate.avg_return > 0 ? '+' : ''}{result.comparison.no_gate.avg_return}%
                    </td>
                    <td className={getValueColor(result.comparison.suppressed.avg_return)}>
                      {result.comparison.suppressed.avg_return > 0 ? '+' : ''}{result.comparison.suppressed.avg_return}%
                    </td>
                    <td className={getValueColor(result.comparison.high_confidence.avg_return)}>
                      {result.comparison.high_confidence.avg_return > 0 ? '+' : ''}{result.comparison.high_confidence.avg_return}%
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Worst Trade</strong></td>
                    <td className="change-down">{result.comparison.no_gate.worst_trade}%</td>
                    <td className="change-down">
                      {result.comparison.suppressed.worst_trade}%
                      {result.comparison.suppressed.worst_trade > result.comparison.no_gate.worst_trade && (
                        <span className="delta-badge positive">better</span>
                      )}
                    </td>
                    <td className="change-down">
                      {result.comparison.high_confidence.worst_trade}%
                      {result.comparison.high_confidence.worst_trade > result.comparison.no_gate.worst_trade && (
                        <span className="delta-badge positive">better</span>
                      )}
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Worst Streak</strong></td>
                    <td>
                      {result.comparison.no_gate.worst_streak_length} trades
                      ({result.comparison.no_gate.worst_streak_loss}%)
                    </td>
                    <td>
                      {result.comparison.suppressed.worst_streak_length} trades
                      ({result.comparison.suppressed.worst_streak_loss}%)
                    </td>
                    <td>
                      {result.comparison.high_confidence.worst_streak_length} trades
                      ({result.comparison.high_confidence.worst_streak_loss}%)
                    </td>
                  </tr>
                  <tr className="highlight-row">
                    <td><strong>Max Drawdown</strong></td>
                    <td className="change-down">{result.comparison.no_gate.max_drawdown}%</td>
                    <td className={result.comparison.suppressed.max_drawdown < result.comparison.no_gate.max_drawdown ? 'change-up' : 'change-down'}>
                      {result.comparison.suppressed.max_drawdown}%
                      {result.comparison.suppressed.drawdown_improvement > 0 && (
                        <span className="delta-badge positive">
                          -{result.comparison.suppressed.drawdown_improvement}%
                        </span>
                      )}
                    </td>
                    <td className={result.comparison.high_confidence.max_drawdown < result.comparison.no_gate.max_drawdown ? 'change-up' : 'change-down'}>
                      {result.comparison.high_confidence.max_drawdown}%
                      {result.comparison.high_confidence.drawdown_improvement > 0 && (
                        <span className="delta-badge positive">
                          -{result.comparison.high_confidence.drawdown_improvement}%
                        </span>
                      )}
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Beat SPY %</strong></td>
                    <td>{result.comparison.no_gate.beat_spy_pct}%</td>
                    <td>
                      {result.comparison.suppressed.beat_spy_pct}%
                      {result.comparison.suppressed.beat_spy_pct > result.comparison.no_gate.beat_spy_pct && (
                        <span className="delta-badge positive">
                          +{(result.comparison.suppressed.beat_spy_pct - result.comparison.no_gate.beat_spy_pct).toFixed(1)}
                        </span>
                      )}
                    </td>
                    <td>
                      {result.comparison.high_confidence.beat_spy_pct}%
                      {result.comparison.high_confidence.beat_spy_pct > result.comparison.no_gate.beat_spy_pct && (
                        <span className="delta-badge positive">
                          +{(result.comparison.high_confidence.beat_spy_pct - result.comparison.no_gate.beat_spy_pct).toFixed(1)}
                        </span>
                      )}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* INTERPRETATION */}
          <div className="backtest-interpretation">
            <h4>Analysis</h4>

            {result.interpretation && result.interpretation.map((note, idx) => (
              <p key={idx} className={
                note.includes('improves') || note.includes('reduces')
                  ? 'interpretation-good'
                  : note.includes('Trade-off')
                    ? 'interpretation-ok'
                    : ''
              }>
                {note.includes('improves') || note.includes('reduces') ? '✓ ' :
                 note.includes('Trade-off') ? '⚠ ' : '• '}
                {note}
              </p>
            ))}

            <div className="interpretation-section">
              <h5>How the Regime Gate Works</h5>
              <p>
                The gate checks SPY's <strong>trailing</strong> 5-day return (ending at decision date, no future data).
                When this return is below -1%, the market is flagged as "bear mode" and stricter rules apply.
              </p>
            </div>

            <div className="interpretation-section">
              <h5>Why This Helps</h5>
              <p>
                In bear markets, correlations increase and most stocks fall together.
                The regime gate reduces exposure during these periods, either by skipping trades entirely
                (suppressed) or requiring higher conviction (80% confidence).
              </p>
            </div>

            <div className="interpretation-section">
              <h5>Trade-offs</h5>
              <p>
                Fewer trades means potentially missing profitable opportunities.
                The suppressed mode is more conservative (lowest drawdown) but may miss rebounds.
                The 80% confidence mode maintains some exposure while filtering out weaker signals.
              </p>
            </div>

            <p className="interpretation-note">
              Note: Uses non-overlapping {result.days}-day periods. SPY trailing return calculated from past data only.
              All thresholds fixed before testing (no optimization on test data).
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default RegimeGatedBacktest
