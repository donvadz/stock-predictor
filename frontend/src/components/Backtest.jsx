import { useState } from 'react'
import useJob from '../hooks/useJob'

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

function Backtest() {
  const [ticker, setTicker] = useState('AAPL')
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
  } = useJob('backtest')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (isLoading || !ticker.trim()) return

    try {
      await startJob('backtest', {
        ticker: ticker.toUpperCase(),
        days: Number(days),
      })
    } catch (err) {
      console.error('Failed to start backtest:', err)
    }
  }

  const handleCancel = async () => {
    await cancelJob()
  }

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 60) return 'accuracy-great'
    if (accuracy >= 55) return 'accuracy-good'
    if (accuracy >= 50) return 'accuracy-ok'
    return 'accuracy-poor'
  }

  return (
    <div className="card backtest-card">
      <h2>Model Backtest</h2>
      <p className="card-description">
        Test model accuracy on historical data using walk-forward validation.
        The model makes predictions at past dates and compares to actual outcomes.
      </p>

      <form onSubmit={handleSubmit} className={isLoading ? 'form-disabled' : ''}>
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="backtest-ticker">Ticker</label>
            <input
              id="backtest-ticker"
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="AAPL"
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="backtest-days">Days Ahead</label>
            <input
              id="backtest-days"
              type="number"
              min="1"
              max="30"
              value={days}
              onChange={(e) => setDays(e.target.value)}
              disabled={isLoading}
            />
          </div>
        </div>

        <div className="button-row">
          <button type="submit" disabled={isLoading || !ticker.trim()}>
            {isLoading && <span className="spinner"></span>}
            {isLoading ? 'Running Backtest...' : 'Run Backtest'}
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
            <span className="job-progress-text">{progressMessage || `Running backtest on ${ticker}...`}</span>
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
        </div>
      )}

      {isCancelled && (
        <div className="cancelled-message">
          <span className="cancelled-icon">⚠️</span>
          <span>Backtest was cancelled. Click "Run Backtest" to start again.</span>
        </div>
      )}

      {error && !isCancelled && (
        <div className="error">{error}</div>
      )}

      {result && (
        <div className="backtest-results">
          <h3>{result.name} ({result.ticker})</h3>
          <p className="backtest-subtitle">
            {result.summary.total_predictions} predictions over {result.days}-day periods
          </p>

          <div className="backtest-metrics">
            <div className={`metric-card ${getAccuracyColor(result.summary.accuracy)}`}>
              <div className="metric-value">{result.summary.accuracy}%</div>
              <div className="metric-label">Direction Accuracy</div>
              <div className="metric-detail">
                {result.summary.correct_predictions}/{result.summary.total_predictions} correct
              </div>
            </div>

            {result.summary.high_confidence_accuracy && (
              <div className={`metric-card ${getAccuracyColor(result.summary.high_confidence_accuracy)}`}>
                <div className="metric-value">{result.summary.high_confidence_accuracy}%</div>
                <div className="metric-label">High Confidence (≥75%)</div>
              </div>
            )}

            <div className={`metric-card ${getAccuracyColor(result.summary.up_win_rate)}`}>
              <div className="metric-value">{result.summary.up_win_rate}%</div>
              <div className="metric-label">UP Prediction Win Rate</div>
              <div className="metric-detail">
                {result.summary.up_predictions} UP calls made
              </div>
            </div>

            <div className={`metric-card ${result.summary.avg_return_when_up >= 0 ? 'accuracy-good' : 'accuracy-poor'}`}>
              <div className="metric-value">{result.summary.avg_return_when_up >= 0 ? '+' : ''}{result.summary.avg_return_when_up}%</div>
              <div className="metric-label">Avg Return (Long-Only)</div>
              <div className="metric-detail">
                per trade when buying UP calls
              </div>
            </div>
          </div>

          <div className="backtest-interpretation">
            <h4>Interpretation (Long-Only Strategy)</h4>
            {result.summary.accuracy >= 55 ? (
              <p className="interpretation-good">
                ✓ The model shows predictive power above random chance (50%).
                {result.summary.accuracy >= 60 && " This is quite good for stock prediction!"}
              </p>
            ) : result.summary.accuracy >= 50 ? (
              <p className="interpretation-ok">
                ~ The model performs near random chance. Predictions should be used with caution.
              </p>
            ) : (
              <p className="interpretation-poor">
                ✗ The model underperforms random chance on this stock. Consider not using predictions for {result.ticker}.
              </p>
            )}

            {result.summary.up_win_rate >= 55 ? (
              <p className="interpretation-good">
                ✓ When model predicted UP, the stock actually went up {result.summary.up_win_rate}% of the time.
              </p>
            ) : (
              <p className="interpretation-ok">
                ~ UP predictions were correct only {result.summary.up_win_rate}% of the time.
              </p>
            )}

            {result.summary.avg_return_when_up > 0 ? (
              <p className="interpretation-good">
                ✓ Average return of +{result.summary.avg_return_when_up}% per trade when following UP calls.
              </p>
            ) : (
              <p className="interpretation-poor">
                ✗ Following UP predictions resulted in avg loss of {result.summary.avg_return_when_up}% per trade.
              </p>
            )}

            {result.summary.avg_return_avoided < 0 && (
              <p className="interpretation-good">
                ✓ Avoided avg loss of {result.summary.avg_return_avoided}% by staying out on DOWN predictions.
              </p>
            )}

            <p className="interpretation-note">
              Note: Uses non-overlapping {result.days}-day periods. No shorting assumed.
            </p>
          </div>

          <div className="recent-predictions">
            <h4>Recent Test Predictions</h4>
            <div className="predictions-table">
              <table>
                <thead>
                  <tr>
                    <th>Predicted</th>
                    <th>Actual</th>
                    <th>Result</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {result.predictions.slice(-10).map((pred, idx) => (
                    <tr key={idx}>
                      <td className={pred.predicted_direction === 'up' ? 'change-up' : 'change-down'}>
                        {pred.predicted_direction.toUpperCase()} ({pred.predicted_return.toFixed(1)}%)
                      </td>
                      <td className={pred.actual_direction === 'up' ? 'change-up' : 'change-down'}>
                        {pred.actual_direction.toUpperCase()} ({pred.actual_return.toFixed(1)}%)
                      </td>
                      <td>
                        {pred.direction_correct ? (
                          <span className="result-correct">✓</span>
                        ) : (
                          <span className="result-wrong">✗</span>
                        )}
                      </td>
                      <td>{(pred.confidence * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Backtest
