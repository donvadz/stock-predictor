import { useState } from 'react'
import useJob from '../hooks/useJob'

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

const CRISIS_PERIODS = {
  '2008_crisis': { name: '2008 Financial Crisis', description: 'Lehman collapse, -50% drawdown' },
  '2018_selloff': { name: '2018 Q4 Selloff', description: 'Fed tightening, -20% correction' },
  '2020_covid': { name: '2020 COVID Crash', description: 'Fastest bear market in history' },
  '2022_bear': { name: '2022 Bear Market', description: 'Fed hiking, tech collapse' },
}

function StressTest() {
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
  } = useJob('stress-test')

  const runStressTest = async () => {
    try {
      await startJob('stress-test', {})
    } catch (err) {
      console.error('Failed to start stress test:', err)
    }
  }

  const handleCancel = async () => {
    await cancelJob()
  }

  const getAccuracyClass = (accuracy) => {
    if (accuracy >= 60) return 'accuracy-great'
    if (accuracy >= 50) return 'accuracy-ok'
    return 'accuracy-poor'
  }

  return (
    <div className="card stress-test-card">
      <h2>Crisis Stress Test</h2>
      <p className="card-description">
        See how the strategy performs during historical market crises.
        <br />
        <strong>This is why we check market regime before trading.</strong>
      </p>

      <div className="button-row">
        <button onClick={runStressTest} disabled={isLoading}>
          {isLoading && <span className="spinner"></span>}
          {isLoading ? 'Running Tests...' : 'Run Stress Tests'}
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

      {isLoading && (
        <div className="job-progress-container">
          <div className="job-progress-header">
            <span className="job-progress-text">{progressMessage || 'Testing against 4 crisis periods...'}</span>
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
          <p className="loading-note">This may take 2-4 minutes</p>
        </div>
      )}

      {isCancelled && (
        <div className="cancelled-message">
          <span className="cancelled-icon">⚠️</span>
          <span>Stress test was cancelled. Click "Run Stress Tests" to start again.</span>
        </div>
      )}

      {error && !isCancelled && <div className="error">{error}</div>}

      {result && (
        <div className="stress-results">
          {/* Verdict */}
          <div className={`stress-verdict ${result.verdict?.toLowerCase()}`}>
            <span className="verdict-label">Verdict:</span>
            <span className="verdict-value">{result.verdict}</span>
          </div>

          {/* Interpretation */}
          {result.interpretation && (
            <div className="stress-interpretation">
              {result.interpretation.map((point, i) => (
                <p key={i}>{point}</p>
              ))}
            </div>
          )}

          {/* Period Results */}
          <div className="stress-periods">
            <h4>Performance by Crisis Period</h4>
            <div className="periods-grid">
              {Object.entries(result.all_periods || {}).map(([key, period]) => {
                if (period.error) return null
                const info = CRISIS_PERIODS[key] || { name: key, description: '' }
                const accuracy = period.high_conviction_accuracy || 0
                const survived = accuracy >= 50

                return (
                  <div key={key} className={`period-card ${survived ? 'survived' : 'failed'}`}>
                    <div className="period-status">
                      {survived ? '✅' : '❌'}
                    </div>
                    <div className="period-name">{info.name}</div>
                    <div className="period-desc">{info.description}</div>
                    <div className={`period-accuracy ${getAccuracyClass(accuracy)}`}>
                      {accuracy}%
                    </div>
                    <div className="period-picks">
                      {period.high_conviction_picks || 0} picks
                    </div>
                    {period.max_drawdown !== undefined && (
                      <div className="period-drawdown">
                        Max DD: {period.max_drawdown}%
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Summary */}
          <div className="stress-summary">
            <div className="summary-column survived">
              <h5>Survived</h5>
              {result.summary?.survived?.map((item, i) => (
                <div key={i} className="summary-item">
                  <span>{item.period}</span>
                  <span className="summary-accuracy">{item.accuracy}%</span>
                </div>
              ))}
              {result.summary?.survived?.length === 0 && (
                <div className="summary-empty">None</div>
              )}
            </div>
            <div className="summary-column failed">
              <h5>Failed</h5>
              {result.summary?.failed?.map((item, i) => (
                <div key={i} className="summary-item">
                  <span>{item.period}</span>
                  <span className="summary-accuracy">{item.accuracy}%</span>
                </div>
              ))}
              {result.summary?.failed?.length === 0 && (
                <div className="summary-empty">None</div>
              )}
            </div>
          </div>

          {/* Warning */}
          <div className="stress-warning">
            <strong>Key Takeaway:</strong> The strategy fails during most crisis periods.
            Always check the market regime before trading. In CRISIS or CAUTION mode,
            do not take new positions.
          </div>
        </div>
      )}
    </div>
  )
}

export default StressTest
