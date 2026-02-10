import { useState, useEffect, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

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
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const timerRef = useRef(null)

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

  const runStressTest = async () => {
    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/stress-test`)
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Stress test failed')
      }
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
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

      <button onClick={runStressTest} disabled={loading}>
        {loading && <span className="spinner"></span>}
        {loading ? 'Running Tests...' : 'Run Stress Tests'}
      </button>

      {loading && (
        <div className="loading-text">
          <p>Testing against 4 crisis periods...</p>
          <p className="loading-timer">
            Elapsed: {formatTime(elapsedSeconds)}
          </p>
          <p className="loading-note">This may take 2-4 minutes</p>
        </div>
      )}

      {error && <div className="error">{error}</div>}

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
