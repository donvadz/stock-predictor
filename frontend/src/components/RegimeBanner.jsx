import { useState, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function RegimeBanner() {
  const [regime, setRegime] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchRegime()
  }, [])

  const fetchRegime = async () => {
    try {
      const response = await fetch(`${API_BASE}/regime`)
      if (!response.ok) throw new Error('Failed to fetch regime')
      const data = await response.json()
      setRegime(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="regime-banner regime-loading">
        <span className="regime-banner-text">Loading market regime...</span>
      </div>
    )
  }

  if (error || !regime) {
    return null
  }

  const regimeConfig = {
    NORMAL: {
      icon: '🟢',
      color: 'regime-normal',
      action: 'Strategy Active',
      description: 'Favorable conditions for the strategy'
    },
    CAUTION: {
      icon: '🟡',
      color: 'regime-caution',
      action: 'Reduced Confidence',
      description: 'Elevated risk - use higher thresholds'
    },
    CRISIS: {
      icon: '🔴',
      color: 'regime-crisis',
      action: 'Strategy Paused',
      description: 'Strategy fails during crisis - no new positions'
    },
    RECOVERY: {
      icon: '🟠',
      color: 'regime-recovery',
      action: 'Cautious Entry',
      description: 'Market recovering - proceed carefully'
    },
    EUPHORIA: {
      icon: '🟣',
      color: 'regime-euphoria',
      action: 'Tighten Stops',
      description: 'Market may be overextended'
    }
  }

  const config = regimeConfig[regime.regime] || regimeConfig.NORMAL
  const metrics = regime.metrics || {}

  return (
    <div className={`regime-banner ${config.color}`}>
      <div className="regime-banner-main">
        <span className="regime-banner-icon">{config.icon}</span>
        <div className="regime-banner-info">
          <span className="regime-banner-label">{regime.regime}</span>
          <span className="regime-banner-action">{config.action}</span>
        </div>
      </div>
      <div className="regime-banner-metrics">
        {metrics.spy_price && (
          <span className="regime-metric">
            SPY: ${metrics.spy_price}
          </span>
        )}
        {metrics.drawdown_from_50d_high !== undefined && (
          <span className={`regime-metric ${metrics.drawdown_from_50d_high < -5 ? 'negative' : ''}`}>
            DD: {metrics.drawdown_from_50d_high}%
          </span>
        )}
        {metrics.volatility_annual && (
          <span className={`regime-metric ${metrics.volatility_annual > 25 ? 'warning' : ''}`}>
            Vol: {metrics.volatility_annual}%
          </span>
        )}
        {metrics.vix && (
          <span className={`regime-metric ${metrics.vix > 20 ? 'warning' : ''}`}>
            VIX: {metrics.vix}
          </span>
        )}
      </div>
      <div className="regime-banner-description">
        {config.description}
      </div>
    </div>
  )
}

export default RegimeBanner
