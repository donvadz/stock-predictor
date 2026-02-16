import { useState, useEffect } from 'react'
import useJob from '../hooks/useJob'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Grade badge colors
const GRADE_COLORS = {
  A: { bg: '#e8f5e9', border: '#4caf50', text: '#2e7d32' },
  B: { bg: '#e3f2fd', border: '#2196f3', text: '#1565c0' },
  C: { bg: '#fff8e1', border: '#ffc107', text: '#f57f17' },
  D: { bg: '#fff3e0', border: '#ff9800', text: '#e65100' },
  F: { bg: '#ffebee', border: '#f44336', text: '#c62828' },
}

// Score bar colors
const getScoreColor = (score) => {
  if (score >= 70) return '#4caf50'
  if (score >= 50) return '#ffc107'
  if (score >= 35) return '#ff9800'
  return '#f44336'
}

function CompositeRanking() {
  const [expandedRow, setExpandedRow] = useState(null)
  const [sectorFilter, setSectorFilter] = useState('')
  const [gradeFilter, setGradeFilter] = useState('')
  const [horizonMonths, setHorizonMonths] = useState(24)
  const [limit, setLimit] = useState(50)
  const [offset, setOffset] = useState(0)
  const [hasStarted, setHasStarted] = useState(false)

  // Use job-based API for progress tracking
  const {
    isLoading: loading,
    result: data,
    error,
    progress,
    progressMessage,
    elapsedSeconds,
    startJob,
    cancelJob,
    reset,
  } = useJob('composite-ranking')

  const formatTime = (seconds) => {
    if (seconds < 60) return `${seconds}s`
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs}s`
  }

  const handleRun = async () => {
    setHasStarted(true)
    try {
      await startJob('composite-ranking', {
        horizon: horizonMonths,
        sector: sectorFilter || null,
        grade: gradeFilter || null,
        limit,
        offset,
      })
    } catch (err) {
      console.error('Failed to start job:', err)
    }
  }

  const handleCancel = async () => {
    await cancelJob()
    reset()
    setHasStarted(false)
  }

  // When filters change and we have data, re-run the job
  useEffect(() => {
    if (hasStarted && data && !loading) {
      handleRun()
    }
  }, [sectorFilter, gradeFilter, horizonMonths, limit, offset])

  const toggleExpand = (ticker) => {
    setExpandedRow(expandedRow === ticker ? null : ticker)
  }

  const handleSectorChange = (e) => {
    setSectorFilter(e.target.value)
    setOffset(0)
  }

  const handleGradeChange = (e) => {
    setGradeFilter(e.target.value)
    setOffset(0)
  }

  const handleNextPage = () => {
    if (data && offset + limit < data.total_stocks) {
      setOffset(offset + limit)
    }
  }

  const handlePrevPage = () => {
    if (offset > 0) {
      setOffset(Math.max(0, offset - limit))
    }
  }

  const formatPercent = (value) => {
    if (value === null || value === undefined) return '-'
    return `${(value * 100).toFixed(1)}%`
  }

  const formatNumber = (value, decimals = 1) => {
    if (value === null || value === undefined) return '-'
    return value.toFixed(decimals)
  }

  const formatMarketCap = (value) => {
    if (!value) return '-'
    if (value >= 1e12) return `$${(value / 1e12).toFixed(1)}T`
    if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`
    if (value >= 1e6) return `$${(value / 1e6).toFixed(0)}M`
    return `$${value.toLocaleString()}`
  }

  // Estimate bid-ask spread based on market cap and liquidity
  // For Trading 212/commission-free brokers, spread is the main cost
  const estimateSpread = (marketCap) => {
    if (!marketCap) return { spread: 0.10, tier: 'Unknown', color: '#888' }

    if (marketCap >= 200e9) {
      // Mega cap (>$200B): AAPL, MSFT, etc. - very tight spreads
      return { spread: 0.02, tier: 'Mega Cap', color: '#4caf50' }
    } else if (marketCap >= 50e9) {
      // Large cap ($50B-$200B): Still very liquid
      return { spread: 0.03, tier: 'Large Cap', color: '#8bc34a' }
    } else if (marketCap >= 10e9) {
      // Mid-large cap ($10B-$50B): Good liquidity
      return { spread: 0.05, tier: 'Mid-Large', color: '#ffc107' }
    } else if (marketCap >= 2e9) {
      // Mid cap ($2B-$10B): Moderate spreads
      return { spread: 0.10, tier: 'Mid Cap', color: '#ff9800' }
    } else {
      // Small cap (<$2B): Wider spreads
      return { spread: 0.20, tier: 'Small Cap', color: '#f44336' }
    }
  }

  // Calculate round-trip cost (buy + sell)
  const calculateTradingCost = (marketCap, investmentAmount = 1000) => {
    const { spread } = estimateSpread(marketCap)
    // Round trip = buy spread + sell spread
    const roundTripPct = spread * 2
    const costAmount = (investmentAmount * roundTripPct) / 100
    return { roundTripPct, costAmount }
  }

  return (
    <div className="card composite-ranking-card">
      <div className="card-header-badge">
        <span className="badge-long-term">LONG-TERM</span>
      </div>
      <h2>Fundamental Rankings</h2>
      <p className="card-description">
        Data-driven composite scores with adaptive weights based on investment horizon.
        {data?.methodology?.weight_profile && (
          <span className="current-profile"> Currently using: <strong>{data.methodology.weight_profile}</strong></span>
        )}
        {data && <span className="spread-note"> Click any stock to see trading costs (spread estimates for T212).</span>}
      </p>

      {/* Horizon Selector */}
      <div className="horizon-selector">
        <span className="horizon-label">Investment Horizon:</span>
        <div className="horizon-buttons">
          {[
            { value: 1, label: '30D', desc: 'Very short-term' },
            { value: 3, label: '3M', desc: 'Short-term' },
            { value: 6, label: '6M', desc: 'Medium-term' },
            { value: 12, label: '1Y', desc: 'Annual' },
            { value: 24, label: '2Y', desc: 'Standard' },
            { value: 60, label: '5Y', desc: 'Long-term' },
            { value: 120, label: '10Y', desc: 'Full cycle' },
          ].map(({ value, label, desc }) => (
            <button
              key={value}
              className={`horizon-btn ${horizonMonths === value ? 'active' : ''}`}
              onClick={() => { setHorizonMonths(value); setOffset(0); }}
              title={desc}
            >
              {label}
            </button>
          ))}
        </div>
        <span className="horizon-desc">
          {horizonMonths === 1 && '30-day momentum - very short-term swings'}
          {horizonMonths === 3 && '3-month momentum - short-term trends'}
          {horizonMonths === 6 && '6-month momentum - medium-term patterns'}
          {horizonMonths === 12 && '1-year momentum - standard annual view'}
          {horizonMonths === 24 && '2-year momentum - captures corrections'}
          {horizonMonths === 60 && '5-year momentum - full market cycles'}
          {horizonMonths === 120 && '10-year momentum - multiple economic cycles'}
        </span>
      </div>

      {/* Run/Cancel Controls */}
      <div className="ranking-controls">
        {!hasStarted && !loading ? (
          <button onClick={handleRun} className="run-ranking-btn">
            Run Fundamental Rankings
          </button>
        ) : loading ? (
          <button onClick={handleCancel} className="cancel-ranking-btn">
            Cancel ({progress}%)
          </button>
        ) : (
          <div className="ranking-actions">
            <button onClick={handleRun} className="refresh-ranking-btn">
              Refresh
            </button>
            <button onClick={handleCancel} className="cancel-ranking-btn secondary">
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Filters - only show when we have data */}
      {data && (
        <div className="ranking-filters">
          <div className="filter-group">
            <label>Sector</label>
            <select value={sectorFilter} onChange={handleSectorChange}>
              <option value="">All Sectors</option>
              {data?.available_sectors?.map(sector => (
                <option key={sector} value={sector}>{sector}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Grade</label>
            <select value={gradeFilter} onChange={handleGradeChange}>
              <option value="">All Grades</option>
              <option value="A">A (75-100)</option>
              <option value="B">B (60-74)</option>
              <option value="C">C (45-59)</option>
              <option value="D">D (30-44)</option>
              <option value="F">F (0-29)</option>
            </select>
          </div>
        </div>
      )}

      {/* Grade Distribution */}
      {data?.grade_distribution && (
        <div className="grade-distribution">
          {['A', 'B', 'C', 'D', 'F'].map(grade => (
            <div
              key={grade}
              className={`grade-dist-item ${gradeFilter === grade ? 'active' : ''}`}
              onClick={() => setGradeFilter(gradeFilter === grade ? '' : grade)}
              style={{
                background: GRADE_COLORS[grade].bg,
                borderColor: gradeFilter === grade ? GRADE_COLORS[grade].border : 'transparent'
              }}
            >
              <span className="grade-letter" style={{ color: GRADE_COLORS[grade].text }}>
                {grade}
              </span>
              <span className="grade-count">{data.grade_distribution[grade] || 0}</span>
            </div>
          ))}
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {/* Initial state - not started yet */}
      {!hasStarted && !loading && !data && (
        <div className="ranking-initial-state">
          <p>Click "Run Fundamental Rankings" to analyze stocks and generate scores.</p>
          <p className="loading-note">This may take 30-60 seconds on first run to fetch all fundamentals from Yahoo Finance.</p>
        </div>
      )}

      {/* Loading state with progress bar */}
      {loading && (
        <div className="job-progress-container">
          <div className="job-progress-header">
            <span className="job-progress-text">{progressMessage || 'Starting analysis...'}</span>
            <span className="job-progress-percent">{progress}%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-bar-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="progress-stats">
            <span>Elapsed: {formatTime(elapsedSeconds)}</span>
            {progress > 0 && progress < 100 && (
              <span>Est. remaining: ~{formatTime(Math.max(0, Math.round((elapsedSeconds / Math.max(progress, 1)) * (100 - progress))))}</span>
            )}
          </div>
          <p className="loading-note">
            This runs in the background - you can close this tab and return later.
          </p>
        </div>
      )}

      {/* Results Table */}
      {data?.stocks && (
        <>
          <div className="ranking-table-container">
            <table className="ranking-table">
              <thead>
                <tr>
                  <th className="rank-col">#</th>
                  <th className="ticker-col">Ticker</th>
                  <th className="name-col">Name</th>
                  <th className="sector-col">Sector</th>
                  <th className="score-col">Score</th>
                  <th className="grade-col">Grade</th>
                  <th className="breakdown-col">Breakdown</th>
                </tr>
              </thead>
              <tbody>
                {data.stocks.map((stock, idx) => (
                  <>
                    <tr
                      key={stock.ticker}
                      className={`ranking-row ${expandedRow === stock.ticker ? 'expanded' : ''}`}
                      onClick={() => toggleExpand(stock.ticker)}
                    >
                      <td className="rank-col">{stock.rank || offset + idx + 1}</td>
                      <td className="ticker-col">
                        <span className="ticker-symbol">{stock.ticker}</span>
                      </td>
                      <td className="name-col">
                        <span className="stock-name-text">{stock.name}</span>
                      </td>
                      <td className="sector-col">
                        <span className="sector-badge">{stock.sector}</span>
                      </td>
                      <td className="score-col">
                        <div className="score-bar-container">
                          <div
                            className="score-bar-fill"
                            style={{
                              width: `${stock.composite_score}%`,
                              background: getScoreColor(stock.composite_score)
                            }}
                          />
                          <span className="score-value">{stock.composite_score}</span>
                        </div>
                      </td>
                      <td className="grade-col">
                        <span
                          className="grade-badge"
                          style={{
                            background: GRADE_COLORS[stock.grade]?.bg,
                            color: GRADE_COLORS[stock.grade]?.text,
                            borderColor: GRADE_COLORS[stock.grade]?.border
                          }}
                        >
                          {stock.grade}
                        </span>
                      </td>
                      <td className="breakdown-col">
                        <div className="mini-breakdown">
                          <span className="mini-score growth" title="Growth">
                            G: {stock.growth_score?.toFixed(0) || '-'}
                          </span>
                          <span className="mini-score quality" title="Quality">
                            Q: {stock.quality_score?.toFixed(0) || '-'}
                          </span>
                          <span className="mini-score financial" title="Financial">
                            F: {stock.financial_strength_score?.toFixed(0) || '-'}
                          </span>
                          <span className="mini-score valuation" title="Valuation">
                            V: {stock.valuation_score?.toFixed(0) || '-'}
                          </span>
                        </div>
                      </td>
                    </tr>

                    {/* Expanded Row Details */}
                    {expandedRow === stock.ticker && (
                      <tr className="expanded-details-row">
                        <td colSpan="7">
                          <div className="expanded-content">
                            <div className="expanded-sections">
                              {/* Score Breakdown */}
                              <div className="expanded-section">
                                <h4>Score Breakdown</h4>
                                <div className="score-breakdown-grid">
                                  <div className="breakdown-item">
                                    <div className="breakdown-header">
                                      <span className="breakdown-label">Growth</span>
                                      <span className="breakdown-weight">30%</span>
                                    </div>
                                    <div className="breakdown-bar">
                                      <div
                                        className="breakdown-bar-fill"
                                        style={{
                                          width: `${stock.growth_score || 0}%`,
                                          background: getScoreColor(stock.growth_score)
                                        }}
                                      />
                                    </div>
                                    <span className="breakdown-value">{formatNumber(stock.growth_score)}</span>
                                  </div>

                                  <div className="breakdown-item">
                                    <div className="breakdown-header">
                                      <span className="breakdown-label">Quality</span>
                                      <span className="breakdown-weight">30%</span>
                                    </div>
                                    <div className="breakdown-bar">
                                      <div
                                        className="breakdown-bar-fill"
                                        style={{
                                          width: `${stock.quality_score || 0}%`,
                                          background: getScoreColor(stock.quality_score)
                                        }}
                                      />
                                    </div>
                                    <span className="breakdown-value">{formatNumber(stock.quality_score)}</span>
                                  </div>

                                  <div className="breakdown-item">
                                    <div className="breakdown-header">
                                      <span className="breakdown-label">Financial Strength</span>
                                      <span className="breakdown-weight">20%</span>
                                    </div>
                                    <div className="breakdown-bar">
                                      <div
                                        className="breakdown-bar-fill"
                                        style={{
                                          width: `${stock.financial_strength_score || 0}%`,
                                          background: getScoreColor(stock.financial_strength_score)
                                        }}
                                      />
                                    </div>
                                    <span className="breakdown-value">{formatNumber(stock.financial_strength_score)}</span>
                                  </div>

                                  <div className="breakdown-item">
                                    <div className="breakdown-header">
                                      <span className="breakdown-label">Valuation</span>
                                      <span className="breakdown-weight">20%</span>
                                    </div>
                                    <div className="breakdown-bar">
                                      <div
                                        className="breakdown-bar-fill"
                                        style={{
                                          width: `${stock.valuation_score || 0}%`,
                                          background: getScoreColor(stock.valuation_score)
                                        }}
                                      />
                                    </div>
                                    <span className="breakdown-value">{formatNumber(stock.valuation_score)}</span>
                                  </div>
                                </div>
                              </div>

                              {/* Raw Metrics */}
                              <div className="expanded-section">
                                <h4>Key Metrics</h4>
                                <div className="metrics-grid">
                                  <div className="metric-item">
                                    <span className="metric-label">Market Cap</span>
                                    <span className="metric-value">{formatMarketCap(stock.raw_fundamentals?.market_cap)}</span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">P/E Ratio</span>
                                    <span className="metric-value">{formatNumber(stock.raw_fundamentals?.pe_ratio)}</span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">PEG Ratio</span>
                                    <span className="metric-value">{formatNumber(stock.raw_fundamentals?.peg_ratio, 2)}</span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">P/B Ratio</span>
                                    <span className="metric-value">{formatNumber(stock.raw_fundamentals?.price_to_book, 2)}</span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">Revenue Growth</span>
                                    <span className={`metric-value ${stock.raw_fundamentals?.revenue_growth > 0 ? 'positive' : 'negative'}`}>
                                      {formatPercent(stock.raw_fundamentals?.revenue_growth)}
                                    </span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">Earnings Growth</span>
                                    <span className={`metric-value ${stock.raw_fundamentals?.earnings_growth > 0 ? 'positive' : 'negative'}`}>
                                      {formatPercent(stock.raw_fundamentals?.earnings_growth)}
                                    </span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">ROE</span>
                                    <span className="metric-value">{formatPercent(stock.raw_fundamentals?.roe)}</span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">ROA</span>
                                    <span className="metric-value">{formatPercent(stock.raw_fundamentals?.roa)}</span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">Profit Margin</span>
                                    <span className="metric-value">{formatPercent(stock.raw_fundamentals?.profit_margin)}</span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">Operating Margin</span>
                                    <span className="metric-value">{formatPercent(stock.raw_fundamentals?.operating_margin)}</span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">Debt/Equity</span>
                                    <span className="metric-value">{formatNumber(stock.raw_fundamentals?.debt_to_equity)}</span>
                                  </div>
                                  <div className="metric-item">
                                    <span className="metric-label">Current Ratio</span>
                                    <span className="metric-value">{formatNumber(stock.raw_fundamentals?.current_ratio, 2)}</span>
                                  </div>
                                </div>
                              </div>

                              {/* Trading Costs - For retail investors */}
                              <div className="expanded-section trading-costs-section">
                                <h4>Trading Costs (Trading 212)</h4>
                                {(() => {
                                  const spreadInfo = estimateSpread(stock.raw_fundamentals?.market_cap)
                                  const costs = calculateTradingCost(stock.raw_fundamentals?.market_cap, 1000)
                                  return (
                                    <div className="trading-costs-grid">
                                      <div className="cost-info-item">
                                        <span className="cost-label">Liquidity Tier</span>
                                        <span className="cost-value" style={{ color: spreadInfo.color }}>
                                          {spreadInfo.tier}
                                        </span>
                                      </div>
                                      <div className="cost-info-item">
                                        <span className="cost-label">Est. Spread</span>
                                        <span className="cost-value">{spreadInfo.spread.toFixed(2)}%</span>
                                      </div>
                                      <div className="cost-info-item">
                                        <span className="cost-label">Round-Trip Cost</span>
                                        <span className="cost-value negative">{costs.roundTripPct.toFixed(2)}%</span>
                                      </div>
                                      <div className="cost-info-item">
                                        <span className="cost-label">Cost on £1,000</span>
                                        <span className="cost-value negative">£{costs.costAmount.toFixed(2)}</span>
                                      </div>
                                      <div className="cost-note">
                                        <p>Commission: £0 (T212 free trades)</p>
                                        <p>FX fee: 0.15% for USD stocks (not included above)</p>
                                      </div>
                                    </div>
                                  )
                                })()}
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="ranking-pagination">
            <button
              onClick={handlePrevPage}
              disabled={offset === 0}
              className="pagination-button"
            >
              Previous
            </button>
            <span className="pagination-info">
              Showing {offset + 1}-{Math.min(offset + limit, data.total_stocks)} of {data.total_stocks}
            </span>
            <button
              onClick={handleNextPage}
              disabled={offset + limit >= data.total_stocks}
              className="pagination-button"
            >
              Next
            </button>
          </div>
        </>
      )}

      {/* Methodology - Dynamic based on horizon */}
      <div className="methodology-box">
        <h4>Scoring Methodology</h4>
        {data?.methodology ? (
          <>
            <div className="weight-profile">
              <span className="profile-badge">{data.methodology.weight_profile}</span>
              <span className="profile-desc">{data.methodology.description}</span>
            </div>
            <ul>
              <li><strong>Growth ({data.methodology.growth_weight}):</strong> Revenue growth, earnings growth, price momentum</li>
              <li><strong>Quality ({data.methodology.quality_weight}):</strong> ROE (15-25% ideal), ROA, profit margin, operating margin</li>
              <li><strong>Financial Strength ({data.methodology.financial_strength_weight}):</strong> Low debt-to-equity, healthy current ratio</li>
              <li><strong>Valuation ({data.methodology.valuation_weight}):</strong> P/E vs sector median, PEG ratio, price-to-book</li>
            </ul>
            {data.methodology.expected_returns && (
              <div className="expected-returns-summary">
                <h5>Expected Annual Returns by Grade</h5>
                <div className="returns-grid">
                  {Object.entries(data.methodology.expected_returns).map(([grade, info]) => (
                    <div key={grade} className={`return-item grade-${grade}`}>
                      <span className="grade-label">{grade}</span>
                      <span className="return-value">{info.annual}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <>
            <div className="weight-profile">
              <span className="profile-badge">Adaptive Weights</span>
              <span className="profile-desc">Weights adjust based on selected investment horizon</span>
            </div>
            <ul>
              <li><strong>Growth (25-40%):</strong> Revenue growth, earnings growth, price momentum</li>
              <li><strong>Quality (20-35%):</strong> ROE (15-25% ideal), ROA, profit margin, operating margin</li>
              <li><strong>Financial Strength (15-30%):</strong> Low debt-to-equity, healthy current ratio</li>
              <li><strong>Valuation (15-25%):</strong> P/E vs sector median, PEG ratio, price-to-book</li>
            </ul>
            <p className="adaptive-note">
              <em>Run rankings to see exact weights for your selected horizon</em>
            </p>
          </>
        )}
        <p className="methodology-note">
          Scores use absolute thresholds based on empirically-derived quality standards.
          Higher scores indicate objectively stronger fundamentals - same stock scores consistently over time.
        </p>
      </div>

      {/* Backtest Section */}
      <CompositeBacktest />
    </div>
  )
}

function CompositeBacktest() {
  const [showBacktest, setShowBacktest] = useState(false)
  const [backtestType, setBacktestType] = useState('basic') // 'basic' or 'grade-validation'
  const [backtestData, setBacktestData] = useState(null)
  const [basicLoading, setBasicLoading] = useState(false)
  const [basicError, setBasicError] = useState(null)
  const [returnPeriod, setReturnPeriod] = useState('90d')

  // Grade validation state
  const [gradeValidationHorizon, setGradeValidationHorizon] = useState(12)
  const [gradeValidationStocks, setGradeValidationStocks] = useState(200)

  // Use job-based API for grade validation (long-running)
  const {
    isLoading: gradeValidationLoading,
    result: gradeValidationData,
    error: gradeValidationError,
    progress: gradeValidationProgress,
    progressMessage: gradeValidationProgressMessage,
    elapsedSeconds: gradeValidationElapsed,
    startJob: startGradeValidationJob,
    cancelJob: cancelGradeValidationJob,
    reset: resetGradeValidation,
  } = useJob('grade-validation')

  const formatTime = (seconds) => {
    if (seconds < 60) return `${seconds}s`
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs}s`
  }

  const runBacktest = async () => {
    setBasicLoading(true)
    setBasicError(null)

    try {
      const response = await fetch(
        `${API_URL}/composite-backtest?return_period=${returnPeriod}&stocks_count=200`
      )
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const result = await response.json()
      setBacktestData(result)
    } catch (err) {
      setBasicError(err.message)
    } finally {
      setBasicLoading(false)
    }
  }

  const runGradeValidation = async () => {
    try {
      await startGradeValidationJob('grade-validation', {
        horizon_months: gradeValidationHorizon,
        stocks_count: gradeValidationStocks,
      })
    } catch (err) {
      console.error('Failed to start grade validation:', err)
    }
  }

  const handleCancelGradeValidation = async () => {
    await cancelGradeValidationJob()
    resetGradeValidation()
  }

  // Clear grade validation results when parameters change
  useEffect(() => {
    if (gradeValidationData && !gradeValidationLoading) {
      resetGradeValidation()
      localStorage.removeItem('job-grade-validation')
    }
  }, [gradeValidationHorizon, gradeValidationStocks])

  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'VALIDATED': return '#2e7d32'
      case 'STRONG': return '#2e7d32'
      case 'STRONG ALPHA': return '#2e7d32'
      case 'STRONG PICKER': return '#2e7d32'
      case 'EXCELLENT': return '#2e7d32'
      case 'POSITIVE ALPHA': return '#388e3c'
      case 'GOOD': return '#388e3c'
      case 'GOOD PICKER': return '#388e3c'
      case 'PARTIALLY VALIDATED': return '#1565c0'
      case 'PROMISING': return '#1565c0'
      case 'FAIR': return '#1565c0'
      case 'MODERATE': return '#f57f17'
      case 'MARGINAL': return '#f57f17'
      case 'MARGINAL ALPHA': return '#f57f17'
      case 'WEAK': return '#e65100'
      case 'NEEDS WORK': return '#e65100'
      case 'NEEDS IMPROVEMENT': return '#c62828'
      case 'NOT VALIDATED': return '#c62828'
      case 'HIGHLY SIGNIFICANT': return '#2e7d32'
      case 'SIGNIFICANT': return '#388e3c'
      case 'MARGINALLY SIGNIFICANT': return '#f57f17'
      default: return '#888'  // Gray for unknown
    }
  }

  return (
    <div className="backtest-section">
      <button
        className="backtest-toggle"
        onClick={() => setShowBacktest(!showBacktest)}
      >
        {showBacktest ? 'Hide Backtest' : 'Validate Scoring System'}
      </button>

      {showBacktest && (
        <div className="backtest-panel">
          <h4>Composite Score Backtest</h4>
          <p className="backtest-description">
            Test whether higher composite scores actually predict better stock returns.
          </p>

          {/* Backtest Type Tabs */}
          <div className="backtest-type-tabs">
            <button
              className={`backtest-type-tab ${backtestType === 'basic' ? 'active' : ''}`}
              onClick={() => setBacktestType('basic')}
            >
              Basic Analysis
            </button>
            <button
              className={`backtest-type-tab ${backtestType === 'grade-validation' ? 'active' : ''}`}
              onClick={() => setBacktestType('grade-validation')}
            >
              Grade Validation
            </button>
          </div>

          {/* Basic Backtest */}
          {backtestType === 'basic' && (
            <>
              <div className="backtest-type-description">
                <p>Compare returns across score quintiles and grades. Quick analysis with some look-ahead bias.</p>
              </div>

              <div className="backtest-controls">
                <div className="filter-group">
                  <label>Return Period</label>
                  <select
                    value={returnPeriod}
                    onChange={(e) => setReturnPeriod(e.target.value)}
                  >
                    <option value="30d">30 Days</option>
                    <option value="90d">90 Days</option>
                    <option value="180d">180 Days</option>
                    <option value="365d">1 Year</option>
                  </select>
                </div>

                <button
                  onClick={runBacktest}
                  disabled={basicLoading}
                  className="run-backtest-btn"
                >
                  {basicLoading ? (
                    <>
                      <span className="spinner"></span>
                      Running...
                    </>
                  ) : (
                    'Run Basic Backtest'
                  )}
                </button>
              </div>
            </>
          )}

          {/* Basic backtest loading */}
          {basicLoading && (
            <div className="loading-text">
              <span className="spinner"></span>
              <p className="loading-note">
                {`Analyzing ${returnPeriod} returns for 200 stocks...`}
              </p>
            </div>
          )}

          {basicError && (
            <div className="error">Backtest failed: {basicError}</div>
          )}

          {/* Basic Backtest Results */}
          {backtestType === 'basic' && backtestData && !basicLoading && (
            <div className="backtest-results">
              {/* Verdict Banner */}
              <div
                className="backtest-verdict"
                style={{
                  borderColor: getVerdictColor(backtestData.overall_verdict),
                  background: `${getVerdictColor(backtestData.overall_verdict)}15`
                }}
              >
                <span className="verdict-label">Result:</span>
                <span
                  className="verdict-value"
                  style={{ color: getVerdictColor(backtestData.overall_verdict) }}
                >
                  {backtestData.overall_verdict}
                </span>
                <span className="verdict-detail">{backtestData.verdict_detail}</span>
              </div>

              {/* Key Findings */}
              <div className="key-findings">
                <h5>Key Findings</h5>
                <ul>
                  {backtestData.key_findings?.map((finding, i) => (
                    <li key={i}>{finding}</li>
                  ))}
                </ul>
              </div>

              {/* Quintile Analysis */}
              {backtestData.quintile_analysis?.quintiles && (
                <div className="quintile-section">
                  <h5>Quintile Analysis</h5>
                  <p className="section-note">
                    Stocks divided into 5 groups by composite score
                  </p>
                  <div className="quintile-table-container">
                    <table className="quintile-table">
                      <thead>
                        <tr>
                          <th>Quintile</th>
                          <th>Avg Score</th>
                          <th>Avg Return</th>
                          <th>Win Rate</th>
                          <th>Sharpe</th>
                        </tr>
                      </thead>
                      <tbody>
                        {backtestData.quintile_analysis.quintiles.map((q) => (
                          <tr key={q.quintile} className={q.quintile === 1 ? 'top-quintile' : q.quintile === 5 ? 'bottom-quintile' : ''}>
                            <td>
                              <span className="quintile-label">{q.label}</span>
                            </td>
                            <td>{q.avg_composite_score}</td>
                            <td className={q.avg_return >= 0 ? 'positive' : 'negative'}>
                              {q.avg_return >= 0 ? '+' : ''}{q.avg_return}%
                            </td>
                            <td>{q.win_rate}%</td>
                            <td>{q.sharpe_ratio || '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="quintile-spread">
                    <span className="spread-label">Top vs Bottom Spread:</span>
                    <span
                      className={`spread-value ${backtestData.quintile_analysis.summary?.spread >= 0 ? 'positive' : 'negative'}`}
                    >
                      {backtestData.quintile_analysis.summary?.spread >= 0 ? '+' : ''}
                      {backtestData.quintile_analysis.summary?.spread}%
                    </span>
                  </div>
                </div>
              )}

              {/* Factor Analysis */}
              {backtestData.factor_analysis?.factors && (
                <div className="factor-section">
                  <h5>Factor Performance</h5>
                  <p className="section-note">
                    Which factors best predict returns
                  </p>
                  <div className="factor-grid">
                    {backtestData.factor_analysis.factors.map((f) => (
                      <div key={f.factor_key} className={`factor-card ${f.predictive_power.toLowerCase()}`}>
                        <div className="factor-name">{f.factor}</div>
                        <div className="factor-spread">
                          <span className={f.spread >= 0 ? 'positive' : 'negative'}>
                            {f.spread >= 0 ? '+' : ''}{f.spread}%
                          </span>
                        </div>
                        <div className="factor-power">{f.predictive_power}</div>
                        <div className="factor-correlation">r = {f.correlation}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Grade Analysis */}
              {backtestData.grade_analysis?.grades && (
                <div className="grade-section">
                  <h5>Performance by Grade</h5>
                  <div className="grade-bars">
                    {backtestData.grade_analysis.grades.map((g) => (
                      <div key={g.grade} className="grade-bar-row">
                        <span
                          className="grade-badge-small"
                          style={{
                            background: GRADE_COLORS[g.grade]?.bg,
                            color: GRADE_COLORS[g.grade]?.text,
                          }}
                        >
                          {g.grade}
                        </span>
                        <div className="grade-bar-container">
                          <div
                            className="grade-bar-fill"
                            style={{
                              width: `${Math.min(100, Math.max(0, (g.avg_return + 30) * 1.5))}%`,
                              background: g.avg_return >= 0 ? '#4caf50' : '#f44336'
                            }}
                          />
                        </div>
                        <span className={`grade-return ${g.avg_return >= 0 ? 'positive' : 'negative'}`}>
                          {g.avg_return >= 0 ? '+' : ''}{g.avg_return}%
                        </span>
                        <span className="grade-win-rate">({g.win_rate}% win)</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Caveats */}
              <div className="backtest-caveats">
                <h5>Important Caveats</h5>
                <ul>
                  {backtestData.caveats?.map((caveat, i) => (
                    <li key={i}>{caveat}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Grade Validation Test */}
          {backtestType === 'grade-validation' && (
            <>
              <div className="backtest-type-description">
                <p>
                  <strong>True test of the grading system:</strong> Goes back 10, 5, 2, and 1 years ago, calculates grades
                  using only data available at each point, then tracks if A/B grade stocks actually delivered expected returns.
                </p>
              </div>

              <div className="backtest-controls">
                <div className="filter-group">
                  <label>Holding Period</label>
                  <select
                    value={gradeValidationHorizon}
                    onChange={(e) => setGradeValidationHorizon(Number(e.target.value))}
                  >
                    <option value={3}>3 Months (Quarterly)</option>
                    <option value={6}>6 Months (Semi-Annual)</option>
                    <option value={12}>12 Months (Annual)</option>
                    <option value={24}>2 Years</option>
                    <option value={60}>5 Years</option>
                    <option value={120}>10 Years</option>
                  </select>
                </div>
                <div className="filter-group">
                  <label>Universe Size</label>
                  <select
                    value={gradeValidationStocks}
                    onChange={(e) => setGradeValidationStocks(Number(e.target.value))}
                  >
                    <option value={100}>100 stocks (faster)</option>
                    <option value={200}>200 stocks (balanced)</option>
                    <option value={300}>300 stocks</option>
                    <option value={500}>500 stocks</option>
                    <option value={679}>Full S&P 500 (679 stocks)</option>
                  </select>
                </div>
                {gradeValidationLoading ? (
                  <button
                    onClick={handleCancelGradeValidation}
                    className="cancel-backtest-btn"
                  >
                    Cancel ({gradeValidationProgress}%)
                  </button>
                ) : (
                  <button
                    onClick={runGradeValidation}
                    className="run-backtest-btn"
                  >
                    Run Grade Validation
                  </button>
                )}
              </div>

              {/* Progress indicator */}
              {gradeValidationLoading && (
                <div className="job-progress-container" style={{ marginTop: '16px' }}>
                  <div className="job-progress-header">
                    <span className="job-progress-text">{gradeValidationProgressMessage || 'Starting grade validation...'}</span>
                    <span className="job-progress-percent">{gradeValidationProgress}%</span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-bar-fill"
                      style={{ width: `${gradeValidationProgress}%` }}
                    />
                  </div>
                  <div className="progress-stats">
                    <span>Elapsed: {formatTime(gradeValidationElapsed)}</span>
                    {gradeValidationProgress > 0 && gradeValidationProgress < 100 && (
                      <span>Est. remaining: ~{formatTime(Math.max(0, Math.round((gradeValidationElapsed / Math.max(gradeValidationProgress, 1)) * (100 - gradeValidationProgress))))}</span>
                    )}
                  </div>
                  <p className="loading-note">
                    This runs in the background - you can close this tab and return later.
                  </p>
                </div>
              )}

              {!gradeValidationLoading && !gradeValidationData && (
                <p className="loading-note" style={{ marginTop: '8px' }}>
                  Estimated time: ~{
                    // Base time by stocks
                    (gradeValidationStocks <= 100 ? 2 : gradeValidationStocks <= 200 ? 4 : gradeValidationStocks <= 300 ? 6 : gradeValidationStocks <= 500 ? 10 : 15) *
                    // Multiplier for longer horizons (need more historical data)
                    (gradeValidationHorizon <= 24 ? 1 : gradeValidationHorizon <= 60 ? 1.5 : 2)
                  } minutes for {gradeValidationStocks} stocks over {
                    gradeValidationHorizon === 3 ? '3 months' :
                    gradeValidationHorizon === 6 ? '6 months' :
                    gradeValidationHorizon === 12 ? '1 year' :
                    gradeValidationHorizon === 24 ? '2 years' :
                    gradeValidationHorizon === 60 ? '5 years' : '10 years'
                  }.
                </p>
              )}

              {gradeValidationError && (
                <div className="error-message">Error: {gradeValidationError}</div>
              )}
            </>
          )}

          {/* Grade Validation Results */}
          {backtestType === 'grade-validation' && gradeValidationData && !gradeValidationLoading && (
            <div className="rigorous-results">
              {/* Overall Verdict */}
              <div
                className="overall-verdict"
                style={{
                  borderColor: getVerdictColor(gradeValidationData.verdict),
                  background: `${getVerdictColor(gradeValidationData.verdict)}15`
                }}
              >
                <h4>Grade Validation Result</h4>
                <div className="verdict-main">
                  <span
                    className="verdict-badge"
                    style={{
                      background: getVerdictColor(gradeValidationData.verdict),
                      color: '#fff'
                    }}
                  >
                    {gradeValidationData.verdict}
                  </span>
                  <p className="verdict-detail">{gradeValidationData.verdict_detail}</p>
                </div>
              </div>

              {/* Summary Stats */}
              <div className="grade-validation-summary">
                <h5>Grade Success Rates</h5>
                <p className="section-note">
                  "Did stocks graded A/B historically deliver their expected returns?"
                </p>
                <div className="grade-success-stats">
                  <div className="grade-stat">
                    <span className="grade-label">A Grade Success</span>
                    <span className={`grade-value ${gradeValidationData.summary?.avg_a_grade_success_rate >= 60 ? 'positive' : gradeValidationData.summary?.avg_a_grade_success_rate >= 50 ? '' : 'negative'}`}>
                      {gradeValidationData.summary?.avg_a_grade_success_rate || 0}%
                    </span>
                  </div>
                  <div className="grade-stat">
                    <span className="grade-label">B Grade Success</span>
                    <span className={`grade-value ${gradeValidationData.summary?.avg_b_grade_success_rate >= 60 ? 'positive' : gradeValidationData.summary?.avg_b_grade_success_rate >= 50 ? '' : 'negative'}`}>
                      {gradeValidationData.summary?.avg_b_grade_success_rate || 0}%
                    </span>
                  </div>
                  <div className="grade-stat primary">
                    <span className="grade-label">A+B Combined</span>
                    <span className={`grade-value ${gradeValidationData.summary?.avg_ab_combined_success_rate >= 60 ? 'positive' : gradeValidationData.summary?.avg_ab_combined_success_rate >= 50 ? '' : 'negative'}`}>
                      {gradeValidationData.summary?.avg_ab_combined_success_rate || 0}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Expected Returns Reference */}
              <div className="expected-returns-ref">
                <h6>Expected Returns by Grade ({gradeValidationData.horizon_months}M horizon)</h6>
                <div className="expected-grid">
                  {Object.entries(gradeValidationData.expected_returns || {}).map(([grade, pct]) => (
                    <span key={grade} className={`expected-item grade-${grade.toLowerCase()}`}>
                      {grade}: {pct >= 0 ? '+' : ''}{pct}%
                    </span>
                  ))}
                </div>
              </div>

              {/* Factor Analysis */}
              {gradeValidationData.factor_analysis && Object.keys(gradeValidationData.factor_analysis).length > 0 && (
                <div className="factor-analysis-section">
                  <h5>Factor Correlation Analysis</h5>
                  <p className="section-note">
                    Which scoring factors actually predict returns?
                  </p>
                  <div className="factor-grid">
                    {Object.entries(gradeValidationData.factor_analysis).map(([factor, data]) => (
                      <div
                        key={factor}
                        className={`factor-card ${data.predictive_power.toLowerCase().replace(' ', '-')}`}
                      >
                        <div className="factor-name">{factor.replace('_', ' ').toUpperCase()}</div>
                        <div className="factor-correlation">
                          <span className="corr-label">Correlation:</span>
                          <span className={`corr-value ${data.correlation >= 0.1 ? 'positive' : data.correlation <= -0.05 ? 'negative' : ''}`}>
                            r = {data.correlation}
                          </span>
                        </div>
                        <div className="factor-spread">
                          <span className="spread-label">High vs Low Spread:</span>
                          <span className={`spread-value ${data.spread >= 0 ? 'positive' : 'negative'}`}>
                            {data.spread >= 0 ? '+' : ''}{data.spread}%
                          </span>
                        </div>
                        <div className="factor-power">
                          <span
                            className={`power-badge ${data.predictive_power.toLowerCase().replace(' ', '-')}`}
                          >
                            {data.predictive_power}
                          </span>
                        </div>
                        <div className="factor-details">
                          <span>High: {data.high_score_avg_return}%</span>
                          <span>Low: {data.low_score_avg_return}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Period Breakdown */}
              {gradeValidationData.period_results && (
                <div className="period-breakdown">
                  <h5>Results by Test Period</h5>
                  <div className="period-cards">
                    {Object.entries(gradeValidationData.period_results).map(([period, data]) => (
                      <div key={period} className="period-card">
                        <h6>{period} Ago</h6>
                        <div className="period-dates">
                          {data.period_start} to {data.period_end}
                        </div>
                        <div className="period-stats">
                          <div className="ps-row">
                            <span>Test Points:</span>
                            <span>{data.test_points}</span>
                          </div>
                          {data.a_and_b_combined && (
                            <>
                              <div className="ps-row">
                                <span>A+B Picks:</span>
                                <span>{data.a_and_b_combined.count}</span>
                              </div>
                              <div className="ps-row highlight">
                                <span>Success Rate:</span>
                                <span className={data.a_and_b_combined.success_rate >= 60 ? 'positive' : data.a_and_b_combined.success_rate >= 50 ? '' : 'negative'}>
                                  {data.a_and_b_combined.success_rate}%
                                </span>
                              </div>
                              <div className="ps-row">
                                <span>Beat Benchmark:</span>
                                <span className={data.a_and_b_combined.beat_benchmark_rate >= 50 ? 'positive' : 'negative'}>
                                  {data.a_and_b_combined.beat_benchmark_rate}%
                                </span>
                              </div>
                              <div className="ps-row">
                                <span>Avg Return:</span>
                                <span className={data.a_and_b_combined.avg_return >= 0 ? 'positive' : 'negative'}>
                                  {data.a_and_b_combined.avg_return >= 0 ? '+' : ''}{data.a_and_b_combined.avg_return}%
                                </span>
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Interpretation */}
              {gradeValidationData.interpretation && (
                <div className="interpretation-box">
                  <h6>Test Summary</h6>
                  <ul>
                    {gradeValidationData.interpretation.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Methodology */}
              <div className="rigorous-methodology">
                <h5>What This Test Does</h5>
                <ul>
                  <li><strong>Point-in-time scoring:</strong> Uses only data available at each historical date (no look-ahead bias)</li>
                  <li><strong>Walk-forward:</strong> Tests across multiple historical periods (10Y, 5Y, 2Y, 1Y ago)</li>
                  <li><strong>Validation metric:</strong> "Did A/B graded stocks achieve their expected returns?"</li>
                  <li><strong>Benchmark:</strong> Compares against SPY to measure alpha generation</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default CompositeRanking
