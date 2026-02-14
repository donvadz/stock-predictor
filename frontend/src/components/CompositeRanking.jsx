import { useState, useEffect, useRef } from 'react'

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
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedRow, setExpandedRow] = useState(null)
  const [sectorFilter, setSectorFilter] = useState('')
  const [gradeFilter, setGradeFilter] = useState('')
  const [horizonMonths, setHorizonMonths] = useState(24)
  const [limit, setLimit] = useState(50)
  const [offset, setOffset] = useState(0)
  const [hasStarted, setHasStarted] = useState(false)
  const abortControllerRef = useRef(null)

  const fetchData = async () => {
    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    // Create new abort controller
    abortControllerRef.current = new AbortController()

    setLoading(true)
    setError(null)

    try {
      let url = `${API_URL}/composite-scores?limit=${limit}&offset=${offset}&horizon=${horizonMonths}`
      if (sectorFilter) url += `&sector=${encodeURIComponent(sectorFilter)}`
      if (gradeFilter) url += `&grade=${gradeFilter}`

      const response = await fetch(url, {
        signal: abortControllerRef.current.signal
      })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const result = await response.json()
      setData(result)
    } catch (err) {
      if (err.name === 'AbortError') {
        // Request was cancelled, don't set error
        return
      }
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRun = () => {
    setHasStarted(true)
    fetchData()
  }

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setLoading(false)
    setHasStarted(false)
    setData(null)
  }

  // Only auto-fetch when filters/pagination change IF user has already started
  useEffect(() => {
    if (hasStarted && data) {
      fetchData()
    }
  }, [sectorFilter, gradeFilter, horizonMonths, limit, offset])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

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
            <span className="spinner"></span>
            Cancel
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
              <option value="A">A (80-100)</option>
              <option value="B">B (65-79)</option>
              <option value="C">C (50-64)</option>
              <option value="D">D (35-49)</option>
              <option value="F">F (0-34)</option>
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

      {/* Loading state */}
      {loading && (
        <div className="loading-text">
          <span className="spinner"></span>
          <p>Loading rankings...</p>
          <p className="loading-note">Fetching fundamentals from Yahoo Finance... This may take 30-60 seconds.</p>
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
          Scores are percentile-ranked across the entire stock universe.
          Higher scores indicate stronger fundamentals for the selected investment horizon.
        </p>
      </div>

      {/* Backtest Section */}
      <CompositeBacktest />
    </div>
  )
}

function CompositeBacktest() {
  const [showBacktest, setShowBacktest] = useState(false)
  const [backtestType, setBacktestType] = useState('basic') // 'basic' or 'rigorous'
  const [backtestData, setBacktestData] = useState(null)
  const [rigorousData, setRigorousData] = useState(null)
  const [backtestLoading, setBacktestLoading] = useState(false)
  const [backtestError, setBacktestError] = useState(null)
  const [returnPeriod, setReturnPeriod] = useState('90d')
  const [rigorousYears, setRigorousYears] = useState(2)

  const runBacktest = async () => {
    setBacktestLoading(true)
    setBacktestError(null)

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
      setBacktestError(err.message)
    } finally {
      setBacktestLoading(false)
    }
  }

  const runRigorousBacktest = async () => {
    setBacktestLoading(true)
    setBacktestError(null)

    try {
      const response = await fetch(
        `${API_URL}/composite-backtest/rigorous?stocks_count=150&years=${rigorousYears}`
      )
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const result = await response.json()
      setRigorousData(result)
    } catch (err) {
      setBacktestError(err.message)
    } finally {
      setBacktestLoading(false)
    }
  }

  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'VALIDATED': return '#2e7d32'
      case 'STRONG': return '#2e7d32'
      case 'STRONG ALPHA': return '#2e7d32'
      case 'POSITIVE ALPHA': return '#388e3c'
      case 'PROMISING': return '#1565c0'
      case 'MODERATE': return '#f57f17'
      case 'MARGINAL ALPHA': return '#f57f17'
      case 'WEAK': return '#e65100'
      case 'HIGHLY SIGNIFICANT': return '#2e7d32'
      case 'SIGNIFICANT': return '#388e3c'
      case 'MARGINALLY SIGNIFICANT': return '#f57f17'
      default: return '#c62828'
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
              className={`backtest-type-tab ${backtestType === 'rigorous' ? 'active' : ''}`}
              onClick={() => setBacktestType('rigorous')}
            >
              Rigorous Test
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
                  disabled={backtestLoading}
                  className="run-backtest-btn"
                >
                  {backtestLoading ? (
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

          {/* Rigorous Backtest */}
          {backtestType === 'rigorous' && (
            <>
              <div className="backtest-type-description rigorous">
                <h5>Walk-Forward Portfolio Simulation + Statistical Significance Test</h5>
                <ul>
                  <li><strong>Portfolio Simulation:</strong> Simulates buying top 20 stocks, rebalancing quarterly</li>
                  <li><strong>Point-in-Time Scoring:</strong> Uses only data available at each decision point (minimizes look-ahead bias)</li>
                  <li><strong>Benchmark Comparison:</strong> Measures alpha vs SPY buy-and-hold</li>
                  <li><strong>Monte Carlo Test:</strong> Compares to 500 random portfolios for statistical significance</li>
                  <li><strong>Long-Term Focus:</strong> Best for finding quality companies during cautious/crisis market periods</li>
                </ul>
              </div>

              <div className="backtest-controls">
                <div className="period-selector">
                  <span className="selector-label">Simulation Period:</span>
                  <div className="period-buttons">
                    {[
                      { value: 2, label: '2 Years' },
                      { value: 5, label: '5 Years' },
                      { value: 10, label: '10 Years' },
                    ].map(({ value, label }) => (
                      <button
                        key={value}
                        className={`period-btn ${rigorousYears === value ? 'active' : ''}`}
                        onClick={() => setRigorousYears(value)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                <button
                  onClick={runRigorousBacktest}
                  disabled={backtestLoading}
                  className="run-backtest-btn rigorous"
                >
                  {backtestLoading ? (
                    <>
                      <span className="spinner"></span>
                      Running {rigorousYears}Y Rigorous Test...
                    </>
                  ) : (
                    `Run ${rigorousYears}-Year Rigorous Backtest`
                  )}
                </button>
              </div>

              <p className="loading-note" style={{ marginTop: '8px' }}>
                {rigorousYears <= 2
                  ? 'This test takes 2-5 minutes as it fetches historical data and runs simulations.'
                  : rigorousYears <= 5
                  ? 'This test takes 5-10 minutes due to extended historical data requirements.'
                  : 'This test takes 10-15 minutes to analyze a full decade of market data.'
                }
              </p>
            </>
          )}

          {backtestLoading && (
            <div className="loading-text">
              <p className="loading-note">
                {backtestType === 'rigorous'
                  ? `Running ${rigorousYears}-year portfolio simulation and Monte Carlo analysis...`
                  : `Analyzing ${returnPeriod} returns for 200 stocks...`
                }
              </p>
            </div>
          )}

          {backtestError && (
            <div className="error">Backtest failed: {backtestError}</div>
          )}

          {/* Basic Backtest Results */}
          {backtestType === 'basic' && backtestData && !backtestLoading && (
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

          {/* Rigorous Backtest Results */}
          {backtestType === 'rigorous' && rigorousData && !backtestLoading && (
            <div className="backtest-results rigorous-results">
              {/* Simulation Period Banner */}
              <div className="simulation-period-banner">
                <span className="period-label">Simulation Period:</span>
                <span className="period-value">{rigorousData.simulation_years || 2} Years</span>
                <span className="period-dates">({rigorousData.portfolio_simulation?.period})</span>
              </div>

              {/* Overall Verdict */}
              <div
                className="backtest-verdict"
                style={{
                  borderColor: getVerdictColor(rigorousData.overall_verdict),
                  background: `${getVerdictColor(rigorousData.overall_verdict)}15`
                }}
              >
                <span className="verdict-label">Overall Result:</span>
                <span
                  className="verdict-value"
                  style={{ color: getVerdictColor(rigorousData.overall_verdict) }}
                >
                  {rigorousData.overall_verdict}
                </span>
                <span className="verdict-detail">{rigorousData.overall_detail}</span>
              </div>

              {/* Key Metrics Summary */}
              <div className="rigorous-summary">
                <div className="summary-card">
                  <span className="summary-label">Alpha vs SPY</span>
                  <span className={`summary-value ${rigorousData.key_metrics?.alpha_vs_spy >= 0 ? 'positive' : 'negative'}`}>
                    {rigorousData.key_metrics?.alpha_vs_spy >= 0 ? '+' : ''}
                    {rigorousData.key_metrics?.alpha_vs_spy}%
                  </span>
                </div>
                <div className="summary-card">
                  <span className="summary-label">Sharpe Ratio</span>
                  <span className="summary-value">
                    {rigorousData.key_metrics?.sharpe_ratio}
                  </span>
                </div>
                <div className="summary-card">
                  <span className="summary-label">Statistical Significance</span>
                  <span
                    className="summary-value"
                    style={{ color: getVerdictColor(rigorousData.key_metrics?.statistical_significance) }}
                  >
                    {rigorousData.key_metrics?.statistical_significance}
                  </span>
                </div>
                <div className="summary-card">
                  <span className="summary-label">P-Value</span>
                  <span className={`summary-value ${rigorousData.key_metrics?.p_value < 0.05 ? 'positive' : ''}`}>
                    {rigorousData.key_metrics?.p_value}
                  </span>
                </div>
              </div>

              {/* Portfolio Simulation Results */}
              {rigorousData.portfolio_simulation && (
                <div className="portfolio-section">
                  <h5>Portfolio Simulation</h5>
                  <p className="section-note">{rigorousData.portfolio_simulation.period}</p>

                  <div className="portfolio-metrics">
                    <div className="metric-row">
                      <span className="metric-name">Initial Capital</span>
                      <span className="metric-val">
                        ${rigorousData.portfolio_simulation.performance?.initial_capital?.toLocaleString()}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-name">Final Portfolio Value</span>
                      <span className="metric-val">
                        ${rigorousData.portfolio_simulation.performance?.final_portfolio_value?.toLocaleString()}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-name">Final Benchmark (SPY)</span>
                      <span className="metric-val">
                        ${rigorousData.portfolio_simulation.performance?.final_benchmark_value?.toLocaleString()}
                      </span>
                    </div>
                    <div className="metric-row highlight">
                      <span className="metric-name">Total Return</span>
                      <span className={`metric-val ${rigorousData.portfolio_simulation.performance?.total_return >= 0 ? 'positive' : 'negative'}`}>
                        {rigorousData.portfolio_simulation.performance?.total_return >= 0 ? '+' : ''}
                        {rigorousData.portfolio_simulation.performance?.total_return}%
                      </span>
                    </div>
                    <div className="metric-row highlight">
                      <span className="metric-name">Benchmark Return</span>
                      <span className={`metric-val ${rigorousData.portfolio_simulation.performance?.benchmark_return >= 0 ? 'positive' : 'negative'}`}>
                        {rigorousData.portfolio_simulation.performance?.benchmark_return >= 0 ? '+' : ''}
                        {rigorousData.portfolio_simulation.performance?.benchmark_return}%
                      </span>
                    </div>
                    <div className="metric-row primary">
                      <span className="metric-name">Alpha (Outperformance)</span>
                      <span className={`metric-val ${rigorousData.portfolio_simulation.performance?.alpha >= 0 ? 'positive' : 'negative'}`}>
                        {rigorousData.portfolio_simulation.performance?.alpha >= 0 ? '+' : ''}
                        {rigorousData.portfolio_simulation.performance?.alpha}%
                      </span>
                    </div>
                  </div>

                  <div className="risk-metrics">
                    <h6>Risk Metrics</h6>
                    <div className="risk-row">
                      <span>Win Rate:</span>
                      <span>{rigorousData.portfolio_simulation.risk_metrics?.win_rate}%</span>
                    </div>
                    <div className="risk-row">
                      <span>Max Drawdown:</span>
                      <span className="negative">-{rigorousData.portfolio_simulation.risk_metrics?.max_drawdown}%</span>
                    </div>
                    <div className="risk-row">
                      <span>Sharpe Ratio:</span>
                      <span>{rigorousData.portfolio_simulation.risk_metrics?.sharpe_ratio}</span>
                    </div>
                  </div>

                  {/* Equity Curve */}
                  {rigorousData.portfolio_simulation.equity_curve && (
                    <div className="equity-curve">
                      <h6>Equity Curve</h6>
                      <div className="equity-table">
                        <div className="equity-header">
                          <span>Date</span>
                          <span>Portfolio</span>
                          <span>Benchmark</span>
                        </div>
                        {rigorousData.portfolio_simulation.equity_curve.slice(-6).map((point, i) => (
                          <div key={i} className="equity-row">
                            <span>{point.date}</span>
                            <span>${point.portfolio?.toLocaleString()}</span>
                            <span>${point.benchmark?.toLocaleString()}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Monte Carlo Results */}
              {rigorousData.statistical_significance && (
                <div className="monte-carlo-section">
                  <h5>Statistical Significance Test</h5>
                  <p className="section-note">
                    Compared to {rigorousData.statistical_significance.num_simulations} randomly selected portfolios
                  </p>

                  <div
                    className="significance-verdict"
                    style={{
                      borderColor: getVerdictColor(rigorousData.statistical_significance.significance),
                      background: `${getVerdictColor(rigorousData.statistical_significance.significance)}15`
                    }}
                  >
                    <span className="sig-label">Result:</span>
                    <span
                      className="sig-value"
                      style={{ color: getVerdictColor(rigorousData.statistical_significance.significance) }}
                    >
                      {rigorousData.statistical_significance.significance}
                    </span>
                    <span className="sig-detail">
                      {rigorousData.statistical_significance.significance_detail}
                    </span>
                  </div>

                  <div className="monte-carlo-stats">
                    <div className="mc-stat">
                      <span className="mc-label">Strategy Return</span>
                      <span className={`mc-value ${rigorousData.statistical_significance.results?.strategy_return >= 0 ? 'positive' : 'negative'}`}>
                        {rigorousData.statistical_significance.results?.strategy_return >= 0 ? '+' : ''}
                        {rigorousData.statistical_significance.results?.strategy_return}%
                      </span>
                    </div>
                    <div className="mc-stat">
                      <span className="mc-label">Random Avg Return</span>
                      <span className="mc-value">
                        {rigorousData.statistical_significance.results?.random_mean_return}%
                      </span>
                    </div>
                    <div className="mc-stat">
                      <span className="mc-label">Percentile Rank</span>
                      <span className="mc-value">
                        {rigorousData.statistical_significance.results?.percentile}%
                      </span>
                    </div>
                    <div className="mc-stat">
                      <span className="mc-label">P-Value</span>
                      <span className={`mc-value ${rigorousData.statistical_significance.results?.p_value < 0.05 ? 'positive' : ''}`}>
                        {rigorousData.statistical_significance.results?.p_value}
                      </span>
                    </div>
                  </div>

                  <div className="interpretation-box">
                    <h6>Interpretation</h6>
                    <ul>
                      {rigorousData.statistical_significance.interpretation?.map((item, i) => (
                        <li key={i}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* Methodology */}
              <div className="rigorous-methodology">
                <h5>Methodology</h5>
                <ul>
                  <li><strong>Scoring:</strong> {rigorousData.portfolio_simulation?.methodology?.scoring}</li>
                  <li><strong>Rebalancing:</strong> {rigorousData.portfolio_simulation?.methodology?.rebalancing}</li>
                  <li><strong>Benchmark:</strong> {rigorousData.portfolio_simulation?.methodology?.benchmark}</li>
                  <li><strong>Bias Mitigation:</strong> {rigorousData.portfolio_simulation?.methodology?.bias_mitigation}</li>
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
