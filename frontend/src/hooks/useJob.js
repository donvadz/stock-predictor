import { useState, useEffect, useRef, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Hook for managing cancellable long-running jobs.
 *
 * Jobs run on the backend and persist across page refresh/tab close.
 * Only explicit cancellation stops a job.
 *
 * @param {string} storageKey - Unique key for localStorage persistence (e.g., 'screener-job')
 * @returns {Object} Job state and control functions
 */
export function useJob(storageKey = null) {
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [progress, setProgress] = useState(0)
  const [progressMessage, setProgressMessage] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  const pollingRef = useRef(null)
  const isMountedRef = useRef(true)

  // Derived state
  const isLoading = status === 'pending' || status === 'running'
  const isCompleted = status === 'completed'
  const isCancelled = status === 'cancelled'
  const isFailed = status === 'failed'

  // Get storage key for this job type
  const getStorageKey = useCallback(() => {
    return storageKey ? `job-${storageKey}` : null
  }, [storageKey])

  // Save job ID to localStorage
  const saveJobId = useCallback((id) => {
    const key = getStorageKey()
    if (key && id) {
      localStorage.setItem(key, id)
    }
  }, [getStorageKey])

  // Clear job ID from localStorage
  const clearJobId = useCallback(() => {
    const key = getStorageKey()
    if (key) {
      localStorage.removeItem(key)
    }
  }, [getStorageKey])

  // Load job ID from localStorage
  const loadJobId = useCallback(() => {
    const key = getStorageKey()
    if (key) {
      return localStorage.getItem(key)
    }
    return null
  }, [getStorageKey])

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }
  }, [])

  // Poll for job status
  const pollJobStatus = useCallback(async (id) => {
    if (!isMountedRef.current) return

    try {
      const response = await fetch(`${API_BASE}/jobs/${id}/status`)
      if (!response.ok) {
        if (response.status === 404) {
          // Job not found - it may have expired
          clearJobId()
          stopPolling()
          setStatus(null)
          setJobId(null)
          return
        }
        throw new Error('Failed to fetch job status')
      }

      const data = await response.json()

      if (!isMountedRef.current) return

      setJobId(id)
      setStatus(data.status)
      setProgress(data.progress)
      setProgressMessage(data.progress_message)
      setElapsedSeconds(data.elapsed_seconds)

      if (data.status === 'completed') {
        setResult(data.result)
        clearJobId()
        stopPolling()
      } else if (data.status === 'failed') {
        setError(data.error || 'Job failed')
        clearJobId()
        stopPolling()
      } else if (data.status === 'cancelled') {
        setError('Job was cancelled')
        clearJobId()
        stopPolling()
      }
    } catch (err) {
      console.error('Error polling job status:', err)
      // Don't stop polling on transient errors
    }
  }, [clearJobId, stopPolling])

  const startPolling = useCallback((id) => {
    stopPolling()
    // Poll immediately, then every 2 seconds
    pollJobStatus(id)
    pollingRef.current = setInterval(() => pollJobStatus(id), 2000)
  }, [pollJobStatus, stopPolling])

  // On mount, check for existing running job
  useEffect(() => {
    isMountedRef.current = true

    const savedJobId = loadJobId()
    if (savedJobId) {
      // Resume tracking the saved job
      setStatus('running') // Assume running until we know otherwise
      startPolling(savedJobId)
    }

    return () => {
      isMountedRef.current = false
      stopPolling()
    }
  }, [loadJobId, startPolling, stopPolling])

  /**
   * Start a new job.
   *
   * @param {string} jobType - Type of job (screener, backtest, etc.)
   * @param {Object} params - Job parameters
   * @returns {Promise<string>} Job ID
   */
  const startJob = useCallback(async (jobType, params = {}) => {
    // Reset state
    setJobId(null)
    setStatus('pending')
    setProgress(0)
    setProgressMessage('Starting...')
    setResult(null)
    setError(null)
    setElapsedSeconds(0)
    stopPolling()
    clearJobId()

    try {
      const response = await fetch(`${API_BASE}/jobs/${jobType}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to start job')
      }

      const data = await response.json()
      const newJobId = data.job_id

      if (!isMountedRef.current) return newJobId

      setJobId(newJobId)
      setStatus(data.status)
      saveJobId(newJobId) // Persist to localStorage
      startPolling(newJobId)

      return newJobId
    } catch (err) {
      if (!isMountedRef.current) return null

      setStatus('failed')
      setError(err.message)
      throw err
    }
  }, [startPolling, stopPolling, saveJobId, clearJobId])

  /**
   * Cancel the current job.
   *
   * @returns {Promise<boolean>} True if cancellation was requested
   */
  const cancelJob = useCallback(async () => {
    const currentJobId = jobId || loadJobId()
    if (!currentJobId) return false

    try {
      const response = await fetch(`${API_BASE}/jobs/${currentJobId}/cancel`, {
        method: 'POST',
      })

      if (!response.ok) {
        throw new Error('Failed to cancel job')
      }

      const data = await response.json()

      if (!isMountedRef.current) return true

      if (data.status === 'cancelling') {
        setProgressMessage('Cancelling...')
      }

      return true
    } catch (err) {
      console.error('Error cancelling job:', err)
      return false
    }
  }, [jobId, loadJobId])

  /**
   * Reset the job state without starting a new job.
   */
  const reset = useCallback(() => {
    stopPolling()
    clearJobId()
    setJobId(null)
    setStatus(null)
    setProgress(0)
    setProgressMessage('')
    setResult(null)
    setError(null)
    setElapsedSeconds(0)
  }, [stopPolling, clearJobId])

  return {
    // State
    jobId,
    status,
    progress,
    progressMessage,
    result,
    error,
    elapsedSeconds,

    // Derived state
    isLoading,
    isCompleted,
    isCancelled,
    isFailed,

    // Actions
    startJob,
    cancelJob,
    reset,
  }
}

export default useJob
