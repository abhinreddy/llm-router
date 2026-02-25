import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * Calls fetchFn immediately and then every `interval` ms.
 *
 * On failure, retries once after `retryDelay` ms before surfacing an error.
 * This prevents a single startup blip from triggering the error state.
 * When a subsequent request succeeds, error is cleared automatically.
 */
export function usePolling(fetchFn, interval = 5000, retryDelay = 1500) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  // Always call the latest versions without destabilising refresh
  const fnRef = useRef(fetchFn)
  const retryDelayRef = useRef(retryDelay)
  useEffect(() => { fnRef.current = fetchFn })
  useEffect(() => { retryDelayRef.current = retryDelay })

  const refresh = useCallback(async () => {
    try {
      const result = await fnRef.current()
      setData(result)
      setError(null)
    } catch {
      // One retry after a short delay before surfacing the error
      await new Promise((res) => setTimeout(res, retryDelayRef.current))
      try {
        const result = await fnRef.current()
        setData(result)
        setError(null)
      } catch (e) {
        setError(e.message)
      }
    } finally {
      setLoading(false)
    }
  }, []) // stable — reads latest fn/delay via refs

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, interval)
    return () => clearInterval(id)
  }, [refresh, interval])

  return { data, error, loading, refresh }
}
