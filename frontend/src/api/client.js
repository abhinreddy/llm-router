const BASE = 'http://localhost:8000'

export async function fetchStats() {
  const res = await fetch(`${BASE}/stats`)
  if (!res.ok) throw new Error(`/stats failed: ${res.status}`)
  return res.json()
}

export async function fetchLogs(params = {}) {
  const url = new URL(`${BASE}/logs`)
  Object.entries(params).forEach(([k, v]) => {
    if (v != null) url.searchParams.set(k, v)
  })
  const res = await fetch(url)
  if (!res.ok) throw new Error(`/logs failed: ${res.status}`)
  return res.json()
}

export async function routePrompt({ prompt, strategy, classifier = 'ml' }) {
  const res = await fetch(`${BASE}/route?classifier=${classifier}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      policy: { strategy },
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}
