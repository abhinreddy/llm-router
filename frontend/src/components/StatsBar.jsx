import { usePolling } from '../hooks/usePolling.js'
import { fetchStats } from '../api/client.js'

function MetricCard({ label, value, sub, valueClass = 'text-white' }) {
  return (
    <div className="bg-zinc-800 border border-zinc-700 rounded-xl p-4 flex flex-col gap-1.5">
      <span className="text-[11px] text-zinc-400 uppercase tracking-widest font-medium">
        {label}
      </span>
      <span className={`text-2xl font-bold leading-none ${valueClass}`}>
        {value}
      </span>
      {sub && (
        <span className="text-xs text-zinc-500 leading-none">{sub}</span>
      )}
    </div>
  )
}

function SkeletonCard() {
  return (
    <div className="bg-zinc-800 border border-zinc-700 rounded-xl p-4 h-[88px] animate-pulse">
      <div className="h-2.5 w-20 bg-zinc-700 rounded mb-3" />
      <div className="h-7 w-28 bg-zinc-700 rounded mb-2" />
      <div className="h-2 w-24 bg-zinc-700/60 rounded" />
    </div>
  )
}

export default function StatsBar() {
  const { data, error, loading } = usePolling(fetchStats, 5000)

  // True first load — no data yet at all
  if (loading && !data) {
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {Array.from({ length: 5 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      </div>
    )
  }

  const d = data

  return (
    <div className="space-y-3">
      {/* No data + error: backend is unreachable (auto-dismissed once a request succeeds) */}
      {error && !d && (
        <div className="bg-red-950/40 border border-red-800 rounded-xl p-4 text-red-400 text-sm flex items-center gap-2">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
          Cannot reach backend at localhost:8000 — retrying…
        </div>
      )}

      {/* Have data but last refresh failed: show stale-data warning, keep cards visible */}
      {error && d && (
        <div className="bg-amber-950/30 border border-amber-900/60 rounded-lg px-4 py-2 text-amber-500/90 text-xs flex items-center gap-2">
          <svg className="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
          Connection issue — showing last known data, retrying…
        </div>
      )}

      {/* Cards: always shown once we have data, regardless of error state */}
      {d && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          <MetricCard
            label="Total Requests"
            value={d.total_requests.toLocaleString()}
            sub={`${d.api_calls} API · ${d.cache_hits} cached`}
          />
          <MetricCard
            label="Total Cost"
            value={`$${d.total_cost_usd.toFixed(4)}`}
            sub={`vs $${d.hypothetical_opus_cost_usd.toFixed(4)} at Opus`}
          />
          <MetricCard
            label="Savings vs Opus"
            value={`$${d.cost_savings_usd.toFixed(4)}`}
            sub={`${d.cost_savings_pct.toFixed(1)}% cheaper`}
            valueClass="text-emerald-400"
          />
          <MetricCard
            label="Avg Latency"
            value={`${(d.average_latency_ms / 1000).toFixed(1)}s`}
            sub="API calls only"
          />
          <MetricCard
            label="Cache Hit Rate"
            value={`${(d.cache_hit_rate * 100).toFixed(1)}%`}
            sub={`${d.cache_size} entries in cache`}
          />
        </div>
      )}
    </div>
  )
}
