import { useCallback } from 'react'
import { usePolling } from '../hooks/usePolling.js'
import { fetchLogs } from '../api/client.js'
import { MODEL_SHORT, MODEL_PILL, TASK_LABEL, TASK_PILL } from '../lib/constants.js'

function Pill({ label, colorClass }) {
  return (
    <span className={`inline-flex px-2 py-0.5 rounded text-[11px] font-medium whitespace-nowrap ${colorClass}`}>
      {label}
    </span>
  )
}

function fmtTime(ts) {
  try {
    return new Date(ts).toLocaleTimeString([], {
      hour:   '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return ts
  }
}

function fmtDate(ts) {
  try {
    return new Date(ts).toLocaleDateString([], { month: 'short', day: 'numeric' })
  } catch {
    return ''
  }
}

function SkeletonRow() {
  return (
    <tr>
      {Array.from({ length: 8 }).map((_, i) => (
        <td key={i} className="py-3 pr-4">
          <div className="h-3 bg-zinc-700/50 rounded animate-pulse" style={{ width: `${40 + (i * 13) % 40}%` }} />
        </td>
      ))}
    </tr>
  )
}

const TH = ({ children, right }) => (
  <th className={`py-2 pr-4 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider ${right ? 'text-right' : 'text-left'}`}>
    {children}
  </th>
)

export default function RequestLog() {
  const logsQuery = useCallback(() => fetchLogs({ limit: 50 }), [])
  const { data, loading, error } = usePolling(logsQuery, 5000)

  return (
    <div className="bg-zinc-800 border border-zinc-700 rounded-xl p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold text-white">Request Log</h2>
        <div className="flex items-center gap-3">
          {data && (
            <span className="text-xs text-zinc-500">
              {data.total_rows.toLocaleString()} total · showing {data.rows.length}
            </span>
          )}
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[11px] text-zinc-500">live</span>
          </div>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="text-red-400 text-sm py-2">Failed to load logs: {error}</div>
      )}

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm min-w-[720px]">
          <thead>
            <tr className="border-b border-zinc-700">
              <TH>Time</TH>
              <TH>Prompt</TH>
              <TH>Task</TH>
              <TH right>Complexity</TH>
              <TH>Model</TH>
              <TH right>Cost</TH>
              <TH right>Latency</TH>
              <TH>Cache</TH>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-700/40">
            {loading && !data
              ? Array.from({ length: 6 }).map((_, i) => <SkeletonRow key={i} />)
              : null
            }

            {data?.rows.length === 0 && (
              <tr>
                <td colSpan={8} className="py-10 text-center text-zinc-500 text-sm">
                  No requests yet — try routing a prompt above.
                </td>
              </tr>
            )}

            {data?.rows.map((row) => (
              <tr
                key={row.id}
                className="hover:bg-zinc-700/20 transition-colors group"
              >
                {/* Time */}
                <td className="py-2.5 pr-4 whitespace-nowrap">
                  <div className="text-xs text-zinc-300 tabular-nums">{fmtTime(row.timestamp)}</div>
                  <div className="text-[10px] text-zinc-600">{fmtDate(row.timestamp)}</div>
                </td>

                {/* Prompt snippet */}
                <td className="py-2.5 pr-4 max-w-[200px]">
                  <span
                    className="text-xs text-zinc-400 truncate block"
                    title={row.prompt_snippet}
                  >
                    {row.prompt_snippet.length > 55
                      ? row.prompt_snippet.slice(0, 55) + '…'
                      : row.prompt_snippet}
                  </span>
                </td>

                {/* Task type */}
                <td className="py-2.5 pr-4">
                  <Pill
                    label={TASK_LABEL[row.task_type] ?? row.task_type}
                    colorClass={TASK_PILL[row.task_type] ?? 'bg-zinc-700 text-zinc-300'}
                  />
                </td>

                {/* Complexity */}
                <td className="py-2.5 pr-4 text-right">
                  <span className="text-xs text-zinc-400 tabular-nums">
                    {(row.complexity_score * 100).toFixed(0)}%
                  </span>
                </td>

                {/* Model */}
                <td className="py-2.5 pr-4">
                  <Pill
                    label={MODEL_SHORT[row.model_selected] ?? row.model_selected}
                    colorClass={MODEL_PILL[row.model_selected] ?? 'bg-zinc-700 text-zinc-300'}
                  />
                </td>

                {/* Cost */}
                <td className="py-2.5 pr-4 text-right">
                  <span className="text-xs text-zinc-300 font-mono tabular-nums">
                    ${row.cost_usd.toFixed(5)}
                  </span>
                </td>

                {/* Latency */}
                <td className="py-2.5 pr-4 text-right">
                  <span className="text-xs text-zinc-400 tabular-nums">
                    {(row.latency_ms / 1000).toFixed(2)}s
                  </span>
                </td>

                {/* Cache */}
                <td className="py-2.5">
                  {row.cache_hit
                    ? <span className="text-[11px] font-medium text-teal-400">HIT</span>
                    : <span className="text-[11px] text-zinc-700">—</span>
                  }
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
