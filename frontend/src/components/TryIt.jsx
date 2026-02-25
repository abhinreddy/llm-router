import { useState } from 'react'
import { routePrompt } from '../api/client.js'
import { MODEL_SHORT, MODEL_PILL, TASK_LABEL, TASK_PILL } from '../lib/constants.js'

const STRATEGIES = [
  { value: 'balanced',         label: 'Balanced' },
  { value: 'minimize_cost',    label: 'Minimize Cost' },
  { value: 'maximize_quality', label: 'Maximize Quality' },
  { value: 'minimize_latency', label: 'Minimize Latency' },
]

function Badge({ label, colorClass }) {
  return (
    <span className={`px-2 py-0.5 rounded-md text-xs font-medium border border-transparent ${colorClass}`}>
      {label}
    </span>
  )
}

function MetaCell({ label, value }) {
  return (
    <div className="bg-zinc-900/80 rounded-lg p-2.5">
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-0.5">{label}</div>
      <div className="text-sm text-white font-medium">{value}</div>
    </div>
  )
}

function ComplexityBar({ score }) {
  const pct = Math.round(score * 100)
  const color =
    pct < 35 ? 'bg-emerald-500' :
    pct < 65 ? 'bg-amber-500'   :
               'bg-rose-500'
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex-1 bg-zinc-700 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-zinc-400 w-8 text-right tabular-nums">{pct}%</span>
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  )
}

export default function TryIt() {
  const [prompt, setPrompt]     = useState('')
  const [strategy, setStrategy] = useState('balanced')
  const [loading, setLoading]   = useState(false)
  const [result, setResult]     = useState(null)
  const [error, setError]       = useState(null)

  const handleRoute = async () => {
    if (!prompt.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await routePrompt({ prompt, strategy }))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleRoute()
  }

  return (
    <div className="bg-zinc-800 border border-zinc-700 rounded-xl p-5 flex flex-col gap-4 h-full">
      <h2 className="text-base font-semibold text-white">Try It</h2>

      {/* Input */}
      <div className="flex flex-col gap-2.5">
        <textarea
          className="w-full bg-zinc-900 border border-zinc-600 rounded-lg p-3 text-sm text-slate-100 placeholder-zinc-500 resize-none focus:outline-none focus:border-indigo-500 transition-colors leading-relaxed"
          rows={5}
          placeholder="Enter a prompt…  (⌘ Enter to send)"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <div className="flex gap-2">
          <select
            className="flex-1 bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 transition-colors cursor-pointer"
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
          >
            {STRATEGIES.map((s) => (
              <option key={s.value} value={s.value}>{s.label}</option>
            ))}
          </select>
          <button
            onClick={handleRoute}
            disabled={loading || !prompt.trim()}
            className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2 whitespace-nowrap"
          >
            {loading ? <><Spinner /> Routing…</> : 'Route →'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-950/40 border border-red-800 rounded-lg p-3 text-sm text-red-400 leading-relaxed">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="flex flex-col gap-3 border-t border-zinc-700 pt-4 min-h-0">
          {/* Badges */}
          <div className="flex flex-wrap gap-2 items-center">
            <Badge
              label={MODEL_SHORT[result.model_used] ?? result.model_used}
              colorClass={MODEL_PILL[result.model_used] ?? 'bg-zinc-700 text-zinc-300'}
            />
            <Badge
              label={TASK_LABEL[result.task_type] ?? result.task_type}
              colorClass={TASK_PILL[result.task_type] ?? 'bg-zinc-700 text-zinc-300'}
            />
            {result.cache_hit && (
              <Badge label="Cache Hit" colorClass="bg-teal-500/20 text-teal-300" />
            )}
          </div>

          {/* Metadata grid */}
          <div className="grid grid-cols-3 gap-2">
            <MetaCell label="Cost" value={`$${result.cost_usd.toFixed(6)}`} />
            <MetaCell label="Latency" value={`${(result.latency_ms / 1000).toFixed(2)}s`} />
            <MetaCell label="Tokens" value={`${result.input_tokens}↑ ${result.output_tokens}↓`} />
          </div>

          {/* Complexity */}
          <div>
            <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1.5">
              Complexity
            </div>
            <ComplexityBar score={result.complexity_score} />
          </div>

          {/* Reasoning */}
          <div className="bg-zinc-900/80 rounded-lg p-3">
            <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1.5">
              Routing Reasoning
            </div>
            <p className="text-xs text-zinc-300 leading-relaxed">{result.routing_reasoning}</p>
          </div>

          {/* Response */}
          <div className="bg-zinc-900/80 rounded-lg p-3 flex-1 min-h-0">
            <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1.5">
              Response
            </div>
            <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap max-h-52 overflow-y-auto">
              {result.response_text}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
