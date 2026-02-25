import StatsBar from './components/StatsBar.jsx'
import TryIt from './components/TryIt.jsx'
import ChartsSection from './components/ChartsSection.jsx'
import RequestLog from './components/RequestLog.jsx'

export default function App() {
  return (
    <div className="min-h-screen bg-zinc-900 text-slate-100">

      {/* ── Header ── */}
      <header className="sticky top-0 z-20 border-b border-zinc-800 bg-zinc-900/90 backdrop-blur-sm">
        <div className="max-w-screen-xl mx-auto px-6 py-3.5 flex items-center justify-between">
          {/* Logo + title */}
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-bold text-white leading-none">LLM Router</p>
              <p className="text-[10px] text-zinc-500 leading-none mt-0.5">
                Cost-optimizing AI router · Anthropic
              </p>
            </div>
          </div>

          {/* Live indicator */}
          <div className="flex items-center gap-2 bg-zinc-800 border border-zinc-700 rounded-full px-3 py-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            <span className="text-[11px] text-zinc-400">Polling every 5s</span>
          </div>
        </div>
      </header>

      {/* ── Main content ── */}
      <main className="max-w-screen-xl mx-auto px-6 py-6 space-y-6">

        {/* Stats bar */}
        <StatsBar />

        {/* Try It (left 2/5) + Charts (right 3/5) */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 items-start">
          <div className="lg:col-span-2">
            <TryIt />
          </div>
          <div className="lg:col-span-3">
            <ChartsSection />
          </div>
        </div>

        {/* Request log */}
        <RequestLog />

      </main>
    </div>
  )
}
