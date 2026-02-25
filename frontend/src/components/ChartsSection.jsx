import { useCallback } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  LineChart, Line,
} from 'recharts'
import { usePolling } from '../hooks/usePolling.js'
import { fetchLogs } from '../api/client.js'
import { MODEL_SHORT, MODEL_COLOR, TASK_LABEL, TASK_COLOR } from '../lib/constants.js'

// Shared dark tooltip style
const TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: '#1c1c1f',
    border: '1px solid #3f3f46',
    borderRadius: '8px',
    padding: '8px 12px',
    fontSize: '12px',
  },
  labelStyle: { color: '#a1a1aa' },
  itemStyle:  { color: '#e4e4e7' },
  cursor:     { fill: 'rgba(255,255,255,0.04)' },
}

const AXIS_STYLE = { fill: '#71717a', fontSize: 11 }

function ChartCard({ title, children }) {
  return (
    <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-4">
      <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
        {title}
      </h3>
      {children}
    </div>
  )
}

function Empty() {
  return (
    <div className="flex items-center justify-center h-40 text-zinc-600 text-sm">
      No data yet
    </div>
  )
}

function SkeletonChart() {
  return <div className="h-48 bg-zinc-700/30 rounded-lg animate-pulse" />
}

// Custom label for pie slices showing percentage
function PieLabel({ cx, cy, midAngle, outerRadius, percent }) {
  if (percent < 0.07) return null
  const RAD = Math.PI / 180
  const x = cx + (outerRadius + 16) * Math.cos(-midAngle * RAD)
  const y = cy + (outerRadius + 16) * Math.sin(-midAngle * RAD)
  return (
    <text x={x} y={y} fill="#a1a1aa" textAnchor="middle" dominantBaseline="central" fontSize={10}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  )
}

export default function ChartsSection() {
  const logsQuery = useCallback(() => fetchLogs({ limit: 200 }), [])
  const { data, loading } = usePolling(logsQuery, 5000)

  if (loading || !data) {
    return (
      <div className="grid grid-cols-2 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <ChartCard key={i} title="">
            <SkeletonChart />
          </ChartCard>
        ))}
      </div>
    )
  }

  const { summary, rows } = data

  // --- Data transforms ---

  // Pie: requests by model
  const modelPie = Object.entries(summary.per_model).map(([id, v]) => ({
    name:  MODEL_SHORT[id] ?? id,
    value: v.count,
    color: MODEL_COLOR[id] ?? '#6b7280',
  }))

  // Pie: requests by task type
  const taskPie = Object.entries(summary.per_task_type).map(([type, v]) => ({
    name:  TASK_LABEL[type] ?? type,
    value: v.count,
    color: TASK_COLOR[type] ?? '#6b7280',
  }))

  // Bar: cost per model
  const modelBar = Object.entries(summary.per_model).map(([id, v]) => ({
    name:  MODEL_SHORT[id] ?? id,
    cost:  parseFloat(v.cost_usd.toFixed(5)),
    color: MODEL_COLOR[id] ?? '#6b7280',
  }))

  // Line: latency over time (non-cache-hit rows, chronological)
  const latencyLine = rows
    .filter((r) => !r.cache_hit)
    .slice()                        // copy before reversing
    .reverse()                      // rows are newest-first; flip to chronological
    .slice(-50)                     // keep last 50 data points
    .map((r, i) => ({
      i:       i + 1,
      latency: parseFloat((r.latency_ms / 1000).toFixed(2)),
      model:   MODEL_SHORT[r.model_selected] ?? r.model_selected,
    }))

  // Custom legend text renderer (smaller, muted)
  const legendText = (value) => (
    <span style={{ color: '#a1a1aa', fontSize: 11 }}>{value}</span>
  )

  return (
    <div className="grid grid-cols-2 gap-4">

      {/* ── Pie: by model ── */}
      <ChartCard title="Requests by Model">
        {modelPie.length === 0 ? <Empty /> : (
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={modelPie}
                cx="50%" cy="50%"
                innerRadius={44} outerRadius={68}
                paddingAngle={3}
                dataKey="value"
                labelLine={false}
                label={<PieLabel />}
              >
                {modelPie.map((entry, i) => (
                  <Cell key={i} fill={entry.color} stroke="transparent" />
                ))}
              </Pie>
              <Tooltip
                contentStyle={TOOLTIP_STYLE.contentStyle}
                itemStyle={TOOLTIP_STYLE.itemStyle}
                formatter={(v, name) => [v, name]}
              />
              <Legend formatter={legendText} iconSize={8} iconType="circle" />
            </PieChart>
          </ResponsiveContainer>
        )}
      </ChartCard>

      {/* ── Pie: by task type ── */}
      <ChartCard title="Requests by Task Type">
        {taskPie.length === 0 ? <Empty /> : (
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={taskPie}
                cx="50%" cy="50%"
                innerRadius={44} outerRadius={68}
                paddingAngle={3}
                dataKey="value"
                labelLine={false}
                label={<PieLabel />}
              >
                {taskPie.map((entry, i) => (
                  <Cell key={i} fill={entry.color} stroke="transparent" />
                ))}
              </Pie>
              <Tooltip
                contentStyle={TOOLTIP_STYLE.contentStyle}
                itemStyle={TOOLTIP_STYLE.itemStyle}
                formatter={(v, name) => [v, name]}
              />
              <Legend formatter={legendText} iconSize={8} iconType="circle" />
            </PieChart>
          </ResponsiveContainer>
        )}
      </ChartCard>

      {/* ── Bar: cost per model ── */}
      <ChartCard title="Cost by Model (USD)">
        {modelBar.length === 0 ? <Empty /> : (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={modelBar} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" vertical={false} />
              <XAxis dataKey="name" tick={AXIS_STYLE} axisLine={false} tickLine={false} />
              <YAxis
                tick={AXIS_STYLE}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => `$${v}`}
                width={48}
              />
              <Tooltip
                {...TOOLTIP_STYLE}
                formatter={(v) => [`$${v.toFixed(5)}`, 'Cost']}
              />
              <Bar dataKey="cost" radius={[4, 4, 0, 0]} maxBarSize={48}>
                {modelBar.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </ChartCard>

      {/* ── Line: latency over time ── */}
      <ChartCard title="API Latency Over Time (s)">
        {latencyLine.length === 0 ? <Empty /> : (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={latencyLine} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" vertical={false} />
              <XAxis dataKey="i" tick={AXIS_STYLE} axisLine={false} tickLine={false} label={null} />
              <YAxis
                tick={AXIS_STYLE}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => `${v}s`}
                width={36}
              />
              <Tooltip
                {...TOOLTIP_STYLE}
                formatter={(v, _, props) => [
                  `${v}s`,
                  props.payload?.model ?? 'Latency',
                ]}
                labelFormatter={(i) => `Request #${i}`}
              />
              <Line
                type="monotone"
                dataKey="latency"
                stroke="#6366f1"
                strokeWidth={2}
                dot={{ fill: '#6366f1', r: 3, strokeWidth: 0 }}
                activeDot={{ r: 5, strokeWidth: 0 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </ChartCard>

    </div>
  )
}
