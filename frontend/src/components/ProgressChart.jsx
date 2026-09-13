import styles from './ProgressChart.module.css'

// Simple SVG line chart of one metric over history entries (newest last).
function LineChart({ title, values, unit, targetLow, targetHigh, color = '#4caf6a', width = 260, height = 100 }) {
  const cleaned = values.map((v) => (v === null || v === undefined ? null : Number(v))).filter(v => v !== null)
  if (cleaned.length < 2) {
    return (
      <div className={styles.chart}>
        <div className={styles.chartHeader}>
          <span className={styles.chartTitle}>{title}</span>
          <span className={styles.chartLatest}>{cleaned.length === 1 ? `${cleaned[0]}${unit}` : '—'}</span>
        </div>
        <div className={styles.chartEmpty}>Need at least 2 swings for a trend</div>
      </div>
    )
  }

  const min = Math.min(...cleaned)
  const max = Math.max(...cleaned)
  const range = Math.max(max - min, 1e-6)
  const pad = 8

  const points = cleaned.map((v, i) => {
    const x = pad + (i / (cleaned.length - 1)) * (width - 2 * pad)
    const y = height - pad - ((v - min) / range) * (height - 2 * pad)
    return [x, y]
  })

  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0]} ${p[1]}`).join(' ')
  const latest = cleaned[cleaned.length - 1]

  // Delta from previous swing
  const delta = cleaned.length >= 2 ? latest - cleaned[cleaned.length - 2] : null
  const deltaSign = delta > 0 ? '+' : ''

  return (
    <div className={styles.chart}>
      <div className={styles.chartHeader}>
        <span className={styles.chartTitle}>{title}</span>
        <span className={styles.chartLatest}>
          {latest.toFixed(1)}{unit}
          {delta !== null && (
            <span className={styles.chartDelta} style={{ color: Math.abs(delta) < 0.1 ? 'var(--text-muted)' : (delta > 0 ? '#8ab4f8' : '#f0a050') }}>
              {' '}{deltaSign}{delta.toFixed(1)}
            </span>
          )}
        </span>
      </div>
      <svg width={width} height={height} className={styles.svg}>
        {/* Target band */}
        {targetLow !== undefined && targetHigh !== undefined && (
          <rect
            x={pad}
            y={height - pad - ((targetHigh - min) / range) * (height - 2 * pad)}
            width={width - 2 * pad}
            height={Math.max(
              ((targetHigh - targetLow) / range) * (height - 2 * pad),
              1
            )}
            fill={color}
            opacity="0.08"
          />
        )}
        {/* Line */}
        <path d={path} stroke={color} strokeWidth="2" fill="none" />
        {/* Dots */}
        {points.map((p, i) => (
          <circle key={i} cx={p[0]} cy={p[1]} r={i === points.length - 1 ? 4 : 2.5} fill={color} />
        ))}
      </svg>
    </div>
  )
}

export default function ProgressChart({ history }) {
  if (!history || history.length < 2) return null

  // Reverse so oldest → newest
  const ordered = [...history].reverse()

  const grades = ordered.map(h => h.grade ?? null)
  const tempoRatios = ordered.map(h => h.tempo?.ratio ?? null)
  const xFactors = ordered.map(h => h.metrics_summary?.top_x_factor_deg ?? null)
  const spineTilts = ordered.map(h => h.metrics_summary?.address_spine_tilt_deg ?? null)
  const headMoves = ordered.map(h => h.metrics_summary?.head_movement_pct ?? null)

  return (
    <section className={styles.wrapper}>
      <h3 className={styles.title}>Progress Over Your Last {ordered.length} Swings</h3>
      <div className={styles.grid}>
        <LineChart title="Overall Grade" values={grades} unit="/100" color="#4caf6a" targetLow={70} targetHigh={100} />
        <LineChart title="Tempo Ratio" values={tempoRatios} unit=":1" color="#e8cc80" targetLow={2.8} targetHigh={3.2} />
        <LineChart title="X-Factor" values={xFactors} unit="°" color="#8ab4f8" targetLow={25} targetHigh={45} />
        <LineChart title="Address Spine" values={spineTilts} unit="°" color="#4caf6a" targetLow={25} targetHigh={35} />
        <LineChart title="Head Movement" values={headMoves} unit="%" color="#f0a050" targetLow={0} targetHigh={8} />
      </div>
    </section>
  )
}
