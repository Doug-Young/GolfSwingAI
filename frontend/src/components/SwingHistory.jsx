import { useEffect, useState } from 'react'
import styles from './SwingHistory.module.css'
import ProgressChart from './ProgressChart'

function formatDate(ts) {
  const d = new Date(ts)
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  if (sameDay) return `Today ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
  const yesterday = new Date(now)
  yesterday.setDate(yesterday.getDate() - 1)
  if (d.toDateString() === yesterday.toDateString()) return `Yesterday ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export default function SwingHistory() {
  const [history, setHistory] = useState([])

  useEffect(() => {
    try {
      const raw = localStorage.getItem('golfai_history')
      if (raw) setHistory(JSON.parse(raw))
    } catch { /* ignore */ }
  }, [])

  function clearHistory() {
    if (!confirm('Clear all saved swing analyses?')) return
    localStorage.removeItem('golfai_history')
    setHistory([])
  }

  if (history.length === 0) return null

  return (
    <div className={styles.wrapper}>
      <ProgressChart history={history} />
      <div className={styles.header}>
        <h3 className={styles.title}>Your Recent Swings</h3>
        <button className={styles.clearBtn} onClick={clearHistory}>Clear</button>
      </div>
      <div className={styles.grid}>
        {history.map((h, i) => (
          <div key={h.timestamp} className={styles.card}>
            <div className={styles.cardHeader}>
              <span className={styles.date}>{formatDate(h.timestamp)}</span>
              {i === 0 && <span className={styles.latest}>Latest</span>}
            </div>
            {h.grade !== undefined && h.grade !== null && (
              <div className={styles.stat}>
                <span className={styles.statLabel}>Grade</span>
                <span className={styles.statValue}>{h.grade}/100</span>
              </div>
            )}
            {h.tempo && (
              <div className={styles.stat}>
                <span className={styles.statLabel}>Tempo</span>
                <span className={styles.statValue}>{h.tempo.ratio}:1</span>
              </div>
            )}
            {h.metrics_summary?.top_x_factor_deg !== undefined && h.metrics_summary?.top_x_factor_deg !== null && (
              <div className={styles.stat}>
                <span className={styles.statLabel}>X-Factor</span>
                <span className={styles.statValue}>{h.metrics_summary.top_x_factor_deg}°</span>
              </div>
            )}
            {h.metrics_summary?.head_movement_pct !== undefined && (
              <div className={styles.stat}>
                <span className={styles.statLabel}>Head Move</span>
                <span className={styles.statValue}>{h.metrics_summary.head_movement_pct}%</span>
              </div>
            )}
            {h.metrics_summary?.address_spine_tilt_deg !== undefined && h.metrics_summary?.address_spine_tilt_deg !== null && (
              <div className={styles.stat}>
                <span className={styles.statLabel}>Spine</span>
                <span className={styles.statValue}>{h.metrics_summary.address_spine_tilt_deg}°</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
