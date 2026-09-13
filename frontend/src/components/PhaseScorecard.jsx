import styles from './PhaseScorecard.module.css'

const STATUS_META = {
  good:    { color: '#4caf6a', label: 'Good',   icon: '✓' },
  watch:   { color: '#c9a84c', label: 'Watch',  icon: '!' },
  issue:   { color: '#e05c5c', label: 'Issue',  icon: '✕' },
  unknown: { color: '#7a8390', label: '—',      icon: '?' },
}

export default function PhaseScorecard({ scores = [] }) {
  if (!scores || scores.length === 0) return null

  return (
    <section className={styles.wrapper}>
      <h3 className={styles.sectionTitle}>Phase Scorecard</h3>
      <div className={styles.grid}>
        {scores.map((s, i) => {
          const meta = STATUS_META[s.status] || STATUS_META.unknown
          return (
            <div key={i} className={styles.card} style={{ borderTopColor: meta.color }}>
              <div className={styles.header}>
                <span className={styles.phaseName}>{s.phase}</span>
                <span
                  className={styles.status}
                  style={{ background: meta.color + '22', color: meta.color, borderColor: meta.color + '55' }}
                >
                  {meta.icon} {meta.label}
                </span>
              </div>

              {s.notes && <p className={styles.notes}>{s.notes}</p>}

              {s.findings && s.findings.length > 0 && (
                <ul className={styles.findings}>
                  {s.findings.map((f, j) => {
                    const findingColor = STATUS_META[
                      f.kind === 'good' ? 'good' : f.kind === 'issue' ? 'issue' : 'watch'
                    ].color
                    return (
                      <li key={j} className={styles.finding}>
                        <span className={styles.findingDot} style={{ background: findingColor }} />
                        <div>
                          <span className={styles.findingText}>{f.text}</span>
                          {f.drill && (
                            <span className={styles.findingDrill}>
                              <b>Drill:</b> {f.drill}
                            </span>
                          )}
                        </div>
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}
