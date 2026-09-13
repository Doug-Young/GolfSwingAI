import { useEffect, useRef, useState } from 'react'
import styles from './AnalysisResults.module.css'
import PhaseScorecard from './PhaseScorecard'

const API_URL = 'http://localhost:8000'

const PHASE_COLORS = {
  'Address / Setup': '#4caf6a',
  'Takeaway': '#7dd09a',
  'Backswing': '#c9a84c',
  'Top of Backswing': '#e8cc80',
  'Downswing': '#f0a050',
  'Impact': '#e05c5c',
  'Follow-Through': '#8ab4f8',
  'Finish': '#b39ddb',
}

function phaseColor(phase) {
  for (const [key, color] of Object.entries(PHASE_COLORS)) {
    if (phase?.toLowerCase().includes(key.toLowerCase())) return color
  }
  return '#7dd09a'
}

function MetricStat({ label, value, unit, hint }) {
  return (
    <div className={styles.metricStat}>
      <span className={styles.metricLabel}>{label}</span>
      <span className={styles.metricValue}>
        {value !== null && value !== undefined ? value : '—'}
        {value !== null && value !== undefined && unit && <span className={styles.metricUnit}>{unit}</span>}
      </span>
      {hint && <span className={styles.metricHint}>{hint}</span>}
    </div>
  )
}

function TempoBar({ tempo }) {
  if (!tempo) return null
  const total = Math.max(0.001, tempo.backswing_s + tempo.downswing_s)
  const backPct = (tempo.backswing_s / total) * 100

  return (
    <div className={styles.tempoBar}>
      <div className={styles.tempoTop}>
        <span>
          <b>{tempo.backswing_s}s</b> backswing
        </span>
        <span className={styles.tempoRatio}>{tempo.ratio}:1</span>
        <span>
          <b>{tempo.downswing_s}s</b> downswing
        </span>
      </div>
      <div className={styles.tempoTrack}>
        <div className={styles.tempoBack} style={{ width: `${backPct}%` }} />
        <div className={styles.tempoDown} style={{ width: `${100 - backPct}%` }} />
      </div>
      <p className={styles.tempoVerdict}>{tempo.verdict}</p>
    </div>
  )
}

export default function AnalysisResults({ analysis, videoPreview, onReset }) {
  const {
    overall_assessment,
    frames = [],
    strengths = [],
    improvements = [],
    key_focus,
    metrics_summary = {},
    poses_detected = 0,
    total_frames = 0,
    camera_note,
    tempo,
    swing_plane_image,
    phase_scores = [],
    handedness,
    overlay_video_url,
    grade,
    grade_band,
    practice_plan = [],
    shaft_plane,
    shaft_detection_rate,
  } = analysis

  const [activeFrame, setActiveFrame] = useState(0)
  const [autoPlay, setAutoPlay] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(0.5)
  const [compareFrame, setCompareFrame] = useState(null)
  const overlayVideoRef = useRef(null)
  const current = frames[activeFrame]
  const compare = compareFrame !== null ? frames[compareFrame] : null

  // Apply playback rate to overlay video whenever it changes
  useEffect(() => {
    if (overlayVideoRef.current) {
      overlayVideoRef.current.playbackRate = playbackRate
    }
  }, [playbackRate, overlay_video_url])

  // Keyboard shortcuts on the scrubber
  useEffect(() => {
    function onKey(e) {
      if (frames.length === 0) return
      // Skip if user is typing in an input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if (e.key === 'ArrowRight') {
        setActiveFrame(f => Math.min(f + 1, frames.length - 1))
        setAutoPlay(false)
        e.preventDefault()
      } else if (e.key === 'ArrowLeft') {
        setActiveFrame(f => Math.max(f - 1, 0))
        setAutoPlay(false)
        e.preventDefault()
      } else if (e.key === ' ') {
        setAutoPlay(p => !p)
        e.preventDefault()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [frames.length])

  const detectionRate = total_frames > 0 ? Math.round((poses_detected / total_frames) * 100) : 0
  const poorDetection = detectionRate < 60

  // Auto-play through key frames
  const playRef = useRef()
  useEffect(() => {
    if (!autoPlay || frames.length === 0) return
    playRef.current = setInterval(() => {
      setActiveFrame((prev) => (prev + 1) % frames.length)
    }, 500)
    return () => clearInterval(playRef.current)
  }, [autoPlay, frames.length])

  // Save this analysis to localStorage for the history feature
  useEffect(() => {
    if (!analysis) return
    try {
      const raw = localStorage.getItem('golfai_history')
      const history = raw ? JSON.parse(raw) : []
      // Compact record — avoid storing full base64 frames (too big)
      const record = {
        timestamp: Date.now(),
        overall_assessment,
        tempo,
        metrics_summary,
        detection_rate: detectionRate,
        key_focus,
        grade,
        grade_band,
      }
      // De-dupe: don't save duplicate of last entry
      if (history.length === 0 || history[0].timestamp < record.timestamp - 5000) {
        history.unshift(record)
        localStorage.setItem('golfai_history', JSON.stringify(history.slice(0, 10)))
      }
    } catch {
      // ignore quota errors
    }
    // Only save once per analysis mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className={styles.wrapper}>
      {/* Header bar */}
      <div className={styles.topBar}>
        <div>
          <h2 className={styles.title}>Swing Analysis Complete</h2>
          <div className={styles.badgeRow}>
            <span className={styles.detectionPill} data-warn={poorDetection}>
              Pose detected in {poses_detected}/{total_frames} samples
            </span>
            {shaft_detection_rate !== undefined && shaft_detection_rate > 0 && (
              <span className={styles.detectionPill} data-warn={shaft_detection_rate < 40}>
                Club shaft tracked on {shaft_detection_rate}% of frames
              </span>
            )}
          </div>
        </div>
        <button className={styles.resetBtn} onClick={onReset}>
          Analyse Another
        </button>
      </div>

      {/* Fundamentals Score */}
      {grade !== null && grade !== undefined && (
        <div className={styles.gradeCard}>
          <div className={styles.gradeCircle} data-band={grade_band?.toLowerCase().replace(' ', '-')}>
            <span className={styles.gradeNumber}>{grade}</span>
            <span className={styles.gradeOutOf}>/100</span>
          </div>
          <div className={styles.gradeBody}>
            <span className={styles.gradeBand}>{grade_band} fundamentals</span>
            <p className={styles.gradeExplain}>
              Score is based on visible body mechanics across Setup, Backswing, Top, Impact and Finish.
              It does <b>not</b> measure ball contact, club face, or shot outcome — a swing can score
              well and still miss the ball.
            </p>
          </div>
        </div>
      )}

      {/* Overall summary */}
      <div className={styles.summary}>
        <p>{overall_assessment}</p>
      </div>

      {/* Camera note (methodology transparency) */}
      {camera_note && (
        <div className={styles.cameraNote}>
          <span className={styles.cameraNoteIcon}>ⓘ</span>
          <span>
            {camera_note}
            {handedness && handedness !== 'unknown' && ` Detected as ${handedness}-handed.`}
          </span>
        </div>
      )}

      {/* Overlay video */}
      {overlay_video_url && (
        <section className={styles.metricsCard}>
          <div className={styles.scrubberHeader}>
            <h3 className={styles.sectionTitle}>Annotated Swing Video</h3>
            <div className={styles.videoControls}>
              {[0.25, 0.5, 1].map(rate => (
                <button
                  key={rate}
                  className={`${styles.speedBtn} ${playbackRate === rate ? styles.speedBtnActive : ''}`}
                  onClick={() => setPlaybackRate(rate)}
                >
                  {rate}x
                </button>
              ))}
              <a
                href={`${API_URL}${overlay_video_url}`}
                download={`swing_analysis${overlay_video_url.endsWith('.webm') ? '.webm' : '.mp4'}`}
                className={styles.downloadBtn}
              >
                ↓ Download
              </a>
            </div>
          </div>
          <video
            ref={overlayVideoRef}
            src={`${API_URL}${overlay_video_url}`}
            controls
            autoPlay
            loop
            muted
            playsInline
            className={styles.overlayVideo}
          />
          <p className={styles.planeCaption}>
            Your swing with skeleton, reference lines, angle overlay, and detected club shaft.
            Gold = spine axis, blue = shoulders, orange = hips, cyan = golf shaft with clubhead dot.
          </p>
        </section>
      )}

      {/* Phase scorecard */}
      <PhaseScorecard scores={phase_scores} />

      {/* Tempo Card */}
      {tempo && (
        <section className={styles.metricsCard}>
          <h3 className={styles.sectionTitle}>Swing Tempo</h3>
          <TempoBar tempo={tempo} />
        </section>
      )}

      {/* Measured Metrics */}
      {(metrics_summary.address_spine_tilt_deg !== undefined) && (
        <section className={styles.metricsCard}>
          <h3 className={styles.sectionTitle}>Measured Metrics</h3>
          <div className={styles.metricsGrid}>
            <MetricStat
              label="Address Spine Tilt"
              value={metrics_summary.address_spine_tilt_deg}
              unit="°"
              hint="Target ~30°"
            />
            <MetricStat
              label="Impact Spine Tilt"
              value={metrics_summary.impact_spine_tilt_deg}
              unit="°"
              hint="Should match address"
            />
            <MetricStat
              label="X-Factor at Top"
              value={metrics_summary.top_x_factor_deg}
              unit="°"
              hint="Higher = more coil"
            />
            <MetricStat
              label="Head Movement"
              value={metrics_summary.head_movement_pct}
              unit="%"
              hint="Lower is better"
            />
          </div>
        </section>
      )}

      {/* Shaft Plane */}
      {shaft_plane && (
        <section className={styles.metricsCard}>
          <h3 className={styles.sectionTitle}>Shaft Plane</h3>
          <div className={styles.shaftPlaneRow}>
            <MetricStat
              label="Backswing plane"
              value={shaft_plane.backswing_plane_angle}
              unit="°"
            />
            <MetricStat
              label="Downswing plane"
              value={shaft_plane.downswing_plane_angle}
              unit="°"
            />
            <MetricStat
              label="Plane difference"
              value={shaft_plane.plane_delta}
              unit="°"
              hint="Lower = more consistent"
            />
          </div>
          <p className={styles.tempoVerdict} data-rating={shaft_plane.rating}>
            {shaft_plane.verdict}
          </p>
        </section>
      )}

      {/* Swing Plane Trace */}
      {swing_plane_image && (
        <section className={styles.metricsCard}>
          <h3 className={styles.sectionTitle}>Swing Plane Trace</h3>
          <img
            src={`data:image/jpeg;base64,${swing_plane_image}`}
            alt="Wrist trajectory across the swing"
            className={styles.planeImage}
          />
          <p className={styles.planeCaption}>
            Grip and clubhead paths across the swing. Overlapping backswing (blue) and downswing (orange)
            grip lines suggest an on-plane swing; the thin cyan trace shows where the clubhead traveled.
          </p>
        </section>
      )}

      {/* Practice Plan */}
      {practice_plan.length > 0 && practice_plan[0].severity !== 'low' && (
        <section className={styles.practicePlan}>
          <h3 className={styles.sectionTitle}>Your Practice Plan</h3>
          <ol className={styles.planList}>
            {practice_plan.map(item => (
              <li key={item.priority} className={styles.planItem} data-sev={item.severity}>
                <div className={styles.planIndex}>{item.priority}</div>
                <div className={styles.planBody}>
                  <div className={styles.planHeader}>
                    <span className={styles.planFocus}>{item.focus}</span>
                    {item.phase && <span className={styles.planPhase}>{item.phase}</span>}
                  </div>
                  <p className={styles.planDrill}>
                    <span className={styles.drillLabel}>Drill: </span>
                    {item.drill}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}

      {/* Key Focus */}
      {key_focus && (
        <div className={styles.keyFocus}>
          <span className={styles.keyFocusLabel}>Key Focus</span>
          <p>{key_focus}</p>
        </div>
      )}

      {/* Frame scrubber */}
      {frames.length > 0 && current && (
        <section className={styles.section}>
          <div className={styles.scrubberHeader}>
            <h3 className={styles.sectionTitle}>Swing Breakdown — Frame by Frame</h3>
            <div className={styles.videoControls}>
              <button
                className={`${styles.speedBtn} ${compareFrame !== null ? styles.speedBtnActive : ''}`}
                onClick={() => setCompareFrame(compareFrame === null ? activeFrame : null)}
                title="Compare with another frame"
              >
                {compareFrame !== null ? '✕ Exit Compare' : '⇔ Compare'}
              </button>
              <button
                className={styles.playBtn}
                onClick={() => setAutoPlay((p) => !p)}
                aria-label={autoPlay ? 'Pause playback' : 'Play through swing'}
              >
                {autoPlay ? '❚❚ Pause' : '▶ Play'}
              </button>
            </div>
          </div>
          <p className={styles.kbdHint}>Tip: ← → to step frames, space to play/pause</p>

          <div className={styles.scrubber}>
            {compareFrame !== null && compare ? (
              <div className={styles.compareWrapper}>
                <div className={styles.compareCol}>
                  <div className={styles.compareLabel}>
                    <span style={{ color: phaseColor(current.phase) }}>{current.phase}</span> · Frame {activeFrame + 1}
                  </div>
                  <img src={`data:image/jpeg;base64,${current.thumbnail}`} className={styles.compareImg} alt={current.phase} />
                  {current.metrics && (
                    <div className={styles.miniMetrics}>
                      <span><b>Spine:</b> {current.metrics.spine_tilt_3d ?? current.metrics.spine_tilt}°</span>
                      <span><b>X-Factor:</b> {current.metrics.x_factor_3d ?? current.metrics.x_factor}°</span>
                    </div>
                  )}
                </div>
                <div className={styles.compareCol}>
                  <div className={styles.compareLabel}>
                    <span style={{ color: phaseColor(compare.phase) }}>{compare.phase}</span> · Frame {compareFrame + 1}
                  </div>
                  <img src={`data:image/jpeg;base64,${compare.thumbnail}`} className={styles.compareImg} alt={compare.phase} />
                  {compare.metrics && (
                    <div className={styles.miniMetrics}>
                      <span><b>Spine:</b> {compare.metrics.spine_tilt_3d ?? compare.metrics.spine_tilt}°</span>
                      <span><b>X-Factor:</b> {compare.metrics.x_factor_3d ?? compare.metrics.x_factor}°</span>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className={styles.scrubberViewer}>
                {current.thumbnail && (
                  <img
                    src={`data:image/jpeg;base64,${current.thumbnail}`}
                    className={styles.scrubberImg}
                    alt={current.phase}
                  />
                )}
                <div className={styles.scrubberOverlay}>
                  <span
                    className={styles.phaseTag}
                    style={{ background: phaseColor(current.phase) + '22', color: phaseColor(current.phase), borderColor: phaseColor(current.phase) + '44' }}
                  >
                    {current.phase}
                  </span>
                  <span className={styles.frameCounter}>
                    Frame {activeFrame + 1} / {frames.length}
                  </span>
                </div>
              </div>
            )}

            <div className={styles.scrubberInfo}>
              <p className={styles.observation}>{current.observation}</p>
              {current.metrics && (
                <div className={styles.miniMetrics}>
                  <span><b>Spine:</b> {current.metrics.spine_tilt}°</span>
                  <span><b>X-Factor:</b> {current.metrics.x_factor}°</span>
                  <span><b>L-Knee:</b> {current.metrics.left_knee_angle}°</span>
                  <span><b>R-Knee:</b> {current.metrics.right_knee_angle}°</span>
                </div>
              )}
            </div>

            <div className={styles.timeline}>
              {frames.map((f, i) => {
                const isActive = i === activeFrame
                const isCompare = i === compareFrame
                let borderColor = 'transparent'
                if (isActive) borderColor = phaseColor(f.phase)
                else if (isCompare) borderColor = '#c9a84c'
                return (
                  <button
                    key={i}
                    className={`${styles.timelineFrame} ${isActive ? styles.timelineActive : ''} ${isCompare ? styles.timelineCompareMarked : ''}`}
                    onClick={(e) => {
                      if (e.shiftKey && compareFrame !== null) {
                        setCompareFrame(i)
                      } else {
                        setActiveFrame(i)
                        setAutoPlay(false)
                      }
                    }}
                    title={compareFrame !== null ? 'Click to set primary frame, Shift+Click to set compare frame' : f.phase}
                    style={{ borderColor }}
                  >
                    <img src={`data:image/jpeg;base64,${f.thumbnail}`} alt={`Frame ${i + 1}`} />
                    <span
                      className={styles.timelineDot}
                      style={{ background: phaseColor(f.phase) }}
                      title={f.phase}
                    />
                  </button>
                )
              })}
            </div>
          </div>
        </section>
      )}

      {/* Strengths + Improvements side by side */}
      <div className={styles.twoCol}>
        {strengths.length > 0 && (
          <section className={`${styles.section} ${styles.card}`}>
            <h3 className={styles.sectionTitle}>
              <span className={styles.greenDot} /> Strengths
            </h3>
            <ul className={styles.list}>
              {strengths.map((s, i) => (
                <li key={i} className={styles.strengthItem}>{s}</li>
              ))}
            </ul>
          </section>
        )}

        {improvements.length > 0 && (
          <section className={`${styles.section} ${styles.card}`}>
            <h3 className={styles.sectionTitle}>
              <span className={styles.goldDot} /> Areas to Improve
            </h3>
            <div className={styles.improvements}>
              {improvements.map((imp, i) => (
                <div key={i} className={styles.improvementItem}>
                  <p className={styles.issue}>{imp.issue}</p>
                  <p className={styles.drill}>
                    <span className={styles.drillLabel}>Drill: </span>
                    {imp.drill}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>

      {/* Video replay */}
      {videoPreview && (
        <section className={styles.section}>
          <h3 className={styles.sectionTitle}>Your Swing</h3>
          <video src={videoPreview} controls className={styles.video} playsInline />
        </section>
      )}
    </div>
  )
}
