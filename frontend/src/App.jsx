import { useState } from 'react'
import VideoUpload from './components/VideoUpload'
import AnalysisResults from './components/AnalysisResults'
import SwingHistory from './components/SwingHistory'
import styles from './App.module.css'

const API_URL = 'http://localhost:8000'

export default function App() {
  const [state, setState] = useState('idle') // idle | uploading | analyzing | done | error
  const [analysis, setAnalysis] = useState(null)
  const [error, setError] = useState(null)
  const [videoPreview, setVideoPreview] = useState(null)

  async function handleVideoSelected(file) {
    setError(null)
    setAnalysis(null)
    setVideoPreview(URL.createObjectURL(file))
    setState('uploading')

    const form = new FormData()
    form.append('video', file)

    try {
      setState('analyzing')
      const res = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: form,
      })

      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || `Server error ${res.status}`)
      }

      setAnalysis(data.analysis)
      setState('done')
    } catch (err) {
      setError(err.message)
      setState('error')
    }
  }

  function handleReset() {
    setState('idle')
    setAnalysis(null)
    setError(null)
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview)
      setVideoPreview(null)
    }
  }

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>⛳</span>
          <span className={styles.logoText}>Golf<strong>AI</strong></span>
        </div>
        <p className={styles.tagline}>Upload your swing. Get pro-level feedback.</p>
      </header>

      <main className={styles.main}>
        {(state === 'idle' || state === 'uploading' || state === 'analyzing') && (
          <>
            <VideoUpload
              onVideoSelected={handleVideoSelected}
              state={state}
              videoPreview={videoPreview}
            />
            {state === 'idle' && <SwingHistory />}
          </>
        )}

        {state === 'error' && (
          <div className={styles.errorCard}>
            <span className={styles.errorIcon}>!</span>
            <div>
              <h3>Something went wrong</h3>
              <p>{error}</p>
            </div>
            <button className={styles.retryBtn} onClick={handleReset}>Try Again</button>
          </div>
        )}

        {state === 'done' && analysis && (
          <AnalysisResults
            analysis={analysis}
            videoPreview={videoPreview}
            onReset={handleReset}
          />
        )}
      </main>

      <footer className={styles.footer}>
        <p>Powered by MediaPipe pose detection &mdash; for practice use only, not a substitute for professional coaching</p>
      </footer>
    </div>
  )
}
