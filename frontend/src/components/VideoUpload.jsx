import { useRef, useState } from 'react'
import styles from './VideoUpload.module.css'

const ACCEPTED = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska']

export default function VideoUpload({ onVideoSelected, state, videoPreview }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const isLoading = state === 'uploading' || state === 'analyzing'

  function handleFile(file) {
    if (!file) return
    if (!ACCEPTED.includes(file.type) && !file.name.match(/\.(mp4|mov|avi|webm|mkv)$/i)) {
      alert('Please upload a video file (MP4, MOV, AVI, WebM, MKV)')
      return
    }
    onVideoSelected(file)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  function onDragOver(e) {
    e.preventDefault()
    setDragOver(true)
  }

  function onDragLeave() {
    setDragOver(false)
  }

  return (
    <div className={styles.wrapper}>
      {!videoPreview && !isLoading && (
        <div
          className={`${styles.dropzone} ${dragOver ? styles.active : ''}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className={styles.hiddenInput}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          <div className={styles.icon}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M15 10l4.553-2.069A1 1 0 0121 8.845v6.31a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/>
            </svg>
          </div>
          <h2 className={styles.heading}>Drop your swing video here</h2>
          <p className={styles.sub}>or click to browse &mdash; MP4, MOV, AVI, WebM up to 100 MB</p>
          <button className={styles.browseBtn} tabIndex={-1} type="button">
            Choose Video
          </button>
        </div>
      )}

      {videoPreview && (
        <div className={styles.previewSection}>
          <video
            src={videoPreview}
            className={styles.video}
            controls
            playsInline
          />
          {isLoading && (
            <div className={styles.overlay}>
              <div className={styles.spinner} />
              <p className={styles.loadingText}>
                {state === 'uploading' ? 'Uploading video...' : 'Analysing your swing with AI...'}
              </p>
              <p className={styles.loadingHint}>This takes about 15–30 seconds</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
