import { useState, useEffect, useRef, useCallback, Fragment } from 'react'
import axios from 'axios'
import {
  ComposedChart, Line, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import './App.css'

// ── Constants ─────────────────────────────────────────────────────────────────

const API_BASE = ''
const FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm']

const PIPELINE_STEPS = [
  { id: 'upload',   label: 'Upload',     phase: 0 },
  { id: 'asr',      label: 'Transcribe', phase: 1 },
  { id: 'diarize',  label: 'Diarize',    phase: 2 },
  { id: 'emotion',  label: 'SER',        phase: 3 },
  { id: 'laptime',  label: 'Lap Chart',  phase: 4 },
]

// Full SER model label mapping
const EMOTION_META = {
  angry:     { emoji: '😠', label: 'Angry',     color: '#ef4444' },
  anger:     { emoji: '😠', label: 'Angry',     color: '#ef4444' },
  disgust:   { emoji: '🤢', label: 'Disgust',   color: '#10b981' },
  fear:      { emoji: '😨', label: 'Fear',      color: '#8b5cf6' },
  fearful:   { emoji: '😨', label: 'Fear',      color: '#8b5cf6' },
  happy:     { emoji: '😊', label: 'Happy',     color: '#a78bfa' },
  happiness: { emoji: '😊', label: 'Happy',     color: '#a78bfa' },
  neutral:   { emoji: '😐', label: 'Neutral',   color: '#6b7280' },
  sad:       { emoji: '😢', label: 'Sad',       color: '#3b82f6' },
  sadness:   { emoji: '😢', label: 'Sad',       color: '#3b82f6' },
  calm:      { emoji: '😌', label: 'Calm',      color: '#00d2be' },
  surprised: { emoji: '😲', label: 'Surprised', color: '#f59e0b' },
  surprise:  { emoji: '😲', label: 'Surprised', color: '#f59e0b' },
  // Short code fallbacks
  ang: { emoji: '😠', label: 'Angry',   color: '#ef4444' },
  hap: { emoji: '😊', label: 'Happy',   color: '#a78bfa' },
  sad: { emoji: '😢', label: 'Sad',     color: '#3b82f6' },
  neu: { emoji: '😐', label: 'Neutral', color: '#6b7280' },
}

function getEmotionMeta(label) {
  if (!label) return { emoji: '❓', label: label || '—', color: '#6b7280' }
  const key = label.toLowerCase().trim()
  return EMOTION_META[key] ?? { emoji: '❓', label, color: '#6b7280' }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function fmtTime(secs) {
  const d = new Date(secs * 1000)
  return d.toISOString().substring(14, 19)
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusDot({ ok, loading }) {
  if (loading) return <span className="dot amber pulse" />
  return <span className={`dot ${ok ? 'green' : 'red'}`} />
}

function PanelHeader({ icon, title, phase, right }) {
  return (
    <div className="panel-header">
      <span className="panel-title">
        <span className="panel-title-icon">{icon}</span>
        {title}
      </span>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {right}
        {phase != null && <span className="panel-phase">PH-{phase}</span>}
      </div>
    </div>
  )
}

function PlaceholderPanel({ icon, title, phase, description }) {
  return (
    <div className="card">
      <PanelHeader icon={icon} title={title} phase={phase} />
      <div className="placeholder-content">
        <span className="placeholder-icon">{icon}</span>
        <p className="placeholder-text">{description}</p>
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [backendOk, setBackendOk] = useState(null)
  const [backendVersion, setBackendVersion] = useState(null)

  const [file, setFile] = useState(null)
  const [fileUrl, setFileUrl] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  const [transcript, setTranscript] = useState(null)
  const [transcribing, setTranscribing] = useState(false)

  const [diarization, setDiarization] = useState(null)
  const [diarizing, setDiarizing] = useState(false)

  const [emotion, setEmotion] = useState(null)
  const [emotioning, setEmotioning] = useState(false)

  const [lapTimes, setLapTimes] = useState(null)
  const [laptiming, setLaptiming] = useState(false)
  const [lapDataUploading, setLapDataUploading] = useState(false)

  const fileInputRef = useRef(null)
  const telemetryInputRef = useRef(null)

  // ── Health check ────────────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false
    async function checkHealth() {
      try {
        const res = await axios.get(`${API_BASE}/health`, { timeout: 5000 })
        if (!cancelled) { setBackendOk(true); setBackendVersion(res.data.version) }
      } catch { if (!cancelled) setBackendOk(false) }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 10_000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  // ── Drag & drop ─────────────────────────────────────────────────────────────
  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) handleFileSelected(dropped)
  }, [])

  const handleDragOver  = useCallback((e) => { e.preventDefault(); setDragOver(true) }, [])
  const handleDragLeave = useCallback(() => setDragOver(false), [])

  // ── File selection ──────────────────────────────────────────────────────────
  function handleFileSelected(selectedFile) {
    const ext = '.' + selectedFile.name.split('.').pop().toLowerCase()
    if (!FORMATS.includes(ext)) {
      setUploadError(`Unsupported format "${ext}". Supported: ${FORMATS.join(', ')}`)
      return
    }
    setUploadError(null)
    setFile(selectedFile)
    setSessionId(null); setTranscript(null); setDiarization(null)
    setEmotion(null); setLapTimes(null)
    if (fileUrl) URL.revokeObjectURL(fileUrl)
    setFileUrl(URL.createObjectURL(selectedFile))
  }

  // ── Upload ──────────────────────────────────────────────────────────────────
  async function handleUpload() {
    if (!file) return
    setUploading(true); setUploadError(null); setSessionId(null)
    const fd = new FormData(); fd.append('file', file)
    try {
      const res = await axios.post(`${API_BASE}/upload-audio`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' }, timeout: 30_000,
      })
      setSessionId(res.data.session_id)
    } catch (err) {
      setUploadError(err.response?.data?.detail || err.message || 'Upload failed')
    } finally { setUploading(false) }
  }

  // ── Transcribe ──────────────────────────────────────────────────────────────
  async function handleTranscribe() {
    if (!sessionId) return
    setTranscribing(true); setUploadError(null)
    try {
      const res = await axios.post(`${API_BASE}/transcribe?session_id=${sessionId}`, null, { timeout: 300_000 })
      setTranscript(res.data)
    } catch (err) {
      setUploadError(err.response?.data?.detail || err.message || 'Transcription failed')
    } finally { setTranscribing(false) }
  }

  // ── Diarize ─────────────────────────────────────────────────────────────────
  async function handleDiarize() {
    if (!sessionId) return
    setDiarizing(true); setUploadError(null)
    try {
      const res = await axios.post(`${API_BASE}/diarize?session_id=${sessionId}`, null, { timeout: 300_000 })
      setDiarization(res.data)
    } catch (err) {
      setUploadError(err.response?.data?.detail || err.message || 'Diarization failed')
    } finally { setDiarizing(false) }
  }

  // ── Emotion / SER ───────────────────────────────────────────────────────────
  async function handleEmotion() {
    if (!sessionId) return
    setEmotioning(true); setUploadError(null)
    try {
      const res = await axios.post(`${API_BASE}/analyze-stress`, {
        session_id: sessionId,
        segments: diarization?.segments || null,
      }, { timeout: 300_000 })
      setEmotion(res.data)
    } catch (err) {
      setUploadError(err.response?.data?.detail || err.message || 'Emotion analysis failed')
    } finally { setEmotioning(false) }
  }

  // ── Lap telemetry upload ────────────────────────────────────────────────────
  async function handleLapTelemetryUpload(e) {
    const f = e.target.files[0]; if (!f) return
    setLapDataUploading(true); setUploadError(null)
    const fd = new FormData(); fd.append('file', f)
    try {
      await axios.post(`${API_BASE}/upload-lap-data`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' }, timeout: 30_000,
      })
      await handleLapTiming()
    } catch (err) {
      setUploadError(err.response?.data?.detail || err.message || 'Telemetry upload failed')
    } finally { setLapDataUploading(false) }
  }

  // ── Lap timing ──────────────────────────────────────────────────────────────
  async function handleLapTiming() {
    setLaptiming(true); setUploadError(null)
    try {
      const res = await axios.get(`${API_BASE}/lap-times`, { timeout: 30_000 })
      const lapsWithStress = res.data.laps.map(lap => {
        const inLap = emotion?.segments
          ? emotion.segments.filter(s => s.start >= lap.timestamp && s.start < lap.timestamp + lap.lap_time)
          : []
        const avgStress = inLap.length > 0
          ? inLap.reduce((sum, s) => sum + s.score, 0) / inLap.length : 0
        return { ...lap, stress: avgStress }
      })
      setLapTimes({ ...res.data, laps: lapsWithStress })
    } catch (err) {
      setUploadError(err.response?.data?.detail || err.message || 'Lap timing failed')
    } finally { setLaptiming(false) }
  }

  // ── Button state ────────────────────────────────────────────────────────────
  const isProcessing = uploading || transcribing || diarizing || emotioning || laptiming
  const currentStep  = lapTimes ? 5 : emotion ? 4 : diarization ? 3 : transcript ? 2 : sessionId ? 1 : 0

  function nextAction() {
    if (lapTimes)    return null
    if (emotion)     return handleLapTiming
    if (diarization) return handleEmotion
    if (transcript)  return handleDiarize
    if (sessionId)   return handleTranscribe
    return handleUpload
  }

  function nextLabel() {
    if (lapTimes)    return <><span>✓</span> Complete</>
    if (emotion)     return <><span>📈</span> Lap Chart</>
    if (diarization) return <><span>🧠</span> Analyze Stress</>
    if (transcript)  return <><span>👥</span> Diarize</>
    if (sessionId)   return <><span>🎙</span> Transcribe</>
    return <><span>↑</span> Upload</>
  }

  function busyLabel() {
    if (uploading)   return <><span className="spinner"/> Uploading…</>
    if (transcribing)return <><span className="spinner"/> Transcribing…</>
    if (diarizing)   return <><span className="spinner"/> Diarizing…</>
    if (emotioning)  return <><span className="spinner"/> Analyzing…</>
    if (laptiming)   return <><span className="spinner"/> Loading…</>
    return null
  }

  // Build combined radio-feed rows (diarization segments joined with transcript text)
  function buildRadioFeed() {
    if (!diarization?.segments) return null
    return diarization.segments.map((seg, i) => {
      const role = seg.role ?? (seg.speaker === 'SPEAKER_00' ? 'driver' : 'engineer')
      // Find overlapping transcript text
      const text = transcript?.segments
        ?.filter(t => !(t.end <= seg.start || t.start >= seg.end))
        ?.map(t => t.text.trim())
        ?.join(' ') ?? ''
      return { ...seg, role, text: text || '—', key: i }
    })
  }

  const radioFeed = buildRadioFeed()

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <>
      {/* ── Header ── */}
      <header className="header">
        <div className="container">
          <div className="header-inner">
            <div className="logo">
              <div className="logo-icon">🏎</div>
              <div className="logo-text">
                <span className="logo-title">StressLine</span>
                <span className="logo-sub">Race Radio AI</span>
              </div>
            </div>

            <div className="status-bar">
              <div className="status-item">
                <StatusDot ok={backendOk} loading={backendOk === null} />
                <span>
                  {backendOk === null ? 'connecting…'
                    : backendOk ? `api v${backendVersion ?? '?'}`
                    : 'offline'}
                </span>
              </div>
              {sessionId && (
                <div className="live-indicator">
                  <span className="dot orange pulse" />
                  live session
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* ── Session bar (only when session active) ── */}
      {sessionId && (
        <div className="session-bar">
          <div className="container">
            <div className="session-bar-inner">
              <span className="session-label">Active Session</span>
              <span className="session-filename">{file?.name ?? sessionId}</span>
              <div className="pipeline">
                {PIPELINE_STEPS.map((step, i) => (
                  <Fragment key={step.id}>
                    <div className={`pipeline-step ${
                      step.phase < currentStep ? 'done' :
                      step.phase === currentStep ? 'active' : ''
                    }`}>
                      {step.phase < currentStep ? '✓ ' : ''}{step.label}
                    </div>
                    {i < PIPELINE_STEPS.length - 1 && (
                      <span className="pipeline-arrow">→</span>
                    )}
                  </Fragment>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <main>
        {/* ── Hero / Upload — shown until session active ── */}
        {!sessionId && (
          <>
            <section className="hero">
              <div className="container">
                <div className="hero-eyebrow">
                  <span>🏁</span> F1 Radio Analysis · Hackathon MVP
                </div>
                <h1 className="hero-title">
                  Hear What the <span className="accent">Data</span> Can't
                </h1>
                <p className="hero-subtitle">
                  Upload race radio audio. Detect driver stress from voice.
                  Correlate with lap times.
                </p>
              </div>
            </section>

            <section className="upload-section">
              <div className="container">
                <div
                  className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
                  onClick={() => fileInputRef.current?.click()}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  role="button"
                  tabIndex={0}
                  id="upload-zone"
                  onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
                  aria-label="Upload audio file"
                >
                  <span className="upload-icon">🎙</span>
                  <p className="upload-title">
                    {dragOver ? 'Drop it here!' : 'Drop race radio audio here'}
                  </p>
                  <p className="upload-hint">Or click to browse — cockpit audio, race radio, comms</p>
                  <div className="upload-formats">
                    {FORMATS.map(f => <span key={f} className="format-tag">{f}</span>)}
                  </div>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept={FORMATS.join(',')}
                  style={{ display: 'none' }}
                  onChange={(e) => e.target.files[0] && handleFileSelected(e.target.files[0])}
                  id="file-input"
                />

                {file && (
                  <div className="file-preview">
                    <div className="file-icon">🎧</div>
                    <div className="file-info">
                      <div className="file-name">{file.name}</div>
                      <div className="file-meta">{formatBytes(file.size)} · {file.type || 'audio'}</div>
                      {fileUrl && (
                        <audio controls src={fileUrl} className="audio-player" id="audio-preview" />
                      )}
                    </div>
                    <button
                      className="btn btn-primary"
                      onClick={handleUpload}
                      disabled={uploading || !backendOk}
                      id="upload-btn"
                      title={!backendOk ? 'Backend offline' : ''}
                    >
                      {isProcessing ? busyLabel() : nextLabel()}
                    </button>
                  </div>
                )}

                {uploadError && (
                  <div className="error-banner">⚠ {uploadError}</div>
                )}
              </div>
            </section>
          </>
        )}

        {/* ── Dashboard — shown once session is active ── */}
        {sessionId && (
          <section className="dashboard">
            <div className="container">

              {/* File preview + action button row */}
              <div className="file-preview" style={{ maxWidth: '100%', marginBottom: 'var(--sp-4)' }}>
                <div className="file-icon">🎧</div>
                <div className="file-info">
                  <div className="file-name">{file?.name}</div>
                  {fileUrl && (
                    <audio controls src={fileUrl} className="audio-player" id="audio-preview" />
                  )}
                </div>
                {!lapTimes && (
                  <button
                    className="btn btn-primary"
                    onClick={nextAction()}
                    disabled={isProcessing || !backendOk}
                    id="action-btn"
                  >
                    {isProcessing ? busyLabel() : nextLabel()}
                  </button>
                )}
              </div>

              {uploadError && (
                <div className="error-banner" style={{ maxWidth: '100%', marginBottom: 'var(--sp-4)' }}>
                  ⚠ {uploadError}
                </div>
              )}

              {/* Dashboard 2-col grid */}
              <div className="dashboard-grid">

                {/* ─── LEFT COLUMN ─── */}
                <div className="col-left">

                  {/* RADIO FEED panel — diarization + transcript merged */}
                  {(radioFeed || transcript) ? (
                    <div className="card animate-fade-in">
                      <PanelHeader
                        icon="📡"
                        title="Radio Feed"
                        phase={radioFeed ? 2 : 1}
                      />
                      <div className="radio-feed">
                        {radioFeed ? (
                          radioFeed.map(line => (
                            <div key={line.key} className="radio-line">
                              <span className="radio-ts">{fmtTime(line.start)}</span>
                              <span className={`radio-speaker-tag ${line.role}`}>
                                {line.role === 'engineer' ? 'Engineer' : 'Driver'}
                              </span>
                              <span className="radio-text">{line.text}</span>
                            </div>
                          ))
                        ) : (
                          transcript?.segments?.map((seg, i) => (
                            <div key={i} className="radio-line">
                              <span className="radio-ts">{fmtTime(seg.start)}</span>
                              <span className="radio-speaker-tag driver">—</span>
                              <span className="radio-text">{seg.text}</span>
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  ) : (
                    <PlaceholderPanel
                      icon="📡"
                      title="Radio Feed"
                      phase={1}
                      description="Transcript and speaker turns will appear here after transcription"
                    />
                  )}

                  {/* STRESS / EMOTION panel */}
                  {emotion ? (
                    <div className="card teal-accent animate-fade-in">
                      <PanelHeader icon="🧠" title="Stress Monitor" phase={3} />
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0', maxHeight: '220px', overflowY: 'auto', paddingRight: '2px' }}>
                        {emotion.segments.map((seg, i) => {
                          const meta = getEmotionMeta(seg.emotion)
                          const role = seg.role ?? (seg.speaker === 'SPEAKER_00' ? 'driver' : 'engineer')
                          return (
                            <div key={i} className="emotion-row">
                              <span className="radio-ts">{fmtTime(seg.start)}</span>
                              <span className={`radio-speaker-tag ${role}`}>
                                {role === 'engineer' ? 'Engineer' : 'Driver'}
                              </span>
                              <div>
                                <div className="emotion-label" style={{ color: meta.color }}>
                                  {meta.emoji} {meta.label}
                                </div>
                                <div className="emotion-bar-wrap" style={{ marginTop: '4px' }}>
                                  <div
                                    className="emotion-bar"
                                    style={{
                                      width: `${(seg.score * 100).toFixed(0)}%`,
                                      background: meta.color,
                                    }}
                                  />
                                </div>
                              </div>
                              <span className="emotion-score">{(seg.score * 100).toFixed(0)}%</span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  ) : (
                    <PlaceholderPanel
                      icon="🧠"
                      title="Stress Monitor"
                      phase={3}
                      description="Per-segment emotion labels with confidence will appear here"
                    />
                  )}
                </div>

                {/* ─── RIGHT COLUMN ─── */}
                <div className="col-right">
                  {lapTimes ? (
                    <div className="card chart-panel animate-fade-in" style={{ height: '100%' }}>
                      <PanelHeader
                        icon="📈"
                        title="Telemetry"
                        phase={4}
                        right={
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span className={`chart-source-badge ${lapTimes.source !== 'uploaded' ? 'default' : ''}`}>
                              {lapTimes.source === 'uploaded' ? '📁 Custom' : '📊 Sample'}
                            </span>
                            <>
                              <input
                                ref={telemetryInputRef}
                                type="file"
                                accept=".csv,.json"
                                style={{ display: 'none' }}
                                onChange={handleLapTelemetryUpload}
                              />
                              <button
                                className="btn btn-secondary"
                                onClick={() => telemetryInputRef.current?.click()}
                                disabled={lapDataUploading}
                              >
                                {lapDataUploading ? 'Uploading…' : '↑ CSV/JSON'}
                              </button>
                            </>
                          </div>
                        }
                      />
                      <div style={{ width: '100%', height: '260px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart data={lapTimes.laps} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
                            <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                              dataKey="lap"
                              stroke="#4a5060"
                              tick={{ fontSize: 10, fontFamily: 'JetBrains Mono' }}
                              label={{ value: 'Lap', position: 'insideBottomRight', offset: -5, fill: '#4a5060', fontSize: 10 }}
                            />
                            <YAxis
                              yAxisId="left"
                              stroke="#4a5060"
                              tick={{ fontSize: 10, fontFamily: 'JetBrains Mono' }}
                              label={{ value: 'Lap Time (s)', angle: -90, position: 'insideLeft', fill: '#4a5060', fontSize: 10 }}
                            />
                            <YAxis
                              yAxisId="right"
                              orientation="right"
                              domain={[0, 1]}
                              stroke="#4a5060"
                              tick={{ fontSize: 10, fontFamily: 'JetBrains Mono' }}
                              label={{ value: 'Stress', angle: 90, position: 'insideRight', fill: '#4a5060', fontSize: 10 }}
                            />
                            <Tooltip
                              contentStyle={{
                                background: 'rgba(17,19,22,0.95)',
                                border: '1px solid rgba(255,128,0,0.3)',
                                borderRadius: '4px',
                                fontSize: '11px',
                                fontFamily: 'JetBrains Mono',
                              }}
                              labelStyle={{ color: '#f0f2f5' }}
                            />
                            <Legend wrapperStyle={{ fontSize: '10px', fontFamily: 'JetBrains Mono' }} />
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="lap_time"
                              stroke="#ff8000"
                              name="Lap Time"
                              strokeWidth={2}
                              dot={{ r: 3, fill: '#ff8000' }}
                              activeDot={{ r: 5 }}
                            />
                            <Scatter
                              yAxisId="right"
                              name="Stress"
                              dataKey="stress"
                              fill="#00d2be"
                            />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Lap summary table */}
                      <div style={{ marginTop: 'var(--sp-4)', borderTop: '1px solid var(--clr-border)', paddingTop: 'var(--sp-3)' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px 8px' }}>
                          {lapTimes.laps.map((lap, i) => {
                            const inLap = emotion?.segments
                              ?.filter(s => s.start >= lap.timestamp && s.start < lap.timestamp + lap.lap_time) ?? []
                            const avg = inLap.length > 0
                              ? inLap.reduce((s, seg) => s + seg.score, 0) / inLap.length : 0
                            const hi = avg > 0.65
                            return (
                              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', fontFamily: 'JetBrains Mono', padding: '3px 6px', background: 'var(--clr-surface-2)', borderRadius: '3px' }}>
                                <span style={{ color: 'var(--clr-text-muted)' }}>L{lap.lap}</span>
                                <span style={{ color: 'var(--clr-text-secondary)' }}>{lap.lap_time.toFixed(2)}s</span>
                                <span style={{ color: hi ? '#ef4444' : '#00d2be' }}>{(avg * 100).toFixed(0)}%</span>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <PlaceholderPanel
                      icon="📈"
                      title="Telemetry"
                      phase={4}
                      description="Lap time + stress overlay chart — complete SER analysis first"
                    />
                  )}
                </div>

              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <div className="container">
          StressLine · Hackathon MVP · React + FastAPI + Hugging Face 🤗
        </div>
      </footer>
    </>
  )
}
