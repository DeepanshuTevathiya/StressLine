# The Silent Co-Driver — Implementation Plan

## Overview

A hackathon MVP that detects driver stress/fatigue from race radio audio and correlates it with lap time data. Full-stack: React frontend + FastAPI backend, all models from Hugging Face Hub, deployed to HF Spaces.

---

## Phase 0 — Setup & Environment (CURRENT)

### Project Structure
```
e:\StressLine\
├── backend/               # FastAPI (Python)
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/              # React (Vite)
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── .gitignore
└── README.md
```

### Backend Stack
- Python venv at `e:\StressLine\backend\.venv`
- FastAPI + Uvicorn
- `transformers`, `torch` (CPU for now), `pyannote.audio`
- `python-multipart` for file uploads
- CORS enabled for local React dev server

### Frontend Stack
- Vite + React (TypeScript optional, skipping for hackathon speed)
- Axios for API calls
- Recharts for the lap-time chart (Phase 4)
- Basic proxy config to FastAPI (`:8000`)

### Health-Check Verification
- `GET /health` on FastAPI returns `{"status": "ok"}`
- React fetches it and shows "Backend connected ✓"

---

## Phase 1 — ASR (Whisper)

### Model Choice
- `openai/whisper-base` — ~74M params, fast on CPU, no gating, permissive license
- Will upgrade to `whisper-small` if base accuracy is insufficient (still HF Spaces free-tier safe)

**Endpoint:** `POST /transcribe` — accepts audio file, returns JSON transcript

---

## Phase 2 — Diarization (pyannote)

> [!IMPORTANT]
> `pyannote/speaker-diarization-3.1` requires accepting terms on HF Hub AND an HF access token. Will flag immediately when we get here and ask the user for the token. If gating is problematic, fallback is `pyannote/speaker-diarization` (older, less gated) or simple energy-based VAD with generic Speaker A/B labels.

---

## Phase 3 — Speech Emotion Recognition

### Candidate Models (to confirm with user before implementing)
| Model | Size | Labels | Notes |
|---|---|---|---|
| `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | ~300MB | angry/disgust/fear/happy/neutral/sad/surprise | Popular, permissive, CPU-runnable |
| `superb/wav2vec2-base-superb-er` | ~95MB | angry/happy/neutral/sad | Smaller, faster |
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | ~1.2GB | valence/arousal/dominance | Too large for free tier |

**Recommendation:** `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` — good balance of accuracy and size. Will confirm with user before Phase 3.

---

## Phase 4 — Lap-Time Correlation

- Simple timestamp alignment: stress-flagged segment timestamps vs lap-time CSV
- Chart: Recharts `ComposedChart` — line for lap times, scatter/reference areas for stress moments

---

## Phase 5 — Deployment (Research Required Before Implementing)

### HF Spaces Options for React + FastAPI
| Approach | Pros | Cons |
|---|---|---|
| Docker Space (single container) | One service, clean | Need Nginx to serve React build + proxy to uvicorn |
| Two Spaces (React static + FastAPI) | Simpler each | CORS complexity, two repos |
| Docker Space (React build served by FastAPI static files) | Simplest architecture | Slight coupling |

**Likely recommendation:** Docker Space with Nginx + uvicorn supervisor, or FastAPI serving React build as static files. Will research and confirm with user before implementing.

---

## Open Questions (Track)

1. ✅ SER model choice — will confirm before Phase 3
2. ✅ HF access token for pyannote — will request before Phase 2  
3. ✅ Deployment packaging — will propose plan before Phase 5
4. ⬜ Test audio clip — user to provide before Phase 1 testing
5. ⬜ Lap-time CSV/JSON — user to provide before Phase 4
