# StressLine 🏎️

> AI-powered race radio stress detection — detect driver stress from voice, correlate with lap times.

Built with **React + FastAPI + Hugging Face** as a Hackathon MVP.

---

## What It Does

StressLine analyzes F1-style race radio audio through a 4-phase AI pipeline:

| Phase | Task | Model |
|-------|------|-------|
| 1 | **ASR** — Speech-to-text | `openai/whisper-small` (HF API) / Whisper `tiny` (local fallback) |
| 2 | **Diarization** — Who's speaking? | `pyannote/speaker-diarization-community-1` |
| 3 | **SER** — Emotion / stress detection | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` |
| 4 | **Lap Timing** — Correlate stress with performance | CSV/JSON telemetry upload or sample data |

Driver vs Engineer roles are assigned using a combined keyword-signal + turn-pattern heuristic.

---

## Project Structure

```
StressLine/
├── backend/          # FastAPI server
│   ├── main.py       # All API routes
│   ├── requirements.txt
│   └── .env.example  # ← copy to .env and add your HF token
└── frontend/         # React + Vite app
    ├── src/
    │   ├── App.jsx
    │   └── App.css
    └── index.html
```

---

## Setup

### 1. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Create your .env file
copy .env.example .env
# Edit .env and add your HF_TOKEN
```

### 2. Frontend

```bash
cd frontend
npm install
```

### 3. Run

```bash
# Terminal 1 — backend
cd backend
.venv\Scripts\uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```

Open **http://localhost:5173**

---

## Environment Variables

Create `backend/.env` (not committed to git):

```env
HF_TOKEN=hf_your_token_here
```

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).  
Required permission: **"Make calls to the serverless Inference API"**

> **Note:** If no HF token is set, or if the API is unavailable, all models automatically fall back to local inference (Whisper tiny + local SER model).

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload-audio` | Upload audio file |
| `POST` | `/transcribe` | ASR transcription |
| `POST` | `/diarize` | Speaker diarization + role assignment |
| `POST` | `/analyze-stress` | Emotion/stress detection per segment |
| `POST` | `/upload-lap-data` | Upload CSV/JSON telemetry |
| `GET` | `/lap-times` | Fetch lap timing data |

---

## Supported Audio Formats

`.wav` · `.mp3` · `.m4a` · `.flac` · `.ogg` · `.webm`

---

## License

MIT
