"""
StressLine — FastAPI Backend
Phase 0: Health check skeleton. Models added phase by phase.
"""

import os
import shutil
import tempfile
import inspect
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
HF_CACHE_DIR = BASE_DIR / "hf-cache"
HF_CACHE_DIR.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR.resolve()))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((HF_CACHE_DIR / "hub").resolve()))
os.environ.setdefault("TRANSFORMERS_CACHE", str((HF_CACHE_DIR / "transformers").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str((HF_CACHE_DIR / "datasets").resolve()))
os.environ.setdefault("HF_MODULES_CACHE", str((HF_CACHE_DIR / "modules").resolve()))

load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

from huggingface_hub import snapshot_download
import yaml

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Lifespan (startup / shutdown) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once at startup to avoid per-request cold starts."""
    print("[STARTUP] Starting StressLine backend...")
    # Phase 1: Whisper model loaded lazily on first request (saves startup time)
    # Later phases will preload models here.
    yield
    print("[SHUTDOWN] Shutting down...")


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="StressLine API",
    description="Race radio analysis: ASR + diarization + stress detection",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration — allows local dev, Netlify, and custom frontend URLs
origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in origins_env.split(",")] if origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Upload directory ───────────────────────────────────────────────────────────

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
HF_MODEL_DIR = BASE_DIR / "hf-models"
HF_MODEL_DIR.mkdir(exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


# ── Response models ────────────────────────────────────────────────────────────

def call_hf_asr(audio_bytes: bytes, token: str, model_id: str = "openai/whisper-small") -> dict:
    """Call HF Inference API for Automatic Speech Recognition using InferenceClient.

    Uses huggingface_hub.InferenceClient which automatically selects the correct
    provider for the given model (avoids hardcoded 'hf-inference' provider issues).

    Returns:
        dict with keys 'text' and optionally 'chunks'.
    Raises:
        huggingface_hub.errors.InferenceTimeoutError: on timeout.
        huggingface_hub.errors.HfHubHTTPError: on 4xx/5xx API errors (triggers local fallback).
        Exception: other errors propagated to caller.
    """
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=token, timeout=60)
    try:
        result = client.automatic_speech_recognition(audio_bytes, model=model_id)
    except StopIteration as e:
        raise RuntimeError(f"InferenceClient StopIteration (ASR): {e}") from e
    # InferenceClient returns AutomaticSpeechRecognitionOutput or a string
    if hasattr(result, "text"):
        text = result.text
        chunks = getattr(result, "chunks", None) or []
        return {"text": text, "chunks": [
            {"text": c.text, "timestamp": list(c.timestamp) if c.timestamp else [0.0, 0.0]}
            for c in chunks
        ] if chunks else []}
    return {"text": str(result), "chunks": []}


def call_hf_ser(audio_bytes: bytes, token: str,
                model_id: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition") -> list:
    """Call HF Inference API for Audio Classification (SER) using InferenceClient.

    Returns:
        list of dicts with 'label' and 'score' keys, sorted descending by score.
    Raises:
        huggingface_hub.errors.HfHubHTTPError: on API errors (triggers local fallback).
    """
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=token, timeout=30)
    try:
        result = client.audio_classification(audio_bytes, model=model_id)
    except StopIteration as e:
        raise RuntimeError(f"InferenceClient StopIteration (SER): {e}") from e
    # Returns list of ClassificationOutput objects
    return [{"label": r.label, "score": float(r.score)} for r in result]

class HealthResponse(BaseModel):
    status: str
    version: str

# ── Global Models & Caches ───────────────────────────────────────────────────
whisper_model = None
diarization_pipeline = None
ser_pipeline = None
latest_lap_data = None

# ── Heuristic keywords for role assignment ────────────────────────────────────────
ENGINEER_KEYWORDS = [
    "box", "push", "lap", "tire", "fuel", "drs", "pit", "strategy",
    "speed", "rpm", "callout", "instruction", "gap", "delta", "mode",
]
DRIVER_KEYWORDS = [
    "i ", "i'", "i\"", "can't", "cannot", "damn", "shit", "fuck",
    "yeah", "wow", "oh", "uh", "look", "hold", "stop", "go", "slow",
    "fast", "please", "thanks", "sorry", "no grip", "understeer", "oversteer",
]


def merge_adjacent_segments(segments: list, max_gap: float = 1.0) -> list:
    """Merge adjacent same-speaker segments separated by <= max_gap seconds."""
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and (seg["start"] - prev["end"]) <= max_gap:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged


from typing import List, Optional


def ensure_local_hf_repo(repo_id: str, local_dir: Path, hf_token: str, marker_file: str):
    """Download a gated HF repo only when its expected local files are missing."""
    if (local_dir / marker_file).exists():
        return local_dir

    snapshot_download(
        repo_id,
        token=hf_token,
        local_dir=str(local_dir.resolve()),
    )
    return local_dir


def load_diarization_pipeline(hf_token: str):
    """Load a Pyannote diarization pipeline using a repo-local cache override."""
    os.environ["HF_HOME"] = str(HF_CACHE_DIR.resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str((HF_CACHE_DIR / "hub").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((HF_CACHE_DIR / "transformers").resolve())
    os.environ["HF_DATASETS_CACHE"] = str((HF_CACHE_DIR / "datasets").resolve())
    os.environ["HF_MODULES_CACHE"] = str((HF_CACHE_DIR / "modules").resolve())
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    cache_dir = HF_CACHE_DIR.resolve()

    from pyannote.audio import Pipeline

    from_pretrained = Pipeline.from_pretrained
    signature = inspect.signature(from_pretrained)
    kwargs = {"cache_dir": str(cache_dir)}

    if "token" in signature.parameters:
        kwargs["token"] = hf_token
    elif "use_auth_token" in signature.parameters:
        kwargs["use_auth_token"] = hf_token

    pipeline_repo = "pyannote/speaker-diarization-community-1"
    print(f"[DIARIZATION] Loading pipeline from {pipeline_repo} with cache {cache_dir}")

    try:
        return from_pretrained(pipeline_repo, **kwargs)
    except TypeError:
        for auth_arg in ("token", "use_auth_token"):
            try:
                return from_pretrained(pipeline_repo, **{auth_arg: hf_token, "cache_dir": str(cache_dir)})
            except TypeError:
                continue
        raise


def read_audio_for_pyannote(audio_path: str):
    """Read an audio file into a waveform dict when torchcodec is unavailable."""
    try:
        import soundfile as sf
        waveform, sample_rate = sf.read(audio_path, always_2d=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read audio file for pyannote: {exc}"
        ) from exc

    waveform = waveform.T.astype("float32")
    if waveform.ndim == 1:
        waveform = waveform[None, :]

    import torch
    return {"waveform": torch.from_numpy(waveform), "sample_rate": int(sample_rate)}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Simple liveness probe — used by frontend to confirm backend is reachable."""
    return HealthResponse(status="ok", version=app.version)


@app.post("/upload-audio", tags=["Audio"])
async def upload_audio(file: UploadFile = File(...)):
    """
    Accept an audio file upload, save it temporarily, and return a session ID.
    Actual processing (ASR, diarization, SER) happens in subsequent endpoints.

    Phase 0: Just validates the file type and saves it.
    Phase 1+: Will trigger the full pipeline.
    """
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. "
                   f"Supported: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    # Save to a temp file (unique name to avoid collisions)
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=UPLOAD_DIR
    )
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    finally:
        tmp.close()
        file.file.close()

    session_id = Path(tmp_path).stem  # Use temp filename as session ID

    return JSONResponse(
        content={
            "session_id": session_id,
            "filename": file.filename,
            "saved_as": Path(tmp_path).name,
            "message": "File uploaded successfully. Ready for processing.",
        }
    )


# ── Placeholder routes (filled in per phase) ──────────────────────────────────

@app.post("/transcribe", tags=["Phase 1 — ASR"])
async def transcribe(session_id: str):
    """
    Phase 1: Run Whisper ASR (openai/whisper-small).
    Tries HF Inference API first -> falls back to local Whisper model if API fails (429, 503, 401/403, connection error).
    """
    global whisper_model

    files = list(UPLOAD_DIR.glob(f"{session_id}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="Session audio not found.")

    audio_path = str(files[0])
    hf_token = os.environ.get("HF_TOKEN")
    execution_path = "api"
    result_data = None

    # 1️⃣ Primary: HF Inference API (openai/whisper-small via InferenceClient)
    if hf_token:
        try:
            print("[ASR] Attempting HF Inference API (InferenceClient) for openai/whisper-small...")
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            api_res = await asyncio.to_thread(call_hf_asr, audio_bytes, hf_token)

            text = api_res.get("text", "")
            segments = []
            for chunk in api_res.get("chunks", []):
                ts = chunk.get("timestamp", [0.0, 0.0])
                segments.append({
                    "start": ts[0] if ts and len(ts) > 0 and ts[0] is not None else 0.0,
                    "end": ts[1] if ts and len(ts) > 1 and ts[1] is not None else 0.0,
                    "text": chunk.get("text", ""),
                })
            if not segments:
                segments = [{"start": 0.0, "end": 0.0, "text": text}]

            result_data = {"text": text, "segments": segments}
            print("[ASR] HF Inference API succeeded.")
        except Exception as exc:
            print(f"[ASR] HF Inference API failed ({exc}). Falling back to local Whisper model...")
            execution_path = "local"
    else:
        print("[ASR] HF_TOKEN not set. Using local model...")
        execution_path = "local"

    # 2️⃣ Fallback: Local Whisper Model
    if result_data is None:
        if whisper_model is None:
            import whisper
            print("[ASR] Lazy-loading local Whisper model (tiny.pt)...")
            whisper_model = whisper.load_model(r"E:\MODELS\Whisper\tiny.pt")
            print("[ASR] Local Whisper model loaded.")

        print(f"[ASR] Running local transcription on {audio_path}...")
        local_res = await asyncio.to_thread(whisper_model.transcribe, audio_path)
        result_data = {
            "text": local_res.get("text", ""),
            "segments": local_res.get("segments", [])
        }

    result_data["execution_path"] = execution_path
    return JSONResponse(content=result_data)


@app.post("/diarize", tags=["Phase 2 — Diarization"])
async def diarize(session_id: str):
    """
    Phase 2: Run speaker diarization on uploaded audio. Merges adjacent same-speaker segments
    and uses keyword signal + turn pattern fallback to assign 'driver' or 'engineer' roles.
    """
    global diarization_pipeline, whisper_model
    if diarization_pipeline is None:
        print("[DIARIZATION] Loading Pyannote model (this may take a moment)...")
        import torch
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise HTTPException(status_code=500, detail="HF_TOKEN not found in environment.")
        
        try:
            diarization_pipeline = load_diarization_pipeline(hf_token)
            if torch.cuda.is_available():
                diarization_pipeline.to(torch.device("cuda"))
        except Exception as e:
            print(f"[DIARIZATION] Error loading pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load Pyannote pipeline: {str(e)}")
            
        print("[DIARIZATION] Model loaded successfully.")

    # Find the audio file matching the session_id
    files = list(UPLOAD_DIR.glob(f"{session_id}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="Session audio not found.")
    
    audio_path = str(files[0])
    print(f"[DIARIZATION] Processing {audio_path}...")
    
    try:
        diarization_result = await asyncio.to_thread(diarization_pipeline, audio_path)
    except RuntimeError as exc:
        if "torchcodec is not available" in str(exc):
            print("[DIARIZATION] torchcodec missing; falling back to soundfile audio loader.")
            audio_input = await asyncio.to_thread(read_audio_for_pyannote, audio_path)
            diarization_result = await asyncio.to_thread(diarization_pipeline, audio_input)
        else:
            raise
    
    if hasattr(diarization_result, "serialize"):
        diarization_annotation = diarization_result.speaker_diarization
    elif hasattr(diarization_result, "speaker_diarization"):
        diarization_annotation = diarization_result.speaker_diarization
    elif hasattr(diarization_result, "itertracks"):
        diarization_annotation = diarization_result
    else:
        raise HTTPException(
            status_code=500,
            detail="Unexpected diarization output type from pyannote.",
        )

    raw_segments = []
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        raw_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    
    # Merge adjacent same-speaker segments (gap <= 1.0s)
    segments = merge_adjacent_segments(raw_segments, max_gap=1.0)
    print(f"[DIARIZATION] Merged {len(raw_segments)} raw turns into {len(segments)} segments.")

    # ── Role Assignment: Transcript Keywords + Turn Pattern Fallback ──
    if whisper_model is None:
        import whisper
        whisper_model = whisper.load_model(r"E:\MODELS\Whisper\tiny.pt")

    whisper_result = await asyncio.to_thread(whisper_model.transcribe, audio_path)
    whisper_segments = whisper_result.get("segments", [])

    speaker_texts = {}
    for seg in segments:
        spk = seg["speaker"]
        speaker_texts.setdefault(spk, "")
    for wseg in whisper_segments:
        w_start = wseg.get("start", 0)
        w_end = wseg.get("end", 0)
        w_text = wseg.get("text", "")
        # Assign the whisper segment to any diarization segment it overlaps with
        for dseg in segments:
            if not (w_end <= dseg["start"] or w_start >= dseg["end"]):
                speaker = dseg["speaker"]
                speaker_texts[speaker] += " " + w_text.lower()

    # 4️⃣ Score each speaker using keyword lists
    def keyword_score(text, keywords):
        return sum(text.count(k) for k in keywords)

    scores = {}
    for spk, txt in speaker_texts.items():
        eng_score = keyword_score(txt, ENGINEER_KEYWORDS)
        drv_score = keyword_score(txt, DRIVER_KEYWORDS)
        scores[spk] = {"engineer": eng_score, "driver": drv_score}

    # 5️⃣ Determine primary roles based on text scores; fallback to turn‑pattern
    # Compute turn‑pattern stats
    turn_stats = {}
    for seg in segments:
        spk = seg["speaker"]
        turn_stats.setdefault(spk, {"turns": 0, "duration": 0.0})
        turn_stats[spk]["turns"] += 1
        turn_stats[spk]["duration"] += seg["end"] - seg["start"]

    # Helper to decide role for a speaker pair
    def assign_roles(speakers):
        spk_a, spk_b = speakers
        a_scores = scores[spk_a]
        b_scores = scores[spk_b]
        
        # Primary decision: Keyword signals
        if a_scores["engineer"] > b_scores["engineer"] and a_scores["driver"] <= b_scores["driver"]:
            return {spk_a: "engineer", spk_b: "driver"}
        if b_scores["engineer"] > a_scores["engineer"] and b_scores["driver"] <= a_scores["driver"]:
            return {spk_b: "engineer", spk_a: "driver"}
        
        # Fallback decision: Turn patterns (engineer = more turns, shorter avg length)
        a_turns = turn_stats[spk_a]["turns"]
        b_turns = turn_stats[spk_b]["turns"]
        a_avg = turn_stats[spk_a]["duration"] / a_turns if a_turns else 0
        b_avg = turn_stats[spk_b]["duration"] / b_turns if b_turns else 0
        if a_turns >= b_turns and a_avg <= b_avg:
            return {spk_a: "engineer", spk_b: "driver"}
        else:
            return {spk_b: "engineer", spk_a: "driver"}

    unique_speakers = list(turn_stats.keys())
    role_map = {}
    if len(unique_speakers) == 2:
        role_map = assign_roles(unique_speakers)
    elif len(unique_speakers) == 1:
        role_map = {unique_speakers[0]: "driver"}
    else:
        # Multiple speakers – assign based on highest engineer score
        sorted_by_eng = sorted(unique_speakers, key=lambda s: scores[s]["engineer"], reverse=True)
        top = sorted_by_eng[0]
        role_map[top] = "engineer"
        for other in unique_speakers:
            if other != top:
                role_map[other] = "driver"

    # 6️⃣ Append role to each segment in the response
    for seg in segments:
        seg["role"] = role_map.get(seg["speaker"], "unknown")

    return JSONResponse(content={"segments": segments})


class SegmentInput(BaseModel):
    start: float
    end: float
    speaker: Optional[str] = None
    role: Optional[str] = None

class StressAnalysisRequest(BaseModel):
    session_id: str
    segments: Optional[List[SegmentInput]] = None


@app.post("/analyze-stress", tags=["Phase 3 — SER"])
async def analyze_stress(req: StressAnalysisRequest):
    """
    Phase 3: Run Speech Emotion Recognition (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition).
    Tries HF Inference API first -> falls back to local SER model if API fails (429, 503, 401/403, connection error).
    """
    global ser_pipeline

    session_id = req.session_id
    files = list(UPLOAD_DIR.glob(f"{session_id}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="Session audio not found.")
    audio_path = str(files[0])

    import librosa
    import numpy as np
    import soundfile as sf
    import io
    import requests

    if req.segments and len(req.segments) > 0:
        segments_to_analyze = [s.model_dump() for s in req.segments]
    else:
        segments_to_analyze = []
        try:
            total_dur = librosa.get_duration(filename=audio_path)
        except Exception:
            total_dur = 0.0
        win = 2.0
        t = 0.0
        while t < total_dur:
            start = t
            end = min(t + win, total_dur)
            segments_to_analyze.append({"start": start, "end": end, "speaker": None, "role": None})
            t += win

    hf_token = os.environ.get("HF_TOKEN")
    use_api = bool(hf_token)
    execution_path = "api" if use_api else "local"
    results = []

    for seg in segments_to_analyze:
        start = seg["start"]
        end = seg["end"]
        duration = max(0.01, end - start)

        try:
            audio_arr, sr = librosa.load(audio_path, sr=16000, offset=start, duration=duration)
            if isinstance(audio_arr, np.ndarray):
                audio_input = audio_arr.astype(np.float32)
            else:
                audio_input = np.array(audio_arr, dtype=np.float32)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read audio segment: {exc}")

        label, score = None, 0.0

        # 1️⃣ Primary: HF Inference API (InferenceClient)
        if use_api:
            try:
                wav_io = io.BytesIO()
                sf.write(wav_io, audio_input, sr, format='WAV')
                wav_bytes = wav_io.getvalue()

                api_res = await asyncio.to_thread(call_hf_ser, wav_bytes, hf_token)

                if api_res:
                    top_item = max(api_res, key=lambda x: x.get("score", 0))
                    label = top_item.get("label", "neutral")
                    score = float(top_item.get("score", 0.0))
                else:
                    raise ValueError("Empty SER API response")
            except Exception as exc:
                print(f"[SER] HF Inference API failed ({exc}). Falling back to local model...")
                use_api = False
                execution_path = "local"

        # 2️⃣ Fallback: Local Model
        if not use_api or label is None:
            if ser_pipeline is None:
                try:
                    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
                    print("[SER] Lazy-loading local SER model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition...")
                    feat = AutoFeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
                    model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
                    model.to("cpu")
                    ser_pipeline = {"feature_extractor": feat, "model": model}
                    print("[SER] Local SER model loaded.")
                except Exception as e:
                    print(f"[SER] Error loading xlsr model ({e}), trying superb fallback...")
                    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
                    feat = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
                    model = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-er")
                    model.to("cpu")
                    ser_pipeline = {"feature_extractor": feat, "model": model}

            fe = ser_pipeline["feature_extractor"]
            model = ser_pipeline["model"]
            import torch

            def infer(array, sample_rate):
                inputs = fe(array, sampling_rate=sample_rate, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = model(**{k: v for k, v in inputs.items()})
                scores = torch.nn.functional.softmax(logits.logits, dim=-1)[0].cpu().numpy()
                idx = int(scores.argmax())
                lbl = model.config.id2label[idx]
                scr = float(scores[idx])
                return lbl, scr

            try:
                label, score = await asyncio.to_thread(infer, audio_input, 16000)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"SER inference failed: {exc}")

        results.append({
            "start": start,
            "end": end,
            "speaker": seg.get("speaker"),
            "role": seg.get("role"),
            "emotion": label,
            "score": score,
        })

    return JSONResponse(content={"segments": results, "execution_path": execution_path})


@app.post("/upload-lap-data", tags=["Phase 4 — Lap Timing"])
async def upload_lap_data(file: UploadFile = File(...)):
    """
    Upload CSV or JSON telemetry data for lap timing correlation.
    """
    global latest_lap_data
    content = await file.read()
    filename = file.filename.lower()
    laps = []
    
    try:
        if filename.endswith(".csv"):
            import csv
            import io
            text = content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            for idx, row in enumerate(reader, start=1):
                lap_num = int(row.get("lap") or row.get("Lap") or idx)
                ts = float(row.get("timestamp") or row.get("Timestamp") or row.get("time") or 0.0)
                lap_t = float(row.get("lap_time") or row.get("LapTime") or row.get("time_s") or row.get("duration") or 0.0)
                laps.append({"lap": lap_num, "timestamp": ts, "lap_time": lap_t})
        elif filename.endswith(".json"):
            import json
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, dict) and "laps" in data:
                data = data["laps"]
            for idx, item in enumerate(data, start=1):
                laps.append({
                    "lap": int(item.get("lap", idx)),
                    "timestamp": float(item.get("timestamp", 0.0)),
                    "lap_time": float(item.get("lap_time", 0.0))
                })
        else:
            raise HTTPException(status_code=400, detail="Only CSV or JSON telemetry files are supported.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error parsing lap data file: {exc}")

    if not laps:
        raise HTTPException(status_code=400, detail="No valid lap entries found in uploaded file.")

    latest_lap_data = laps
    return JSONResponse(content={"message": "Lap data uploaded successfully.", "laps": latest_lap_data, "source": "uploaded"})


@app.get("/lap-times", tags=["Phase 4 — Lap Timing"])
async def get_lap_times():
    """
    Phase 4: Return telemetry lap-time data (uploaded or hardcoded fallback).
    """
    global latest_lap_data
    if latest_lap_data is not None:
        return JSONResponse(content={"laps": latest_lap_data, "source": "uploaded"})

    sample_laps = [
        {"lap": 1, "timestamp": 0, "lap_time": 11.2},
        {"lap": 2, "timestamp": 11.2, "lap_time": 11.8},
        {"lap": 3, "timestamp": 23.0, "lap_time": 10.9},
        {"lap": 4, "timestamp": 33.9, "lap_time": 12.4},
        {"lap": 5, "timestamp": 46.3, "lap_time": 11.5},
    ]
    return JSONResponse(content={"laps": sample_laps, "source": "hardcoded"})

