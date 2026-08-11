# The Silent Co-Driver: Progress Report (through Phase 2 debugging)

This document summarizes what has been completed so far and the current blocker in Phase 2 as of August 11, 2026.

## Project status

- Stack is in place: React + Vite frontend, FastAPI backend.
- Audio upload works end to end.
- ASR works through the local Whisper model.
- Diarization UI flow is wired, but Phase 2 is still blocked by model-loading issues in the Pyannote dependency chain.

## Phase 0: Setup and environment

- Full-stack project structure is set up under `backend/` and `frontend/`.
- FastAPI health endpoint is implemented and responds successfully.
- Frontend can connect to the backend and drive the upload/transcribe/diarize flow.
- Sample frontend dev server was observed running on `http://127.0.0.1:5174` when `5173` was occupied.

## Phase 1: Speech-to-text

- Whisper transcription endpoint is implemented at `POST /transcribe`.
- Backend loads local Whisper model from `E:\MODELS\Whisper\tiny.pt`.
- Frontend transcript panel is implemented and connected to backend output.
- Upload plus transcription flow is functionally in place.

## Phase 2: Speaker diarization

### What has been implemented

- `POST /diarize` exists in the backend.
- Frontend diarization panel and button are implemented.
- Hugging Face token loading from `.env` is in place.
- Loader compatibility was added for different Pyannote auth argument styles:
  - `token=...`
  - `use_auth_token=...`

### What was investigated

The following issues were reproduced and explored during debugging:

1. Initial Pyannote auth keyword mismatch
- Earlier error: `Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'`
- Backend was updated to support the installed Pyannote signature dynamically.

2. Windows cache permission failures
- Repeated error seen in app:
  - `Failed to load Pyannote pipeline: [WinError 5] Access is denied: 'C:\\Users\\LENOVO\\.cache\\huggingface\\hub\\models--pyannote--speaker-diarization-3.1'`
- Multiple attempts were made to redirect Hugging Face cache/model downloads into project-local folders.

3. Localized model loading attempts
- Local download/load flow was added for:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`
  - `pyannote/wespeaker-voxceleb-resnet34-LM`
- A local rewritten pipeline config was also introduced so dependent models could be referenced from local folders instead of hub IDs.

4. Additional gated dependency discovery
- Direct loader tests later revealed that the dependency chain also attempts to access:
  - `pyannote/speaker-diarization-community-1`
- This produced a gated repository access error during direct backend-side testing.

5. Audio/runtime warning found in environment
- Pyannote emitted warnings that `torchcodec` / FFmpeg support is not correctly installed.
- This has not yet been handled because model loading is still failing earlier in the pipeline.

## Direct tests performed

- Backend restart was performed and `/health` responded successfully afterward.
- User-provided sample file was tested directly against backend:
  - `C:\Users\LENOVO\Downloads\WhatsApp Audio 2026-08-11 at 1.47.45 AM.wav`
- Result of direct backend test:
  - upload succeeded
  - diarization still failed

## Phase 2: Speaker diarization

### What has been implemented

- `POST /diarize` exists in the backend and the frontend diarization button is now functional.
- The backend loads `pyannote/speaker-diarization-community-1` successfully using a repo-local Hugging Face cache override.
- The backend now handles missing `torchcodec` by falling back to `soundfile` and passing a waveform dictionary to the pipeline.
- The pyannote output object is parsed correctly via `output.speaker_diarization`, so diarization segments are returned to the frontend.

### What was fixed

1. Early HF cache env vars were set so pyannote does not try to write into `C:\Users\LENOVO\.cache\huggingface`.
2. The pipeline was changed from legacy 3.1 to `pyannote/speaker-diarization-community-1` to avoid nested gated dependency issues.
3. Audio decoding now falls back to `soundfile` when `torchcodec` is unavailable.
4. The backend no longer assumes `diarization_result.itertracks()` exists directly; it reads `diarization_result.speaker_diarization` first.

## Result

- Phase 2 is now working.
- The frontend shows diarization segments and the backend health is stable.
- Phase 3 (SER) remains the next feature to implement.

## Summary

Phase 0 and Phase 1 are complete.
Phase 2 is also complete with the current backend fix and frontend flow. The next step is Phase 3 speech emotion recognition.
