# LISA — Stack 1 End-to-End Demo (Open Source, On-Prem)
**Stack:** ASR = faster-whisper (Whisper), MT = Marian (OPUS-MT), TTS = Piper

You can demo LISA two ways:
1) **Local pipeline (no Docker):** file → ASR → incremental MT (wait-k) → Piper → WAV + SRT
2) **Docker microservices:** ASR/MT/TTS services + a **microservice client** that orchestrates them

---

## 0) Requirements & Models

- Python 3.9+ (for local pipeline)
- For speed: NVIDIA GPU + CUDA (optional)
- **Piper** binary installed and a licensed **voice** (e.g., `de_DE-thorsten-high.onnx` CC0)
- MT models (downloaded on first run): OPUS-MT (*CC-BY-4.0* — add attribution)
- Whisper downloads on first run (small)

> See **ATTRIBUTION.md** for license notes.

Pinned OPUS-MT IDs:
- EN→DE: `Helsinki-NLP/opus-mt-en-de`
- DE→EN: `Helsinki-NLP/opus-mt-de-en`
- EN→ES: `Helsinki-NLP/opus-mt-en-es`
- ES→EN: `Helsinki-NLP/opus-mt-es-en`
- EN→FR: `Helsinki-NLP/opus-mt-en-fr`
- FR→EN: `Helsinki-NLP/opus-mt-fr-en`

---

## 1) Local pipeline (no Docker)

### 1.1 Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r orchestrator/requirements.txt
```

### 1.2 Configure
Copy `.env.example` → `.env` and set:
- `SRC_LANG` (en/de/es/fr) and `TGT_LANG`
- `MT_MODEL_ID` (e.g., `Helsinki-NLP/opus-mt-en-de`)
- `PIPER_BIN` (path to piper)
- `PIPER_VOICE` (path to voice .onnx, e.g., `voices/de_DE-thorsten-high.onnx`)

### 1.3 Put input audio
Place mono WAV at `samples/demo.wav` (16 kHz or 48 kHz). WAV is recommended for no-ffmpeg path.

### 1.4 Run
```bash
python orchestrator/file_demo.py --in samples/demo.wav --out out/demo_translated.wav --srt out/demo.srt --wait_k 5
```
Outputs:
- `out/demo_translated.wav` — synthesized audio
- `out/demo.srt` — translated subtitles

---

## 2) Docker microservices + client

### 2.1 Start services
```bash
docker compose up --build
```
- ASR: http://localhost:8001
- MT : http://localhost:8002
- TTS: http://localhost:8003

> **TTS note:** The container expects a `piper` binary. Either have `piper` in the container PATH or pass `-e PIPER_BIN=/mounted/piper` and mount the host binary, e.g.:
> ```yaml
>   tts_service:
>     environment:
>       - PIPER_BIN=/usr/local/bin/piper
>     volumes:
>       - /usr/local/bin/piper:/usr/local/bin/piper:ro
>       - ./voices:/voices:ro
> ```

### 2.2 Run microservice client
```bash
python orchestrator/microservice_client.py --in samples/demo.wav --out out/ms_translated.wav --srt out/ms_demo.srt --wait_k 5   --src en --tgt de --mt_model Helsinki-NLP/opus-mt-en-de
```
The client calls:
- `/transcribe` (ASR) → word timestamps
- Incremental **wait-k** MT calls to `/translate`
- `/tts` for each new suffix; concatenates WAV chunks

---

## 3) Notes
- This demo uses file mode to keep setup simple. For a live stream, feed frames to ASR, apply **wait-k** on the fly, and stream Piper audio to a jitter buffer.
- To optimize throughput: use **CUDA + int8** for faster-whisper on GPUs/CPUs; cache MT models; keep Piper voices on fast storage.
- Watch end-to-end latency by segment; target **p95 < 30 s** for long sessions.
