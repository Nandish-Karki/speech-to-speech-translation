from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import os, tempfile

app = FastAPI(title="ASR Service (Whisper)")

model = WhisperModel(
    "small",
    device=os.getenv("ASR_DEVICE","cpu"),
    compute_type="int8" if os.getenv("ASR_DEVICE","cpu")=="cpu" else "float16"
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp.flush()
        segments, info = model.transcribe(tmp.name, beam_size=int(os.getenv("ASR_BEAM_SIZE","1")), word_timestamps=True, vad_filter=True)
        out = []
        for seg in segments:
            out.append({
                "start": float(seg.start), "end": float(seg.end), "text": seg.text,
                "words": [{"w": w.word, "s": float(w.start), "e": float(w.end)} for w in (seg.words or [])]
            })
    return {"segments": out}

"let me make the changes here as well"
