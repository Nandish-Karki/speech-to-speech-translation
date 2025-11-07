from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, os, tempfile, base64

app = FastAPI(title="TTS Service (Piper wrapper)")
PIPER_BIN = os.getenv("PIPER_BIN", "piper")

class TTSIn(BaseModel):
    text: str
    voice: str           # path to /voices/*.onnx (mounted volume)
    sample_rate: int = 22050

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/tts")
def tts(inp: TTSIn):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    cmd = [PIPER_BIN, "-m", inp.voice, "-w", out_path, "-s", str(inp.sample_rate)]
    proc = subprocess.run(cmd, input=inp.text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return {"error": proc.stderr.decode("utf-8","ignore")}
    with open(out_path, "rb") as f:
        wav_bytes = f.read()
    os.remove(out_path)
    return {"wav_base64": base64.b64encode(wav_bytes).decode("ascii")}
