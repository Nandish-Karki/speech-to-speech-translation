#!/usr/bin/env python3
"""
Microservice client: calls ASR (/transcribe), MT (/translate), TTS (/tts) with wait-k.
"""
import os, sys, argparse, json, base64, tempfile
import requests
import numpy as np
import soundfile as sf
from tqdm import tqdm

def srt_ts(t):
    ms = int(round(t*1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_srt(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(items, 1):
            f.write(f"{i}\n{srt_ts(start)} --> {srt_ts(end)}\n{text}\n\n")

def concat_wavs(paths, out_path):
    sigs, sr = [], None
    for p in paths:
        data, rate = sf.read(p, dtype="int16")
        if sr is None:
            sr = rate
        elif sr != rate:
            raise ValueError("Sample rate mismatch")
        sigs.append(data)
    if not sigs:
        return None
    full = np.concatenate(sigs, axis=0)
    sf.write(out_path, full, sr)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_wav", required=True)
    ap.add_argument("--out", dest="out_wav", required=True)
    ap.add_argument("--srt", dest="srt_path", required=True)
    ap.add_argument("--wait_k", type=int, default=5)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--mt_model", required=True)
    ap.add_argument("--asr_url", default="http://localhost:8001/transcribe")
    ap.add_argument("--mt_url", default="http://localhost:8002/translate")
    ap.add_argument("--tts_url", default="http://localhost:8003/tts")
    ap.add_argument("--voice", default="voices/de_DE-thorsten-high.onnx")
    args = ap.parse_args()

    # 1) ASR
    with open(args.in_wav, "rb") as f:
        files = {"audio": ("demo.wav", f, "audio/wav")}
        r = requests.post(args.asr_url, files=files, timeout=600)
    r.raise_for_status()
    asr = r.json()
    words = []
    for seg in asr.get("segments", []):
        for w in seg.get("words", []):
            ww = w.get("w","").strip()
            if ww:
                words.append((ww, float(w.get("s",0.0)), float(w.get("e",0.0))))
    if not words:
        print("No words recognized"); sys.exit(1)

    # 2) Incremental MT + TTS
    emitted_chars = 0
    prefix = []
    srt_items = []
    tts_chunks = []
    os.makedirs(os.path.dirname(args.out_wav), exist_ok=True)
    os.makedirs(os.path.dirname(args.srt_path), exist_ok=True)
    tmpd = tempfile.mkdtemp(prefix="lisa_ms_tts_")

    def toks_to_text(toks): return " ".join(toks).replace("  "," ").strip()

    for i, (w, s, e) in enumerate(tqdm(words)):
        prefix.append(w)
        boundary = (w.endswith(('.', '!', '?'))) or (i>0 and (s - words[i-1][2]) > 0.8)

        if len(prefix) >= args.wait_k:
            src = toks_to_text(prefix)
            r = requests.post(args.mt_url, json={"text": src, "model_id": args.mt_model}, timeout=600)
            r.raise_for_status()
            mt_out = r.json()["text"]
            new_text = mt_out[emitted_chars:].strip()
            if new_text:
                srt_items.append((s, e, new_text))
                emitted_chars = len(mt_out)
                # TTS
                r2 = requests.post(args.tts_url, json={"text": new_text, "voice": args.voice}, timeout=600)
                r2.raise_for_status()
                data = r2.json()
                wav_b64 = data.get("wav_base64")
                if wav_b64:
                    b = base64.b64decode(wav_b64.encode("ascii"))
                    p = os.path.join(tmpd, f"chunk_{len(tts_chunks):04d}.wav")
                    with open(p, "wb") as f:
                        f.write(b)
                    tts_chunks.append(p)

        if boundary:
            prefix = []
            emitted_chars = 0

    write_srt(srt_items, args.srt_path)
    if tts_chunks:
        concat_wavs(tts_chunks, args.out_wav)
        print("WAV written:", args.out_wav)
    else:
        print("No audio synthesized; only SRT written.")

if __name__ == "__main__":
    main()
