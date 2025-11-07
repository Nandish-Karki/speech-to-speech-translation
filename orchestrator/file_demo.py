#!/usr/bin/env python3
"""
Local file→file pipeline:
ASR (faster-whisper) → Incremental MT (wait-k) → Piper TTS → WAV + SRT
"""
import os, sys, argparse, subprocess, shutil, tempfile
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ASR
from faster_whisper import WhisperModel
# MT
from transformers import MarianMTModel, MarianTokenizer

def load_env():
    load_dotenv()
    cfg = {
        "SRC_LANG": os.getenv("SRC_LANG", "en"),
        "TGT_LANG": os.getenv("TGT_LANG", "de"),
        "MT_MODEL_ID": os.getenv("MT_MODEL_ID", "Helsinki-NLP/opus-mt-en-de"),
        "ASR_DEVICE": os.getenv("ASR_DEVICE", "cpu"),
        "ASR_BEAM_SIZE": int(os.getenv("ASR_BEAM_SIZE", "1")),
        "PIPER_BIN": os.getenv("PIPER_BIN", "piper"),
        "PIPER_VOICE": os.getenv("PIPER_VOICE", ""),
    }
    if not cfg["PIPER_VOICE"]:
        print("[WARN] PIPER_VOICE not set. TTS will be skipped.")
    return cfg

def transcribe_words(audio_path, device="cpu", beam_size=1):
    model_size = "small"
    model = WhisperModel(model_size, device=device, compute_type="int8" if device=="cpu" else "float16")
    segments, _ = model.transcribe(audio_path, beam_size=beam_size, word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            if w.word.strip():
                words.append((w.word.strip(), float(w.start), float(w.end)))
    return words

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

def load_mt(model_id):
    tok = MarianTokenizer.from_pretrained(model_id)
    mdl = MarianMTModel.from_pretrained(model_id)
    return tok, mdl

def mt_translate(tok, mdl, text):
    batch = tok([text], return_tensors="pt", padding=True)
    gen = mdl.generate(**batch, max_new_tokens=200)
    out = tok.batch_decode(gen, skip_special_tokens=True)
    return out[0]

def piper_tts(piper_bin, voice_path, text, out_wav):
    #  correct Piper args for Windows build
    cmd = [piper_bin, "--model", voice_path, "--output_file", out_wav]
    proc = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "ignore")
        print(f"[PIPER STDERR] {err}", file=sys.stderr)
        raise RuntimeError(err)
    return out_wav

def concat_wavs(paths, out_path, stderr_path):
    """Concatenate multiple WAV files safely, skipping any missing or empty ones."""
    sigs = []
    sr = None
    for p in paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            with open(stderr_path, "a", encoding="utf-8") as f:
                f.write(f"--- Skipped missing or empty chunk: {p} ---\n")
            continue
        try:
            data, rate = sf.read(p, dtype="int16")
        except Exception as ex:
            with open(stderr_path, "a", encoding="utf-8") as f:
                f.write(f"--- Failed to read chunk: {p} ---\n{ex}\n")
            continue

        if sr is None:
            sr = rate
        elif sr != rate:
            with open(stderr_path, "a", encoding="utf-8") as f:
                f.write(f"--- Sample rate mismatch in {p} ---\n")
            continue

        if data.size == 0:
            with open(stderr_path, "a", encoding="utf-8") as f:
                f.write(f"--- Empty audio in {p} ---\n")
            continue

        sigs.append(data)

    if not sigs:
        # 🔇 Create a short silent WAV so file always exists
        sr = 22050 if sr is None else sr
        silent = np.zeros(int(sr * 0.5), dtype=np.int16)
        sf.write(out_path, silent, sr)
        print(f"[INFO] No valid TTS chunks. Created silent file at {out_path}")
        return out_path

    full = np.concatenate(sigs, axis=0)
    sf.write(out_path, full, sr)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_wav", required=True)
    ap.add_argument("--out", dest="out_wav", required=True)
    ap.add_argument("--srt", dest="srt_path", required=True)
    ap.add_argument("--wait_k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_env()
    os.makedirs(os.path.dirname(args.out_wav), exist_ok=True)
    os.makedirs(os.path.dirname(args.srt_path), exist_ok=True)

    #  create isolated temporary working directory
    temp_dir = tempfile.mkdtemp(prefix="piper_pipeline_")
    stderr_path = os.path.join(temp_dir, "piper_stderr.txt")

    print("[1/5] ASR…")
    words = transcribe_words(args.in_wav, device=cfg["ASR_DEVICE"], beam_size=cfg["ASR_BEAM_SIZE"])
    if not words:
        concat_wavs([], args.out_wav, stderr_path)
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(0)

    print("[2/5] Load MT:", cfg["MT_MODEL_ID"])
    tok, mdl = load_mt(cfg["MT_MODEL_ID"])

    emitted_chars = 0
    srt_items = []
    tts_chunks = []
    prefix = []
    last_emit_time = words[0][1]

    def toks_to_text(toks): return " ".join(toks).replace("  "," ").strip()

    print("[3/5] Incremental MT + TTS…")
    for i, (w, s, e) in enumerate(tqdm(words)):
        prefix.append(w)
        boundary = (w.endswith(('.', '!', '?'))) or (i>0 and (s - words[i-1][2]) > 0.8)

        if len(prefix) >= args.wait_k:
            src = toks_to_text(prefix)
            mt_out = mt_translate(tok, mdl, src)
            new_text = mt_out[emitted_chars:].strip()
            if new_text:
                srt_items.append((last_emit_time, e, new_text))
                last_emit_time = e
                emitted_chars = len(mt_out)
                if cfg["PIPER_VOICE"]:
                    if not new_text.strip() or len(new_text.strip()) < 2:
                        print(f"[INFO] Skipping short/empty segment: '{new_text}'")
                    else:
                        out_chunk = os.path.join(temp_dir, f"chunk_{len(tts_chunks):04d}.wav")
                        try:
                            piper_tts(cfg["PIPER_BIN"], cfg["PIPER_VOICE"], new_text, out_chunk)
                            if os.path.exists(out_chunk) and os.path.getsize(out_chunk) > 0:
                                tts_chunks.append(out_chunk)
                            else:
                                with open(stderr_path, "a", encoding="utf-8") as f:
                                    f.write(f"--- Piper produced empty/invalid WAV for text: '{new_text}' ---\n")
                        except Exception as ex:
                            with open(stderr_path, "a", encoding="utf-8") as f:
                                f.write(f"--- Piper call failed for text: '{new_text}' ---\n{str(ex)}\n")

        if boundary:
            prefix = []
            emitted_chars = 0

    print("[4/5] Write SRT:", args.srt_path)
    write_srt(srt_items, args.srt_path)

    print("[5/5] Concatenate TTS chunks →", args.out_wav)
    concat_wavs(tts_chunks, args.out_wav, stderr_path)

    #  clean temp files
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("[CLEANUP] Removed temp directory.")

    print(" Done. Output WAV:", args.out_wav)

if __name__ == "__main__":
    main()
