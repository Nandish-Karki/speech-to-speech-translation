from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
from functools import lru_cache

app = FastAPI(title="MT Service (Marian/OPUS-MT)")

class MTIn(BaseModel):
    text: str
    model_id: str

@lru_cache(maxsize=8)
def load_model(model_id: str):
    tok = MarianTokenizer.from_pretrained(model_id)
    mdl = MarianMTModel.from_pretrained(model_id)
    return tok, mdl

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/translate")
def translate(inp: MTIn):
    tok, mdl = load_model(inp.model_id)
    batch = tok([inp.text], return_tensors="pt", padding=True)
    gen = mdl.generate(**batch, max_new_tokens=200)
    out = tok.batch_decode(gen, skip_special_tokens=True)
    return {"text": out[0]}
