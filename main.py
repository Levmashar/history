import os
import json
import math
import re
from typing import List, Dict, Any, Tuple, Literal
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse

# -----------------------------
# Config
# -----------------------------
CHUNKS_FILE = os.environ.get("CHUNKS_FILE", "history_chunks.jsonl")
TOP_K = 8               # how many chunks to retrieve
MAX_RETURN_CHUNKS = 3   # how many chunks' IDs to return in the response

# -----------------------------
# Tokenization (Armenian-friendly)
# -----------------------------
# Keep Armenian + Latin letters + digits. Lowercase Latin; Armenian has no case
TOKEN_RE = re.compile(r"[0-9A-Za-z\u0531-\u058F]+")

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    return TOKEN_RE.findall(text)

# -----------------------------
# Load chunks
# -----------------------------
def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # normalize required fields
            obj["text"] = obj.get("text", "").strip()
            chunks.append(obj)
    return chunks

# -----------------------------
# Build a tiny BM25 index
# -----------------------------
class BM25Index:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.doc_tokens: List[List[str]] = [tokenize(d) for d in docs]
        self.doc_len: List[int] = [len(toks) for toks in self.doc_tokens]
        self.avgdl: float = (sum(self.doc_len) / self.N) if self.N else 0.0

        # doc frequency
        df: Dict[str, int] = {}
        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        self.df = df

        # idf (BM25 variant that avoids negatives)
        self.idf: Dict[str, float] = {}
        for term, dfi in df.items():
            # classic BM25+ variant
            self.idf[term] = math.log(1 + (self.N - dfi + 0.5) / (dfi + 0.5))

    def score(self, q_tokens: List[str], doc_idx: int) -> float:
        toks = self.doc_tokens[doc_idx]
        if not toks:
            return 0.0
        # term frequency in doc
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1

        dl = self.doc_len[doc_idx]
        denom_norm = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
        score = 0.0
        for qt in q_tokens:
            if qt not in tf:
                continue
            idf = self.idf.get(qt, 0.0)
            freq = tf[qt]
            score += idf * (freq * (self.k1 + 1)) / (freq + denom_norm)
        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores: List[Tuple[int, float]] = []
        for i in range(self.N):
            s = self.score(q_tokens, i)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# -----------------------------
# Sentence selection (extractive)
# -----------------------------
SPLIT_RE = re.compile(r"(?<=[\.\?\!։…])\s+")

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = SPLIT_RE.split(text)
    # filter very short splits
    return [p.strip() for p in parts if len(p.strip()) > 2]

def sent_score(sent: str, q_tokens: List[str]) -> float:
    s_toks = tokenize(sent)
    if not s_toks:
        return 0.0
    hit = 0
    bucket = set(s_toks)
    for qt in q_tokens:
        if qt in bucket:
            hit += 1
    # weight by density (shorter, more focused sentences get a tiny boost)
    return hit / (1 + 0.02 * max(0, len(s_toks) - 20))

def pick_sentences(contexts: List[str], query: str, max_sentences: int) -> List[str]:
    q_tokens = tokenize(query)
    scored: List[Tuple[float, str]] = []
    for ctx in contexts:
        for s in split_sentences(ctx):
            if len(s) < 5:
                continue
            scored.append((sent_score(s, q_tokens), s))
    scored.sort(key=lambda x: x[0], reverse=True)
    # diverse selection (avoid near duplicates)
    chosen: List[str] = []
    seen = set()
    for sc, s in scored:
        if sc <= 0:
            break
        sig = " ".join(tokenize(s))[:80]
        if sig in seen:
            continue
        chosen.append(s)
        seen.add(sig)
        if len(chosen) >= max_sentences:
            break
    return chosen

def trim_words(text: str, max_words: int) -> str:
    toks = text.split()
    if len(toks) <= max_words:
        return text
    return " ".join(toks[:max_words]) + "…"

# -----------------------------
# Assemble the final answer
# -----------------------------
def answer_from_chunks(chunks: List[Dict[str, Any]], hits: List[Tuple[int, float]], query: str, mode: Literal["short","sentence","essay"]) -> Dict[str, Any]:
    if not hits:
        return {
            "answer": "Չգտնվեցին համապատասխան աղբյուրներ այս հարցի համար՝ ձեր տրամադրած դասագրքերում։ Փորձեք վերաձևակերպել հարցը կամ ավելացնել կոնկրետ թվականներ/անուններ:",
            "mode": mode,
            "used_chunks": [],
        }

    # build contexts by score order
    top_texts = [chunks[i]["text"] for i, _ in hits]
    top_meta  = [chunks[i] for i, _ in hits]

    # Pick sentences based on requested length
    if mode == "short":
        # 1 strong sentence, trimmed to ~25 words
        sents = pick_sentences(top_texts, query, max_sentences=1)
        if not sents:
            sents = [split_sentences(top_texts[0])[0] if split_sentences(top_texts[0]) else top_texts[0][:200]]
        final = trim_words(sents[0], 25)
    elif mode == "sentence":
        # 1–2 sentences (≈50 words)
        sents = pick_sentences(top_texts, query, max_sentences=2)
        if not sents:
            sents = split_sentences(top_texts[0])[:2] or [top_texts[0]]
        final = trim_words(" ".join(sents), 60)
    else:  # essay
        # 3–6 sentences stitched from best contexts (≈180–220 words)
        sents = pick_sentences(top_texts, query, max_sentences=6)
        if not sents:
            sents = split_sentences(top_texts[0])[:6] or [top_texts[0]]
        final = trim_words(" ".join(sents), 220)

    # light metadata for transparency
    used = []
    for (i, score) in hits[:MAX_RETURN_CHUNKS]:
        m = chunks[i]
        used.append({
            "id": i,
            "score": round(score, 4),
            "class": m.get("class"),
            "part": m.get("part"),
            "chapter": m.get("chapter"),
            "paragraph": m.get("paragraph"),
            "part_title": m.get("part_title"),
            "chapter_title": m.get("chapter_title"),
            "paragraph_title": m.get("paragraph_title"),
        })

    return {
        "answer": final,
        "mode": mode,
        "used_chunks": used,
    }

# -----------------------------
# FastAPI app + startup
# -----------------------------
app = FastAPI(title="History QA (chunks-only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals after startup
CHUNKS: List[Dict[str, Any]] = []
BM25: BM25Index | None = None

@app.on_event("startup")
def _startup():
    global CHUNKS, BM25
    if not os.path.exists(CHUNKS_FILE):
        # Optional: if you prefer, serve a friendly error
        raise RuntimeError(f"Chunks file not found: {CHUNKS_FILE}")
    CHUNKS = load_chunks(CHUNKS_FILE)
    BM25 = BM25Index([c.get("text", "") for c in CHUNKS])
    print(f"[startup] Loaded {len(CHUNKS)} chunks. avgdl={BM25.avgdl if BM25 else 'n/a'}")

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(CHUNKS)}

# optional: serve the jsonl back for convenience
@app.get("/history_chunks.jsonl")
def get_chunks_file():
    if not os.path.exists(CHUNKS_FILE):
        return PlainTextResponse("history_chunks.jsonl not found", status_code=404)
    return FileResponse(CHUNKS_FILE, media_type="application/jsonl")

# -----------------------------
# /ask endpoint (GET and POST)
# -----------------------------
class AskBody(BaseModel):
    question: str
    mode: Literal["short","sentence","essay"] = "short"
    k: int = TOP_K

def _do_ask(question: str, mode: str, k: int):
    if not question or not question.strip():
        return JSONResponse({"detail": "Empty question"}, status_code=400)
    hits = BM25.search(question, top_k=max(1, min(k, 30))) if BM25 else []
    payload = answer_from_chunks(CHUNKS, hits, question, mode)  # type: ignore
    return payload

@app.post("/ask")
def ask_post(body: AskBody = Body(...)):
    return _do_ask(body.question, body.mode, body.k)

@app.get("/ask")
def ask_get(
    question: str = Query(..., description="The user question in Armenian"),
    mode: Literal["short","sentence","essay"] = Query("short"),
    k: int = Query(TOP_K, ge=1, le=30),
):
    return _do_ask(question, mode, k)

# -----------------------------
# Local dev entrypoint
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
