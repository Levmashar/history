# main.py
import os
import json
import re
from typing import List, Dict, Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Config ----------
HISTORY_FILE = os.environ.get("HISTORY_FILE", "history_chunks.jsonl")
TOP_K_DEFAULT = 5

# ---------- Data loading ----------

def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        print(f"[WARN] History file not found: {path}")
        return chunks

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping bad JSONL line: {e}")
                continue

    print(f"[INFO] Loaded {len(chunks)} chunks from {path}")
    return chunks


CHUNKS: List[Dict[str, Any]] = load_chunks(HISTORY_FILE)
CHUNK_TEXTS: List[str] = [c.get("text", "") for c in CHUNKS]

if CHUNKS:
    VECTORIZER = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        token_pattern=r"(?u)\b\w+\b"
    )
    CHUNK_MATRIX = VECTORIZER.fit_transform(CHUNK_TEXTS)
    print(f"[INFO] TF-IDF index built: {CHUNK_MATRIX.shape[0]} docs.")
else:
    VECTORIZER = None
    CHUNK_MATRIX = None
    print("[WARN] No chunks loaded; /ask will return a fallback message.")

# ---------- QA logic ----------

SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.!\?։])\s+')

def split_sentences(text: str) -> List[str]:
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def retrieve_top_chunks(question: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    if not CHUNKS or VECTORIZER is None or CHUNK_MATRIX is None:
        return []

    q_vec = VECTORIZER.transform([question])
    sims = cosine_similarity(q_vec, CHUNK_MATRIX)[0]  # shape: (n_chunks,)
    # Indices of top_k most similar chunks
    top_indices = sims.argsort()[::-1][:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        chunk = CHUNKS[idx].copy()
        chunk["score"] = float(sims[idx])
        results.append(chunk)
    return results


def build_answers(question: str, top_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    if not top_chunks:
        msg = "Ցավոք, տվյալ հարցի համար համապատասխան տեքստ չգտա այս բազայում։"
        return {
            "answer_short": msg,
            "answer_medium": msg,
            "answer_long": msg,
        }

    # Join texts of best chunks
    context = "\n\n".join(c.get("text", "") for c in top_chunks if c.get("text"))
    sentences = split_sentences(context)
    if not sentences:
        sentences = [context.strip()]

    # Short: first ~25 words of first sentence
    first_sentence_words = sentences[0].split()
    short = " ".join(first_sentence_words[:25])
    if len(first_sentence_words) > 25:
        short += "..."

    # Medium: first 1–2 sentences
    medium_sentences = sentences[:2]
    medium = " ".join(medium_sentences)

    # Long: small "essay": first ~5 sentences, capped by length
    long_sentences = sentences[:5]
    long = " ".join(long_sentences)
    if len(long) > 1200:
        long = long[:1200].rsplit(" ", 1)[0] + "..."

    return {
        "answer_short": short,
        "answer_medium": medium,
        "answer_long": long,
    }

# ---------- FastAPI setup ----------

app = FastAPI(
    title="History QA API",
    description="Armenian school history Q&A over textbook chunks",
    version="1.0.0",
)

# Allow your frontend (e.g. Firebase hosting). For now, allow all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    mode: Literal["all", "short", "medium", "long"] = "all"
    top_k: int = TOP_K_DEFAULT


class SourceChunk(BaseModel):
    score: float
    cls: Optional[str] = None
    part: Optional[int] = None
    part_title: Optional[str] = None
    chapter: Optional[int] = None
    chapter_title: Optional[str] = None
    paragraph: Optional[str] = None
    paragraph_title: Optional[str] = None
    preview: str


class AnswerResponse(BaseModel):
    answer_short: str
    answer_medium: str
    answer_long: str
    sources: List[SourceChunk]


@app.get("/")
async def root():
    return {"status": "ok", "message": "History QA backend is running."}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chunks_loaded": len(CHUNKS),
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")

    top_chunks = retrieve_top_chunks(req.question, top_k=req.top_k)
    answers = build_answers(req.question, top_chunks)

    # Filter by mode on the backend, but still send all fields in response
    # so frontend can reuse them without extra calls.
    if req.mode == "short":
        answers["answer_medium"] = ""
        answers["answer_long"] = ""
    elif req.mode == "medium":
        answers["answer_short"] = ""
        answers["answer_long"] = ""
    elif req.mode == "long":
        answers["answer_short"] = ""
        answers["answer_medium"] = ""
    # "all" -> keep all

    sources: List[SourceChunk] = []
    for c in top_chunks:
        preview_text = c.get("text", "")
        # short preview for UI
        preview_text = preview_text.replace("\n", " ")
        if len(preview_text) > 200:
            preview_text = preview_text[:200].rsplit(" ", 1)[0] + "..."

        src = SourceChunk(
            score=float(c.get("score", 0.0)),
            cls=str(c.get("class")) if c.get("class") is not None else None,
            part=c.get("part"),
            part_title=c.get("part_title"),
            chapter=c.get("chapter"),
            chapter_title=c.get("chapter_title"),
            paragraph=c.get("paragraph"),
            paragraph_title=c.get("paragraph_title"),
            preview=preview_text,
        )
        sources.append(src)

    return AnswerResponse(
        answer_short=answers["answer_short"],
        answer_medium=answers["answer_medium"],
        answer_long=answers["answer_long"],
        sources=sources,
    )
