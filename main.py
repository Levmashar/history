# main.py
import json
import re
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "history_chunks.jsonl"
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

app = FastAPI(title="Armenian History Exam Helper")

# Allow frontend (Firebase) to call backend (Render / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # in production you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    # "short" -> very brief, "sentence" -> 1–2 sentences, "essay" -> small essay
    mode: str = "sentence"


class AnswerResponse(BaseModel):
    answer: str
    mode: str
    sources: List[Dict[str, Any]]


def load_chunks(path: str) -> List[Dict[str, Any]]:
    """Load JSONL chunks created by your chunker script."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


print("🔹 Loading chunks from history_chunks.jsonl ...")
CHUNKS = load_chunks(CHUNKS_FILE)
TEXTS = [c["text"] for c in CHUNKS]
print(f"   Loaded {len(CHUNKS)} chunks.")

print("🔹 Loading embedding model (multilingual, supports Armenian)...")
EMBEDDER = SentenceTransformer(MODEL_NAME)

print("🔹 Encoding chunks (runs once at startup)...")
EMBEDDINGS = EMBEDDER.encode(TEXTS, convert_to_numpy=True, normalize_embeddings=True)
print("✅ Backend is ready to answer questions.")

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[\.!?։])\s+")


def tokenize(text: str) -> List[str]:
    """Very simple tokenizer (works fine for Armenian too)."""
    return WORD_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using . ! ? և հայերեն «։»."""
    text = text.replace("\n", " ")
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def retrieve_chunks(question: str, top_k: int = 4):
    """Semantic search: find top_k most relevant chunks using embeddings."""
    q_vec = EMBEDDER.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    sims = EMBEDDINGS @ q_vec  # cosine similarity because vectors are normalized
    idxs = np.argsort(-sims)[:top_k]
    return [(CHUNKS[i], float(sims[i])) for i in idxs]


def generate_answer(question: str, contexts: List[str], mode: str) -> str:
    """
    Simple extractive summarizer:
    - picks sentences from the most relevant paragraphs
    - scores by word overlap with the question
    - returns short / 1–2 sentence / essay-length answer
    """
    combined = " ".join(contexts)
    sentences = split_sentences(combined)

    if not sentences:
        return (
            "Չկարողացա գտնել համապատասխան տեղեկություն դասագրքերում։ "
            "Փորձիր հարցդ ձևակերպել այլ կերպ կամ ավելի կոնկրետ։"
        )

    q_tokens = set(tokenize(question))
    scored = []

    for idx, sent in enumerate(sentences):
        tokens = set(tokenize(sent))
        if not tokens:
            continue
        overlap = len(tokens & q_tokens)
        score = overlap / (len(tokens) ** 0.5 + 1e-6)
        scored.append((idx, sent, score))

    if not scored:
        # fallback: keep original order
        scored = [(idx, s, 0.0) for idx, s in enumerate(sentences)]

    # sort by relevance
    scored.sort(key=lambda x: x[2], reverse=True)

    mode = mode.lower()

    # 1) Very short answer
    if mode == "short":
        top_sentence = scored[0][1]
        words = top_sentence.split()
        if len(words) > 25:
            return " ".join(words[:25]) + "…"
        return top_sentence

    # 2) Decide how many sentences we want
    if mode == "essay":
        n = min(7, len(scored))  # small essay
    else:  # "sentence" or anything else
        n = min(2, len(scored))  # 1–2 sentences

    # Keep chronological order of the chosen sentences
    chosen = sorted(scored[:n], key=lambda x: x[0])
    return " ".join(s for _, s, _ in chosen)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        return AnswerResponse(
            answer="Խնդրում եմ գրիր հարցդ։",
            mode=req.mode,
            sources=[]
        )

    mode = req.mode.lower()
    if mode not in {"short", "sentence", "essay"}:
        mode = "sentence"

    top_chunks = retrieve_chunks(question, top_k=4)
    contexts = [c["text"] for c, _ in top_chunks]
    answer = generate_answer(question, contexts, mode)

    # Metadata about used paragraphs for UI
    sources: List[Dict[str, Any]] = []
    for c, score in top_chunks:
        sources.append({
            "class": c.get("class"),
            "part": c.get("part"),
            "chapter": c.get("chapter"),
            "paragraph": c.get("paragraph"),
            "paragraph_title": c.get("paragraph_title"),
            "score": round(score, 3),
        })

    return AnswerResponse(
        answer=answer,
        mode=mode,
        sources=sources
    )
