import json
import os
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------- CONFIG ----------
CHUNKS_PATH = "history_chunks.jsonl"  # produced by your chunker
TOP_K = 5  # how many chunks to pull for an answer

# ---------- LOAD CHUNKS (JSONL!) ----------

def load_chunks(path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSONL file: one JSON object per line.
    Returns a list of dicts.
    """
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(f"Chunk file not found: {CHUNKS_PATH}")

CHUNKS: List[Dict[str, Any]] = load_chunks(CHUNKS_PATH)
CORPUS = [c["text"] for c in CHUNKS]

# ---------- VECTORIZER & MATRIX (FREE, LOCAL) ----------

# Simple word + bigram TF-IDF over Armenian text
VECTORIZER = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    # no stop_words="english" because text is Armenian
)

MATRIX = VECTORIZER.fit_transform(CORPUS)


# ---------- RETRIEVAL ----------

def retrieve(question: str, top_k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
    """
    Returns top_k chunks with cosine similarity scores.
    """
    q_vec = VECTORIZER.transform([question])
    sims = cosine_similarity(q_vec, MATRIX)[0]
    # sort indices by similarity
    top_indices = np.argsort(sims)[::-1][:top_k]

    results: List[Tuple[Dict[str, Any], float]] = []
    for idx in top_indices:
        score = float(sims[idx])
        if score <= 0:
            continue
        results.append((CHUNKS[idx], score))
    return results


# ---------- SIMPLE ARMENIAN-FRIENDLY SENTENCE SPLIT ----------

SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.!\?։])\s+')

def pick_sentences(text: str, max_words: int) -> str:
    """
    Pick sentences from text until we reach ~max_words.
    Very simple, but works for short/sentence/essay modes.
    """
    sentences = SENTENCE_SPLIT_RE.split(text)
    out: List[str] = []
    count = 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if not words:
            continue

        # if already have something and this would overshoot a lot, stop
        if count + len(words) > max_words and out:
            break

        out.append(s)
        count += len(words)
        if count >= max_words:
            break

    return " ".join(out).strip()


def build_answer(question: str,
                 style: Literal["short", "sentence", "essay"],
                 max_words: int | None = None) -> Tuple[str, List[Tuple[Dict[str, Any], float]]]:
    """
    Build an answer completely from textbook chunks.
    Returns (answer_text, sources_with_scores)
    """
    hits = retrieve(question)

    if not hits:
        # Armenian fallback message
        return (
            "Չգտնվեց համապատասխան հատված ուսումնական նյութում։ Փորձիր հարցը ձևակերպել այլ կերպ կամ ավելացնել մանրամասներ։",
            []
        )

    # Just concatenate the top chunks' text and then cut sentences.
    source_text = " ".join([c["text"] for c, _ in hits])

    if style == "essay":
        if max_words is None:
            max_words = 250
    elif style == "sentence":
        if max_words is None:
            max_words = 60
    else:  # short
        if max_words is None:
            max_words = 25

    answer = pick_sentences(source_text, max_words)
    return answer, hits


# ---------- FASTAPI APP ----------

app = FastAPI(
    title="History QA (Textbooks 7–12)",
    description="Free Armenian history QA over textbook chunks only.",
    version="0.1.0",
)

# Allow requests from your future frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    answer_style: Literal["short", "sentence", "essay"] = "short"
    max_words: int | None = None


@app.get("/")
def root():
    return {
        "message": "History QA is running.",
        "usage": {
            "GET": "/ask?question=...&answer_style=short|sentence|essay",
            "POST": "/ask",
            "POST_body_example": {
                "question": "Հարցդ այստեղ գրի",
                "answer_style": "sentence"
            }
        },
    }


@app.get("/ask")
def ask_get(
    question: str = Query(..., description="Քո հարցը հայերենով"),
    answer_style: Literal["short", "sentence", "essay"] = Query(
        "short", description=" desired answer length "
    ),
    max_words: int | None = Query(
        None, description="Optional max word count override"
    ),
):
    answer, hits = build_answer(question, answer_style, max_words)

    sources = [
        {
            "class": c.get("class"),
            "part": c.get("part"),
            "chapter": c.get("chapter"),
            "paragraph": c.get("paragraph"),
            "paragraph_title": c.get("paragraph_title"),
            "score": score,
        }
        for c, score in hits
    ]

    return {
        "answer": answer,
        "answer_style": answer_style,
        "sources": sources,
    }


@app.post("/ask")
def ask_post(req: QuestionRequest):
    """
    POST with JSON body:
    {
      "question": "քո հարցը",
      "answer_style": "short" | "sentence" | "essay"
    }
    """
    answer, hits = build_answer(req.question, req.answer_style, req.max_words)

    sources = [
        {
            "class": c.get("class"),
            "part": c.get("part"),
            "chapter": c.get("chapter"),
            "paragraph": c.get("paragraph"),
            "paragraph_title": c.get("paragraph_title"),
            "score": score,
        }
        for c, score in hits
    ]

    return {
        "answer": answer,
        "answer_style": req.answer_style,
        "sources": sources,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
