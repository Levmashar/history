# main.py
import json
import re
from typing import Any, Dict, List, Optional, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_FILE = "history_chunks.jsonl"

# ---------- Load chunks from JSONL ----------

def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                # If some line is corrupted, skip but don’t crash everything
                print(f"Skipping bad JSON line: {e}")
    return chunks

CHUNKS: List[Dict[str, Any]] = load_chunks(CHUNKS_FILE)

if not CHUNKS:
    raise RuntimeError(
        "No chunks loaded. Did you generate history_chunks.jsonl "
        "and deploy it with the app?"
    )

# ---------- Build searchable corpus ----------

def build_doc_text(chunk: Dict[str, Any]) -> str:
    """
    Combine titles + paragraph text into one string for TF-IDF.
    """
    parts = []
    for key in ("part_title", "chapter_title", "paragraph_title"):
        val = chunk.get(key)
        if val:
            parts.append(str(val))
    parts.append(chunk.get("text", ""))
    return "\n".join(parts)

DOCS: List[str] = [build_doc_text(c) for c in CHUNKS]

# Simple sentence splitting (handles Armenian "։" + usual punctuation)
SENT_SPLIT_RE = re.compile(r"(?<=[\.?!։])\s+")

def split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return sentences

# TF-IDF vectorizer
VECTORIZER = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"(?u)\b\w+\b",
    lowercase=True,
)
DOC_MATRIX = VECTORIZER.fit_transform(DOCS)


def search_chunks(
    question: str,
    top_k: int = 5,
    class_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return top_k most similar chunks to the question using cosine similarity.
    Optionally filter by class ("7", "8", "9", etc).
    """
    if not question.strip():
        raise ValueError("Empty question")

    q_vec = VECTORIZER.transform([question])
    sims = cosine_similarity(q_vec, DOC_MATRIX)[0]  # (n_docs,)

    # Sort by similarity descending
    indices = np.argsort(-sims)

    results: List[Dict[str, Any]] = []
    for idx in indices:
        score = float(sims[idx])
        if score <= 0:
            break  # nothing more relevant

        chunk = CHUNKS[idx]

        if class_filter is not None and class_filter != "":
            if str(chunk.get("class")) != str(class_filter):
                continue

        item = {
            "score": score,
            "chunk_index": int(idx),
            "class": chunk.get("class"),
            "part": chunk.get("part"),
            "part_title": chunk.get("part_title"),
            "chapter": chunk.get("chapter"),
            "chapter_title": chunk.get("chapter_title"),
            "paragraph": chunk.get("paragraph"),
            "paragraph_title": chunk.get("paragraph_title"),
            "text": chunk.get("text", ""),
        }
        results.append(item)

        if len(results) >= top_k:
            break

    return results

# ---------- Answer generation ----------

AnswerMode = Literal["short", "sentence", "essay"]

def build_answer(
    question: str,
    mode: AnswerMode,
    class_filter: Optional[str] = None,
) -> Dict[str, Any]:
    hits = search_chunks(question, top_k=5, class_filter=class_filter)

    if not hits:
        return {
            "answer": (
                "Չկարողացա համապատասխան տեղեկություն գտնել դասագրքերի տեքստում։ "
                "Փորձիր ձևակերպել հարցը այլ կերպ կամ նշիր դասարանը։"
            ),
            "mode": mode,
            "sources": [],
        }

    best = hits[0]
    sentences = split_sentences(best["text"])

    # Fallback: if splitting fails, use raw text
    if not sentences:
        sentences = [best["text"]]

    if mode == "short":
        # 1 sentence, trimmed
        selected = sentences[0][:280]
    elif mode == "sentence":
        # 1–2 sentences
        selected = " ".join(sentences[:2])
        if len(selected) > 600:
            selected = selected[:600] + "…"
    else:  # "essay"
        # Combine a few sentences from top 2 chunks for a mini-essay
        essay_sentences: List[str] = []
        for hit in hits[:2]:
            essay_sentences.extend(split_sentences(hit["text"])[:5])

        if not essay_sentences:
            essay_sentences = sentences

        selected = " ".join(essay_sentences)
        if len(selected) > 1200:
            selected = selected[:1200] + "…"

    # Clean whitespace
    selected = re.sub(r"\s+", " ", selected).strip()

    # Prepare source snippets
    sources: List[Dict[str, Any]] = []
    for hit in hits:
        src_sentences = split_sentences(hit["text"])[:3]
        snippet = " ".join(src_sentences)
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"

        sources.append(
            {
                "score": hit["score"],
                "class": hit["class"],
                "part": hit["part"],
                "part_title": hit["part_title"],
                "chapter": hit["chapter"],
                "chapter_title": hit["chapter_title"],
                "paragraph": hit["paragraph"],
                "paragraph_title": hit["paragraph_title"],
                "snippet": snippet,
            }
        )

    return {
        "answer": selected,
        "mode": mode,
        "sources": sources,
    }

# ---------- FastAPI app ----------

app = FastAPI(
    title="Armenian History Tutor",
    description="Answers questions using ONLY the Armenian history textbooks (7–12th grade).",
    version="1.0.0",
)

# CORS so that Firebase/local front-end can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # optionally restrict to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    mode: AnswerMode = "sentence"  # "short" | "sentence" | "essay"
    school_class: Optional[str] = None  # e.g. "7","8","9","10","11","12"


class AskResponse(BaseModel):
    answer: str
    mode: AnswerMode
    sources: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(CHUNKS)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        result = build_answer(req.question, req.mode, req.school_class)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AskResponse(**result)
