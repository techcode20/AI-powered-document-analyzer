"""
ai_engine.py
------------
Core AI features powered by HuggingFace Transformers:
  - Summarization        (facebook/bart-large-cnn)
  - Question Answering   (RAG pipeline — TF-IDF retrieval + deepset/roberta-base-squad2)
  - Sentiment Analysis   (distilbert-base-uncased-finetuned-sst-2-english)
  - Language Detection   (langdetect)
"""

import time
import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import pipeline as hf_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Make langdetect deterministic ──────────────────────────────────────────
DetectorFactory.seed = 42

# ── Lazy model cache ────────────────────────────────────────────────────────
_summarizer         = None
_qa_pipeline        = None
_sentiment_pipeline = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        print("[ai_engine] Loading summarization model…")
        _summarizer = hf_pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
    return _summarizer


def _get_qa():
    global _qa_pipeline
    if _qa_pipeline is None:
        print("[ai_engine] Loading QA model…")
        _qa_pipeline = hf_pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1
        )
    return _qa_pipeline


def _get_sentiment():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("[ai_engine] Loading sentiment model…")
        _sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
    return _sentiment_pipeline


# ─────────────────────────────────────────────
# HELPER: safe text truncation
# ─────────────────────────────────────────────

def _truncate(text: str, max_chars: int = 3000) -> str:
    return text[:max_chars] if len(text) > max_chars else text


# ─────────────────────────────────────────────
# SUMMARIZATION
# ─────────────────────────────────────────────

def summarize(text: str) -> dict:
    start = time.time()
    text = _truncate(text, 3000)

    if len(text.split()) < 30:
        return {
            "summary": text,
            "confidence": 1.0,
            "processing_time_ms": 0
        }

    try:
        summarizer = _get_summarizer()
        result = summarizer(
            text,
            max_length=200,
            min_length=40,
            do_sample=False,
            truncation=True
        )
        summary_text = result[0]["summary_text"]
    except Exception as e:
        # Fallback: extractive summary (first 3 sentences)
        print(f"[ai_engine] Summarizer error: {e}, using extractive fallback")
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        summary_text = " ".join(sentences[:3])

    elapsed_ms = round((time.time() - start) * 1000, 2)
    confidence = round(
        min(1.0, len(summary_text.split()) / max(1, len(text.split()) * 0.3)),
        2
    )

    return {
        "summary": summary_text,
        "confidence": confidence,
        "processing_time_ms": elapsed_ms
    }


# ─────────────────────────────────────────────
# RAG PIPELINE — RETRIEVAL AUGMENTED Q&A
# ─────────────────────────────────────────────

def _split_into_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """
    Split document text into overlapping word chunks for retrieval.
    chunk_size: words per chunk
    overlap   : words shared between consecutive chunks
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def _retrieve_top_chunks(question: str, chunks: list, top_k: int = 5) -> list:
    """
    Use TF-IDF cosine similarity to find the most relevant chunks for a question.
    Returns top_k chunks sorted by relevance.
    """
    if not chunks:
        return []

    corpus    = chunks + [question]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except Exception:
        # If TF-IDF fails (e.g. all stop words), return first chunks
        return chunks[:top_k]

    # Question vector is the last row
    question_vec   = tfidf_matrix[-1]
    chunk_vecs     = tfidf_matrix[:-1]
    similarities   = cosine_similarity(question_vec, chunk_vecs).flatten()

    # Get top_k chunk indices sorted by score
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_chunks  = [chunks[i] for i in top_indices]
    return top_chunks


def answer_question(context: str, question: str) -> dict:
    """
    RAG-based Question Answering:
      1. Split document into overlapping chunks
      2. Retrieve top-5 most relevant chunks via TF-IDF
      3. Run QA model on each retrieved chunk
      4. Return the answer with the highest confidence score

    Args:
        context : Full document text
        question: User's question

    Returns:
        dict with keys: answer, confidence, source_chunk, processing_time_ms
    """
    start_time = time.time()

    # ── Step 1: Split into chunks ──────────────────────────────────
    chunks = _split_into_chunks(context, chunk_size=200, overlap=50)

    if not chunks:
        return {
            "answer": "Could not extract text from document.",
            "confidence": 0.0,
            "source_chunk": "",
            "processing_time_ms": 0
        }

    # ── Step 2: Retrieve top relevant chunks ───────────────────────
    top_chunks = _retrieve_top_chunks(question, chunks, top_k=5)

    # ── Step 3: Run QA on each chunk, pick best answer ─────────────
    qa = _get_qa()
    best_answer    = ""
    best_score     = 0.0
    best_chunk     = ""

    for chunk in top_chunks:
        if len(chunk.strip()) < 10:
            continue
        try:
            result = qa(
                question=question,
                context=chunk,
                max_answer_len=100,
                handle_impossible_answer=True
            )
            if result["score"] > best_score:
                best_score  = result["score"]
                best_answer = result["answer"]
                best_chunk  = chunk
        except Exception as e:
            print(f"[ai_engine] QA chunk error: {e}")
            continue

    # ── Step 4: Fallback if no good answer ─────────────────────────
    if not best_answer or best_score < 0.01:
        # Try with first 512 words as last resort
        fallback_context = " ".join(context.split()[:512])
        try:
            result = qa(question=question, context=fallback_context)
            best_answer = result["answer"]
            best_score  = result["score"]
            best_chunk  = fallback_context[:200]
        except Exception:
            best_answer = "I could not find a specific answer in the document. Try rephrasing your question."
            best_score  = 0.0

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "answer":             best_answer,
        "confidence":         round(float(best_score), 4),
        "source_chunk":       best_chunk[:300] + "…" if len(best_chunk) > 300 else best_chunk,
        "processing_time_ms": elapsed_ms
    }


# ─────────────────────────────────────────────
# SENTIMENT ANALYSIS
# ─────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    start = time.time()
    text  = _truncate(text, 512)

    try:
        sentiment = _get_sentiment()
        result    = sentiment(text, truncation=True)[0]
        label     = result["label"]
        confidence = round(float(result["score"]), 4)
    except Exception as e:
        print(f"[ai_engine] Sentiment error: {e}")
        label      = "NEUTRAL"
        confidence = 0.5

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "label":              label,
        "confidence":         confidence,
        "processing_time_ms": elapsed_ms
    }


# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "bn": "Bengali",
    "ur": "Urdu", "tr": "Turkish", "pl": "Polish", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "no": "Norwegian", "cs": "Czech",
    "hu": "Hungarian", "ro": "Romanian", "vi": "Vietnamese", "th": "Thai",
    "id": "Indonesian", "ms": "Malay", "el": "Greek", "he": "Hebrew",
    "fa": "Persian", "uk": "Ukrainian", "ca": "Catalan", "sk": "Slovak",
}


def detect_language(text: str) -> dict:
    start  = time.time()
    sample = text[:1000]

    try:
        lang_code     = detect(sample)
        language_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
        confidence    = 0.95
    except LangDetectException:
        lang_code     = "unknown"
        language_name = "Unknown"
        confidence    = 0.0

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "language_code":      lang_code,
        "language_name":      language_name,
        "confidence":         confidence,
        "processing_time_ms": elapsed_ms
    }
