
def _detect_doc_type(text: str) -> str:
    text_lower = text.lower()
    opinion_signals = [
        "i believe", "i think", "i feel", "happiness",
        "fear", "love", "hate", "amazing", "terrible",
        "story", "anecdote", "lesson", "mindset",
        "you should", "the best", "the worst"
    ]
    formal_signals = [
        "whereas", "hereby", "pursuant", "therefore",
        "section", "clause", "regulation", "the party",
        "agreement", "terms and conditions"
    ]
    opinion_count = sum(1 for s in opinion_signals if s in text_lower)
    formal_count = sum(1 for s in formal_signals if s in text_lower)
    if formal_count >= 3 and formal_count > opinion_count:
        return "formal"
    return "opinion"

import re

def _is_noise_amount(value):
    val = value.strip()
    if re.match(r'^97[89][0-9\-]{10,}$', val): return True
    if re.match(r'^[+]?[0-9 ()\-.]{7,}$', val) and len(val) > 8: return True
    return False

import re as _re

def _clean_text_for_summary(text: str) -> str:
    """Remove noise before sending to AI — emails, phones, URLs, symbols"""
    # Remove emails
    text = _re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    # Remove URLs
    text = _re.sub(r'http[s]?://\S+', '', text)
    # Remove phone numbers
    text = _re.sub(r'[\+]?[0-9][\s\-\.]?[(]?[0-9]{3}[)]?[\s\-\.]?[0-9]{3}[\s\-\.]?[0-9]{4,}', '', text)
    # Remove LinkedIn/GitHub URLs
    text = _re.sub(r'(linkedin|github)\.com/\S+', '', text)
    # Remove lines that are just symbols or single words (headers)
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip very short lines (likely headers/labels)
        if len(stripped) < 15 and stripped.isupper():
            continue
        # Skip lines with mostly special characters
        alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
        if alpha_ratio < 0.4 and len(stripped) > 0:
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines).strip()

"""
ai_engine.py
------------
Core AI features powered by Groq LLM + RAG Pipeline:
  - Summarization        (Groq llama-3.3-70b-versatile)
  - Question Answering   (RAG pipeline — TF-IDF retrieval + Groq LLM)
  - Sentiment Analysis   (Groq llama-3.3-70b-versatile)
  - Language Detection   (langdetect)

API Key is loaded from environment variable GROQ_API_KEY.
Never hardcode API keys in source files.
"""

import os
import re
import time

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq

DetectorFactory.seed = 42

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. "
                "Run: import os; os.environ['GROQ_API_KEY'] = 'your-key'"
            )
        _client = Groq(api_key=api_key)
    return _client


def _ask(prompt: str, max_tokens: int = 400) -> str:
    try:
        r = _get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ai_engine] Groq error: {e}")
        return ""


def _truncate(text: str, max_chars: int = 4000) -> str:
    return str(text)[:max_chars] if len(str(text)) > max_chars else str(text)


def summarize(text: str) -> dict:
    start = time.time()
    t = _truncate(text, 4000)
    if len(t.split()) < 30:
        return {"summary": t, "confidence": 1.0, "processing_time_ms": 0}
    out = _ask(
        "Read the document below and write a clear, accurate 3-5 sentence summary.\n"
        "Write ONLY the summary sentences. No labels, no bullet points, no preamble.\n\n"
        f"Document:\n{t}", max_tokens=250)
    if not out:
        sentences = re.split(r"(?<=[.!?])\s+", t.strip())
        out = " ".join(sentences[:3])
    confidence = round(min(1.0, len(out.split()) / max(1, len(t.split()) * 0.3)), 2)
    return {"summary": out, "confidence": confidence,
            "processing_time_ms": round((time.time() - start) * 1000, 2)}


def _split_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    words = str(text).split()
    chunks, i = [], 0
    while i < len(words):
        end = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words): break
        i += chunk_size - overlap
    return chunks


def _top_chunks(question: str, chunks: list, top_k: int = 5) -> list:
    if not chunks: return chunks[:top_k]
    try:
        corpus = chunks + [question]
        mat = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus)
        sims = cosine_similarity(mat[-1], mat[:-1]).flatten()
        return [chunks[j] for j in np.argsort(sims)[::-1][:top_k]]
    except Exception:
        return chunks[:top_k]


def answer_question(context: str, question: str) -> dict:
    start  = time.time()
    chunks = _split_chunks(context)
    tops   = _top_chunks(question, chunks)
    ctx    = " ".join(tops)
    out = _ask(
        "Answer the question using ONLY the document excerpt below.\n"
        "Give a detailed, specific answer in 3-5 sentences.\n"
        "If the answer is not present say: 'This information is not found in the document.'\n\n"
        f"Question: {question}\n\nDocument excerpt:\n{ctx[:3000]}\n\nAnswer:",
        max_tokens=350)
    if not out:
        out = "Could not find a specific answer. Try rephrasing your question."
    conf = 0.88 if out and "not found in the document" not in out.lower() else 0.2
    return {"answer": out, "confidence": conf,
            "processing_time_ms": round((time.time() - start) * 1000, 2)}


def analyze_sentiment(text: str) -> dict:
    start = time.time()
    out = _ask(
        "What is the sentiment of this text?\n"
        "Reply with ONLY one word: POSITIVE, NEGATIVE, or NEUTRAL. Nothing else.\n\n"
        f"Text: {_truncate(text, 800)}", max_tokens=5)
    label = out.strip().upper()
    if label not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        label = "NEUTRAL"
    conf_map = {"POSITIVE": 0.92, "NEGATIVE": 0.91, "NEUTRAL": 0.85}
    return {"label": label, "confidence": conf_map[label],
            "processing_time_ms": round((time.time() - start) * 1000, 2)}


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
    sample = str(text)[:1000]
    try:
        lang_code     = detect(sample)
        language_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
        confidence    = 0.95
    except LangDetectException:
        lang_code, language_name, confidence = "unknown", "Unknown", 0.0
    return {"language_code": lang_code, "language_name": language_name,
            "confidence": confidence,
            "processing_time_ms": round((time.time() - start) * 1000, 2)}
