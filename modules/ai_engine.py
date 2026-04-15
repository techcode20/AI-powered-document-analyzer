"""
ai_engine.py
------------
Core AI features powered by HuggingFace Transformers:
  - Summarization        (facebook/bart-large-cnn)
  - Question Answering   (deepset/roberta-base-squad2)
  - Sentiment Analysis   (distilbert-base-uncased-finetuned-sst-2-english)
  - Language Detection   (langdetect)

Models are lazy-loaded once and cached in module-level variables
to avoid reloading on every request (critical for free-tier Colab).
"""

import time
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import pipeline

# ── Make langdetect deterministic ──────────────────────────────────────────
DetectorFactory.seed = 42

# ── Lazy model cache ────────────────────────────────────────────────────────
_summarizer   = None
_qa_pipeline  = None
_sentiment_pipeline = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        print("[ai_engine] Loading summarization model…")
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1          # CPU; change to 0 for GPU
        )
    return _summarizer


def _get_qa():
    global _qa_pipeline
    if _qa_pipeline is None:
        print("[ai_engine] Loading QA model…")
        _qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1
        )
    return _qa_pipeline


def _get_sentiment():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("[ai_engine] Loading sentiment model…")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
    return _sentiment_pipeline


# ─────────────────────────────────────────────
# HELPER: safe text truncation
# ─────────────────────────────────────────────

def _truncate(text: str, max_chars: int = 3000) -> str:
    """Truncate text to max_chars to avoid token limit errors."""
    return text[:max_chars] if len(text) > max_chars else text


# ─────────────────────────────────────────────
# SUMMARIZATION
# ─────────────────────────────────────────────

def summarize(text: str) -> dict:
    """
    Generate an abstractive summary of the input text.

    Args:
        text: Document text to summarize.

    Returns:
        dict with keys: summary, confidence, processing_time_ms
    """
    start = time.time()
    text = _truncate(text, 3000)

    if len(text.split()) < 30:
        return {
            "summary": text,
            "confidence": 1.0,
            "processing_time_ms": 0
        }

    summarizer = _get_summarizer()
    result = summarizer(
        text,
        max_length=200,
        min_length=40,
        do_sample=False
    )

    summary_text = result[0]["summary_text"]
    elapsed_ms = round((time.time() - start) * 1000, 2)

    # Rough confidence: ratio of summary length to source length (capped at 1)
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
# QUESTION ANSWERING
# ─────────────────────────────────────────────

def answer_question(context: str, question: str) -> dict:
    """
    Answer a question based on document context.

    Args:
        context : Full document text used as context.
        question: The user's question.

    Returns:
        dict with keys: answer, confidence, start, end, processing_time_ms
    """
    start_time = time.time()
    context = _truncate(context, 4000)

    qa = _get_qa()
    result = qa(question=question, context=context)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "answer": result["answer"],
        "confidence": round(float(result["score"]), 4),
        "start_char": result["start"],
        "end_char": result["end"],
        "processing_time_ms": elapsed_ms
    }


# ─────────────────────────────────────────────
# SENTIMENT ANALYSIS
# ─────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    """
    Classify the overall sentiment of the document.

    Args:
        text: Document text.

    Returns:
        dict with keys: label, confidence, processing_time_ms
        label is one of: POSITIVE, NEGATIVE
    """
    start = time.time()
    text = _truncate(text, 512)

    sentiment = _get_sentiment()
    result = sentiment(text)[0]

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "label": result["label"],           # POSITIVE / NEGATIVE
        "confidence": round(float(result["score"]), 4),
        "processing_time_ms": elapsed_ms
    }


# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────

# Full language name map for common ISO 639-1 codes
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
    """
    Detect the language of the document text.

    Args:
        text: Document text (first 1000 chars used for speed).

    Returns:
        dict with keys: language_code, language_name, confidence, processing_time_ms
    """
    start = time.time()
    sample = text[:1000]

    try:
        lang_code = detect(sample)
        language_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
        # langdetect doesn't expose a probability directly; we set a fixed high value
        confidence = 0.95
    except LangDetectException:
        lang_code = "unknown"
        language_name = "Unknown"
        confidence = 0.0

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "language_code": lang_code,
        "language_name": language_name,
        "confidence": confidence,
        "processing_time_ms": elapsed_ms
    }
