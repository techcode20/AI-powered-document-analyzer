"""
nlp_tools.py
------------
NLP-powered features:
  - Named Entity Recognition  (spaCy)
  - Keyword Extraction        (TF-IDF via scikit-learn)
  - Topic Classification      (zero-shot via facebook/bart-large-mnli)
"""

import time
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# ── Lazy model cache ────────────────────────────────────────────────────────
_nlp_model         = None
_zero_shot_pipeline = None


def _get_nlp():
    global _nlp_model
    if _nlp_model is None:
        print("[nlp_tools] Loading spaCy model…")
        _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model


def _get_zero_shot():
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        print("[nlp_tools] Loading zero-shot classifier…")
        _zero_shot_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
    return _zero_shot_pipeline


# ─────────────────────────────────────────────
# NAMED ENTITY RECOGNITION
# ─────────────────────────────────────────────

# Map spaCy label → our clean category
ENTITY_LABEL_MAP = {
    "PERSON":   "names",
    "ORG":      "organizations",
    "GPE":      "locations",
    "LOC":      "locations",
    "DATE":     "dates",
    "TIME":     "dates",
    "MONEY":    "amounts",
    "CARDINAL": "amounts",
    "PERCENT":  "amounts",
    "QUANTITY": "amounts",
}


def extract_entities(text: str) -> dict:
    """
    Extract named entities from text using spaCy NER.

    Args:
        text: Document text.

    Returns:
        dict with keys:
            entities → { names, organizations, locations, dates, amounts, other }
            total_count, confidence, processing_time_ms
    """
    start = time.time()
    nlp = _get_nlp()

    # spaCy max length guard
    sample = text[:50000]
    doc = nlp(sample)

    buckets = {
        "names":         [],
        "organizations": [],
        "locations":     [],
        "dates":         [],
        "amounts":       [],
        "other":         []
    }

    seen = set()
    for ent in doc.ents:
        val = ent.text.strip()
        key = val.lower()
        if key in seen or not val:
            continue
        seen.add(key)

        category = ENTITY_LABEL_MAP.get(ent.label_, "other")
        buckets[category].append({
            "text":  val,
            "label": ent.label_
        })

    total = sum(len(v) for v in buckets.values())
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "entities":          buckets,
        "total_count":       total,
        "confidence":        0.88,   # spaCy en_core_web_sm reported F1 ~0.88
        "processing_time_ms": elapsed_ms
    }


# ─────────────────────────────────────────────
# KEYWORD EXTRACTION  (TF-IDF)
# ─────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 15) -> dict:
    """
    Extract top-N keywords using TF-IDF scoring.

    Args:
        text  : Document text.
        top_n : Number of keywords to return (default 15).

    Returns:
        dict with keys: keywords (list of {word, score}), processing_time_ms
    """
    start = time.time()

    # TF-IDF on single doc: treat each sentence as a mini-document
    sentences = [s.strip() for s in text.replace("\n", ". ").split(". ") if len(s.strip()) > 5]

    if len(sentences) < 2:
        sentences = [text]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=500,
            ngram_range=(1, 2)       # unigrams + bigrams
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()

        # Sum scores across all sentences
        scores = tfidf_matrix.sum(axis=0).A1
        scored_words = sorted(
            zip(feature_names, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        # Normalize scores to 0–1
        max_score = scored_words[0][1] if scored_words else 1.0
        keywords = [
            {
                "word":  word,
                "score": round(float(score) / float(max_score), 4)
            }
            for word, score in scored_words
        ]
    except Exception as e:
        keywords = []

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "keywords":          keywords,
        "processing_time_ms": elapsed_ms
    }


# ─────────────────────────────────────────────
# TOPIC CLASSIFICATION  (Zero-Shot)
# ─────────────────────────────────────────────

TOPIC_LABELS = [
    "Technology",
    "Finance & Business",
    "Healthcare & Medicine",
    "Legal & Compliance",
    "Science & Research",
    "Education",
    "Politics & Government",
    "Sports",
    "Entertainment & Media",
    "Environment & Climate",
    "Human Resources",
    "Marketing & Sales",
]


def classify_topic(text: str) -> dict:
    """
    Auto-classify document into one or more topic categories using zero-shot classification.

    Args:
        text: Document text.

    Returns:
        dict with keys:
            top_topic, all_scores (list of {label, score}),
            confidence, processing_time_ms
    """
    start = time.time()
    sample = text[:1500]      # keep it fast on free tier

    classifier = _get_zero_shot()
    result = classifier(
        sample,
        candidate_labels=TOPIC_LABELS,
        multi_label=True       # document can belong to multiple topics
    )

    all_scores = [
        {
            "label": label,
            "score": round(float(score), 4)
        }
        for label, score in zip(result["labels"], result["scores"])
    ]

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "top_topic":          all_scores[0]["label"],
        "all_scores":         all_scores,
        "confidence":         round(float(all_scores[0]["score"]), 4),
        "processing_time_ms": elapsed_ms
    }
