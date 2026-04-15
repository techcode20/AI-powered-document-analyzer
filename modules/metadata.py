"""
metadata.py
-----------
Compute document metadata:
  - Word count
  - Character count
  - Sentence count
  - Paragraph count
  - Estimated reading time
  - Processing speed summary
"""

import re
import time


# Average adult reading speed (words per minute)
READING_SPEED_WPM = 200


def compute_metadata(text: str, extra: dict = None) -> dict:
    """
    Compute all metadata for a document.

    Args:
        text  : Extracted document text.
        extra : Optional dict with extra info from extractor
                (e.g., page_count, processing_time_ms).

    Returns:
        dict with full metadata.
    """
    start = time.time()

    # ── Basic counts ──────────────────────────────────────────────────
    words      = text.split()
    word_count = len(words)
    char_count = len(text)
    char_no_spaces = len(text.replace(" ", "").replace("\n", ""))

    # Sentence count: split on . ! ?
    sentences  = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sent_count = len(sentences)

    # Paragraph count: split on blank lines
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    para_count = len(paragraphs)

    # ── Reading time ─────────────────────────────────────────────────
    total_minutes  = word_count / READING_SPEED_WPM
    read_minutes   = int(total_minutes)
    read_seconds   = int((total_minutes - read_minutes) * 60)

    if read_minutes == 0 and read_seconds < 30:
        reading_time_label = "Less than 1 minute"
    elif read_minutes == 0:
        reading_time_label = "About 1 minute"
    else:
        reading_time_label = f"{read_minutes} min {read_seconds} sec"

    # ── Lexical diversity ────────────────────────────────────────────
    unique_words  = len(set(w.lower() for w in words))
    lexical_diversity = round(unique_words / word_count, 4) if word_count > 0 else 0.0

    # ── Average sentence length ──────────────────────────────────────
    avg_sentence_length = round(word_count / sent_count, 1) if sent_count > 0 else 0.0

    elapsed_ms = round((time.time() - start) * 1000, 2)

    result = {
        "word_count":            word_count,
        "character_count":       char_count,
        "character_count_no_spaces": char_no_spaces,
        "sentence_count":        sent_count,
        "paragraph_count":       para_count,
        "unique_word_count":     unique_words,
        "lexical_diversity":     lexical_diversity,
        "avg_sentence_length":   avg_sentence_length,
        "reading_time": {
            "label":             reading_time_label,
            "total_minutes":     round(total_minutes, 2),
            "minutes":           read_minutes,
            "seconds":           read_seconds,
            "based_on_wpm":      READING_SPEED_WPM
        },
        "metadata_processing_time_ms": elapsed_ms
    }

    # Merge in extractor-level metadata (page_count, etc.)
    if extra:
        result.update({
            k: v for k, v in extra.items()
            if k not in result
        })

    return result
