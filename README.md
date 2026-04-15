# 🧠 AI Document Analyzer

> Multi-format document analysis powered by HuggingFace Transformers, spaCy & Flask.

---

## 📁 Project Structure

```
ai-doc-analyzer/
├── app.py                  ← Flask API (Phase 2)
├── requirements.txt        ← All dependencies
├── modules/
│   ├── extractor.py        ← PDF / DOCX / Image OCR extraction
│   ├── ai_engine.py        ← Summarization, QA, Sentiment, Language Detection
│   ├── nlp_tools.py        ← Entities, Keywords, Topic Classification
│   ├── comparator.py       ← Document similarity comparison
│   └── metadata.py         ← Word count, reading time, metadata
├── static/
│   └── index.html          ← Frontend UI (Phase 4)
└── uploads/                ← Temp storage for uploaded files
```

---

## ⚙️ Phase 1 Setup

### Step 1 — Install system dependency (for OCR)
```bash
# Ubuntu / Colab
apt-get install -y tesseract-ocr

# Mac
brew install tesseract
```

### Step 2 — Install Python packages
```bash
pip install -r requirements.txt
```

### Step 3 — Download spaCy language model
```bash
python -m spacy download en_core_web_sm
```

### Step 4 — Create uploads folder
```bash
mkdir -p uploads static
```

### Step 5 — Verify all modules import correctly
```bash
python -c "
from modules.extractor import extract
from modules.ai_engine import summarize, answer_question, analyze_sentiment, detect_language
from modules.nlp_tools import extract_entities, extract_keywords, classify_topic
from modules.comparator import compare_docs
from modules.metadata import compute_metadata
print('✅ All modules loaded successfully!')
"
```

---

## 🚀 Features Covered

| # | Feature | Module | Model |
|---|---------|--------|-------|
| 1 | PDF Extraction | extractor.py | pdfplumber |
| 2 | DOCX Extraction | extractor.py | python-docx |
| 3 | Image OCR | extractor.py | pytesseract |
| 4 | AI Summarization | ai_engine.py | facebook/bart-large-cnn |
| 5 | Question Answering | ai_engine.py | deepset/roberta-base-squad2 |
| 6 | Sentiment Analysis | ai_engine.py | distilbert-base-uncased-finetuned-sst-2-english |
| 7 | Language Detection | ai_engine.py | langdetect |
| 8 | Entity Extraction | nlp_tools.py | spaCy en_core_web_sm |
| 9 | Keyword Extraction | nlp_tools.py | TF-IDF (scikit-learn) |
| 10 | Topic Classification | nlp_tools.py | facebook/bart-large-mnli |
| 11 | Document Comparison | comparator.py | all-MiniLM-L6-v2 |
| 12 | Reading Time / Metadata | metadata.py | Rule-based |
| 13 | Table Extraction | extractor.py | pdfplumber |
| 14 | Confidence Scores | All modules | Per-model |
| 15 | Processing Metadata | All modules | Timing + counts |

---

## 📌 Next Steps

- **Phase 2** → Build `app.py` Flask API with all endpoints + Swagger docs
- **Phase 3** → Add `/qa` and `/compare` endpoints
- **Phase 4** → Build `static/index.html` frontend dashboard
- **Phase 5** → Test on Google Colab with proxy tunneling
