# 🤖 AI Document Analyzer
Multi-format document analysis powered by Groq LLM, spaCy & Flask.

## 📁 Project Structure
```
AI-powered-document-analyzer/
├── app.py                  ← Flask API (4 endpoints)
├── requirements.txt
├── modules/
│   ├── extractor.py        ← PDF / DOCX / Image OCR extraction
│   ├── ai_engine.py        ← Summarization, QA, Sentiment, Language Detection
│   ├── nlp_tools.py        ← Entities, Keywords, Topic Classification
│   ├── comparator.py       ← Document similarity comparison
│   └── metadata.py         ← Word count, reading time, metadata
├── static/
│   └── index.html          ← Frontend UI
└── uploads/                ← Temp storage
```

## ⚙️ Setup

**Step 1 — Install system dependency**
```bash
apt-get install -y tesseract-ocr
```
**Step 2 — Install Python packages**
```bash
pip install -r requirements.txt
```
**Step 3 — Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```
**Step 4 — Set Groq API key**
```bash
export GROQ_API_KEY=your-gsk-key-here
```
**Step 5 — Run Flask**
```bash
python app.py
```

## 🚀 Features

| # | Feature | Module | Model |
|---|---------|--------|-------|
| 1 | PDF Extraction | extractor.py | pdfplumber |
| 2 | DOCX Extraction | extractor.py | python-docx |
| 3 | Image OCR | extractor.py | pytesseract |
| 4 | AI Summarization | ai_engine.py | Groq llama-3.3-70b |
| 5 | RAG Question Answering | ai_engine.py | TF-IDF + Groq |
| 6 | Sentiment Analysis | ai_engine.py | Groq llama-3.3-70b |
| 7 | Language Detection | ai_engine.py | langdetect |
| 8 | Named Entity Recognition | nlp_tools.py | Groq + spaCy fallback |
| 9 | Keyword Extraction | nlp_tools.py | TF-IDF (scikit-learn) |
| 10 | Topic Classification | nlp_tools.py | Groq llama-3.3-70b |
| 11 | Document Comparison | comparator.py | all-MiniLM-L6-v2 |
| 12 | Table Extraction | extractor.py | pdfplumber |
| 13 | Reading Time / Metadata | metadata.py | Rule-based |
| 14 | Confidence Scores | All modules | Per-feature |
| 15 | Q&A Chat History | ai_engine.py | Groq + RAG |

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | API status check |
| POST | /analyze | Full 15-feature analysis |
| POST | /qa | RAG-based question answering |
| POST | /compare | Document similarity comparison |
| GET | /docs | Swagger UI |