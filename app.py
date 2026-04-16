"""
app.py
------
AI Document Analyzer — Flask REST API
Endpoints:
  GET  /health        → API status
  POST /analyze       → Full analysis (all 15 features)
  POST /qa            → Question answering (chat with doc)
  POST /compare       → Compare 2 documents
  GET  /docs          → Swagger UI
"""

import os
import time
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger

from modules.extractor  import extract
from modules.ai_engine  import summarize, answer_question, analyze_sentiment, detect_language
from modules.nlp_tools  import extract_entities, extract_keywords, classify_topic
from modules.comparator import compare_docs
from modules.metadata   import compute_metadata

# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────

app = Flask(__name__, static_folder="static")
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ─────────────────────────────────────────────
# SWAGGER CONFIG
# ─────────────────────────────────────────────

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route":    "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs",
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title":       "AI Document Analyzer API",
        "description": "Multi-format document analysis: PDF, DOCX, Image OCR + 15 AI features",
        "version":     "1.0.0",
        "contact": {
            "name":  "Khanishka",
            "url":   "https://github.com/techcode20/AI-powered-document-analyzer"
        }
    },
    "basePath":  "/",
    "schemes":   ["http", "https"],
    "consumes":  ["multipart/form-data", "application/json"],
    "produces":  ["application/json"],
    "tags": [
        {"name": "Health",   "description": "API status"},
        {"name": "Analysis", "description": "Document analysis endpoints"},
        {"name": "Chat",     "description": "Question answering"},
        {"name": "Compare",  "description": "Document comparison"},
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "png", "jpg", "jpeg", "bmp", "tiff", "webp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        return "pdf"
    elif ext in ("docx", "doc"):
        return "docx"
    else:
        return "image"

def error_response(message: str, code: int = 400) -> tuple:
    return jsonify({
        "success": False,
        "error":   message
    }), code

def success_response(data: dict) -> tuple:
    return jsonify({
        "success": True,
        **data
    }), 200


# ─────────────────────────────────────────────
# ROUTE 1 — HEALTH CHECK
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """
    API Health Check
    ---
    tags:
      - Health
    responses:
      200:
        description: API is running
        schema:
          properties:
            success:  { type: boolean, example: true }
            status:   { type: string,  example: ok }
            version:  { type: string,  example: 1.0.0 }
    """
    return success_response({
        "status":  "ok",
        "version": "1.0.0",
        "message": "AI Document Analyzer is running 🚀"
    })


# ─────────────────────────────────────────────
# ROUTE 2 — FULL ANALYSIS
# ─────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Full Document Analysis — All 15 Features
    ---
    tags:
      - Analysis
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: PDF, DOCX, or Image file to analyze
    responses:
      200:
        description: Complete analysis result
      400:
        description: Bad request (missing file or unsupported format)
      500:
        description: Internal server error
    """
    global_start = time.time()

    # ── Validate file ──────────────────────────────────────────────
    if "file" not in request.files:
        return error_response("No file provided. Send file in multipart/form-data.")

    file = request.files["file"]
    if file.filename == "":
        return error_response("Empty filename.")
    if not allowed_file(file.filename):
        return error_response(
            f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        file_bytes = file.read()
        file_type  = get_file_type(file.filename)

        # ── STEP 1: Extract text ───────────────────────────────────
        extraction = extract(file_bytes, file_type)
        text = extraction.get("text", "")

        if not text or len(text.strip()) < 10:
            return error_response(
                "Could not extract meaningful text from the document. "
                "Check if the file is readable or not a scanned image."
            )

        # ── STEP 2: Metadata ───────────────────────────────────────
        extra_meta = {k: v for k, v in extraction.items() if k != "text"}
        meta = compute_metadata(text, extra=extra_meta)

        # ── STEP 3: AI Features ────────────────────────────────────
        summary   = summarize(text)
        sentiment = analyze_sentiment(text)
        language  = detect_language(text)
        entities  = extract_entities(text)
        keywords  = extract_keywords(text, top_n=15)
        topic     = classify_topic(text)

        # ── STEP 4: Tables (PDF only) ──────────────────────────────
        tables = extraction.get("tables", [])

        # ── STEP 5: Total time ─────────────────────────────────────
        total_ms = round((time.time() - global_start) * 1000, 2)

        return success_response({
            "filename":    file.filename,
            "file_type":   file_type,
            "metadata":    meta,
            "summary":     summary,
            "sentiment":   sentiment,
            "language":    language,
            "entities":    entities,
            "keywords":    keywords,
            "topic":       topic,
            "tables":      tables,
            "total_processing_time_ms": total_ms
        })

    except Exception as e:
        traceback.print_exc()
        return error_response(f"Analysis failed: {str(e)}", code=500)


# ─────────────────────────────────────────────
# ROUTE 3 — QUESTION ANSWERING
# ─────────────────────────────────────────────

@app.route("/qa", methods=["POST"])
def qa():
    """
    Chat With Your Document — Question Answering
    ---
    tags:
      - Chat
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: PDF, DOCX, or Image file to query
      - name: question
        in: formData
        type: string
        required: true
        description: Your question about the document
    responses:
      200:
        description: Answer with confidence score
      400:
        description: Missing file or question
      500:
        description: Internal server error
    """
    # ── Validate ───────────────────────────────────────────────────
    if "file" not in request.files:
        return error_response("No file provided.")
    if "question" not in request.form or not request.form["question"].strip():
        return error_response("No question provided. Add 'question' field.")

    file     = request.files["file"]
    question = request.form["question"].strip()

    if not allowed_file(file.filename):
        return error_response("Unsupported file type.")

    try:
        file_bytes = file.read()
        file_type  = get_file_type(file.filename)

        extraction = extract(file_bytes, file_type)
        text = extraction.get("text", "")

        if not text or len(text.strip()) < 10:
            return error_response("Could not extract text from document.")

        result = answer_question(text, question)

        return success_response({
            "question":  question,
            "answer":    result["answer"],
            "confidence": result["confidence"],
            "processing_time_ms": result["processing_time_ms"]
        })

    except Exception as e:
        traceback.print_exc()
        return error_response(f"QA failed: {str(e)}", code=500)


# ─────────────────────────────────────────────
# ROUTE 4 — DOCUMENT COMPARISON
# ─────────────────────────────────────────────

@app.route("/compare", methods=["POST"])
def compare():
    """
    Compare Two Documents for Similarity
    ---
    tags:
      - Compare
    consumes:
      - multipart/form-data
    parameters:
      - name: file1
        in: formData
        type: file
        required: true
        description: First document (PDF, DOCX, or Image)
      - name: file2
        in: formData
        type: file
        required: true
        description: Second document (PDF, DOCX, or Image)
    responses:
      200:
        description: Similarity score and diff analysis
      400:
        description: Missing files
      500:
        description: Internal server error
    """
    if "file1" not in request.files or "file2" not in request.files:
        return error_response("Both 'file1' and 'file2' are required.")

    file1 = request.files["file1"]
    file2 = request.files["file2"]

    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        return error_response("Unsupported file type in one or both files.")

    try:
        bytes1 = file1.read()
        bytes2 = file2.read()

        text1 = extract(bytes1, get_file_type(file1.filename)).get("text", "")
        text2 = extract(bytes2, get_file_type(file2.filename)).get("text", "")

        if not text1 or not text2:
            return error_response("Could not extract text from one or both documents.")

        result = compare_docs(text1, text2)

        return success_response({
            "file1": file1.filename,
            "file2": file2.filename,
            **result
        })

    except Exception as e:
        traceback.print_exc()
        return error_response(f"Comparison failed: {str(e)}", code=500)


# ─────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend UI"""
    return app.send_static_file("index.html")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 AI Document Analyzer API starting...")
    print("📚 Swagger docs → http://localhost:5001/docs")
    print("🏠 Frontend     → http://localhost:5001/")
    print("❤️  Health check → http://localhost:5001/health\n")
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,    # Keep False in Colab
        use_reloader=False
    )
