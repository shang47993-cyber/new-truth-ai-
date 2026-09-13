"""app.py — Main Flask Server for Mysha Hybrid Detection Engine with File Upload & Document Extraction."""

import os
import io
import uuid
import json
import logging
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.analysis.stylometry import extract_features
from backend.analysis.detector import detect_ai_vs_human
from backend.analysis.authorship import detect_authors
from backend.analysis.neural_waveform import _load_neural_engine

# Document Extractors
import pypdf
import docx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"txt", "pdf", "docx", "md", "csv"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

class NumpyJSONProvider(DefaultJSONProvider):
    """Recursively converts NumPy scalar and array types to serializable Python objects."""
    def default(self, o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
app.config["SECRET_KEY"] = os.urandom(24).hex()
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

analyses_store = {}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_storage) -> tuple[str, str]:
    """Extracts clean text and metadata from PDF, DOCX, TXT, MD, and CSV files."""
    filename = secure_filename(file_storage.filename or "uploaded_file")
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    raw_bytes = file_storage.read()

    text = ""
    if ext == "pdf":
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            extracted_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(page_text.strip())
            text = "\n\n".join(extracted_pages)
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise ValueError(f"Could not parse PDF file: {str(e)}")

    elif ext == "docx":
        try:
            doc = docx.Document(io.BytesIO(raw_bytes))
            text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            raise ValueError(f"Could not parse DOCX file: {str(e)}")

    elif ext in ["txt", "md"]:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1", errors="replace")

    elif ext == "csv":
        try:
            decoded = raw_bytes.decode("utf-8", errors="replace")
            # If CSV, join all text columns
            import pandas as pd
            df = pd.read_csv(io.StringIO(decoded))
            # Pick first text column or concat all string columns
            str_cols = df.select_dtypes(include=['object']).columns
            if len(str_cols) > 0:
                text = "\n\n".join(df[str_cols[0]].dropna().astype(str).tolist()[:50]) # First 50 samples
            else:
                text = decoded
        except Exception as e:
            text = raw_bytes.decode("utf-8", errors="replace")

    else:
        raise ValueError(f"Unsupported file format: .{ext}")

    return text.strip(), filename

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Mysha Hybrid Intelligence Platform",
        "version": "2.5.0",
        "engine": "Stylometric + DistilGPT-2 Neural Waveform + Deep Transformer Head",
        "supported_formats": list(ALLOWED_EXTENSIONS)
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text submitted for analysis."}), 400

    return process_and_analyze(text, source_name="Pasted Text")

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in the request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Selected file is empty."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        text, filename = extract_text_from_file(file)
        if not text or len(text.strip()) < 15:
            return jsonify({"error": "Extracted text from document is too short or empty for analysis."}), 400

        return process_and_analyze(text, source_name=filename)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("File upload processing error")
        return jsonify({"error": f"Failed to process uploaded file: {str(e)}"}), 500

def process_and_analyze(text: str, source_name: str = "Pasted Text"):
    try:
        # 1. Stylometric Extraction
        features = extract_features(text)
        
        # 2. Hybrid AI vs Human Detection
        detection = detect_ai_vs_human(text)
        
        # 3. Authorship & Style Clustering
        authorship = detect_authors(text)

        analysis_id = str(uuid.uuid4())[:8]
        result = {
            "id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "source": source_name,
            "text_preview": text[:300] + ("..." if len(text) > 300 else ""),
            "full_text_length": len(text),
            "word_count": features.get("word_count", len(text.split())),
            "detection": detection,
            "authorship": authorship,
            "features": features
        }

        analyses_store[analysis_id] = result
        return jsonify({"success": True, "analysis": result})

    except Exception as e:
        logger.exception("Analysis pipeline exception")
        return jsonify({"error": f"Internal Analysis Error: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info("Pre-warming Neural Waveform Engine...")
    _load_neural_engine()
    logger.info("Starting Mysha Flask Server on port 5050...")
    app.run(host="0.0.0.0", port=5050, debug=False)
