"""app.py — Main Flask Server for Mysha Hybrid Detection Engine."""

import os
import uuid
import json
import logging
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

from backend.analysis.stylometry import extract_features
from backend.analysis.detector import detect_ai_vs_human
from backend.analysis.authorship import detect_authors
from backend.analysis.neural_waveform import _load_neural_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
CORS(app)

analyses_store = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Mysha Hybrid Intelligence Platform",
        "version": "2.0.0",
        "engine": "Stylometric + DistilGPT-2 Neural Waveform"
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text submitted for analysis."}), 400

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
            "text_preview": text[:200] + ("..." if len(text) > 200 else ""),
            "word_count": features.get("word_count", 0),
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
    app.run(host="127.0.0.1", port=5050, debug=False)
