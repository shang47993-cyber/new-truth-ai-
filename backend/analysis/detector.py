"""detector.py — Triple-Engine Fusion AI vs Human Detection System.

Ensemble Architecture:
1. Deep Supervised Transformer Classification Head (BERT-AI-Detector)
2. Causal Autoregressive Token-Level Perplexity & GLTR Waveform Engine (DistilGPT-2)
3. 23-Dimension Empirical Mathematical Stylometry (Vocabulary Richness, Zipf, Sentence Variations)
"""

import os
import json
import pickle
import numpy as np

# Ensure OpenMP runtime is found if xgboost is imported or dynamically loaded
_DYLIBS = os.path.expanduser("~/workspace/mysha/.venv/lib/python3.9/site-packages/sklearn/.dylibs")
if os.path.exists(_DYLIBS):
    os.environ["DYLD_LIBRARY_PATH"] = f"{_DYLIBS}:{os.environ.get('DYLD_LIBRARY_PATH', '')}".rstrip(":")

from .stylometry import extract_features
from .neural_waveform import compute_token_waveform, get_neural_ai_score
from .transformer_detector import detect_with_transformer

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_model")

def _load_model():
    model_path = os.path.join(_MODEL_DIR, "classifier.pkl")
    scaler_path = os.path.join(_MODEL_DIR, "scaler.pkl")
    meta_path = os.path.join(_MODEL_DIR, "model_metadata.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return model, scaler, metadata

_model, _scaler, _metadata = _load_model()
FEATURE_NAMES = _metadata["feature_names"]

def detect_ai_vs_human(text: str) -> dict:
    """Runs triple-engine ensemble for high-accuracy AI vs Human text detection."""
    features = extract_features(text)
    if "error" in features:
        return {
            "human_probability": 50.0,
            "ai_probability": 50.0,
            "confidence": "low",
            "evidence": [{"signal": "insufficient_text", "detail": features["error"]}],
            "features": features,
            "verdict": "Text too short for reliable analysis",
            "neural_metrics": None
        }

    # --- ENGINE 1: Deep Supervised Transformer Head ---
    tr_res = detect_with_transformer(text)
    tr_ai_prob = tr_res["transformer_ai_prob"] # 0.0 to 1.0

    # --- ENGINE 2: Neural Token-Level Waveform (SeqXGPT / GLTR) ---
    wave_metrics = compute_token_waveform(text)
    neural_score_data = get_neural_ai_score(wave_metrics)
    neural_bias = neural_score_data["neural_ai_bias"] # -1.5 to +1.5

    # --- ENGINE 3: 23-Dimension Stylometric Classifier (Calibrated XGBoost / Trees) ---
    feature_vec = np.array([[features.get(f, 0) for f in FEATURE_NAMES]], dtype=np.float32)
    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
    feature_scaled = _scaler.transform(feature_vec)
    
    # Check if model supports predict_proba or decision_function
    if hasattr(_model, "predict_proba"):
        sty_ai_prob = float(_model.predict_proba(feature_scaled)[0][1])
        sty_logit = np.log(max(sty_ai_prob, 1e-4) / max(1.0 - sty_ai_prob, 1e-4))
    elif hasattr(_model, "decision_function"):
        sty_logit = float(_model.decision_function(feature_scaled)[0])
        sty_ai_prob = 1.0 / (1.0 + np.exp(-sty_logit))
    else:
        sty_logit = 0.0
        sty_ai_prob = 0.5

    # --- ENSEMBLE DECISION FUSION ---
    # Convert transformer probability to logit
    tr_logit = np.log(max(tr_ai_prob, 1e-4) / max(1.0 - tr_ai_prob, 1e-4))
    
    # Weighted Logit Fusion:
    # Transformer Head (60% weight) + Neural Waveform Perplexity (25% weight) + Stylometry (15% weight)
    fused_logit = (0.60 * tr_logit) + (0.90 * neural_bias) + (0.25 * sty_logit)
    
    # Sharp, calibrated sigmoid mapping
    calibrated_ai_prob = 1.0 / (1.0 + np.exp(-1.3 * fused_logit))
    
    # Final Percentages
    ai_percentage = round(float(np.clip(calibrated_ai_prob * 100.0, 0.5, 99.5)), 1)
    human_percentage = round(float(100.0 - ai_percentage), 1)

    # Compile Evidence Trail
    evidence = []
    
    # Deep Transformer Signal
    evidence.append({
        "signal": "Transformer Semantic Head",
        "direction": "AI" if tr_ai_prob > 0.5 else "Human",
        "value": round(tr_ai_prob * 100, 1),
        "contribution": round(tr_logit, 3),
        "description": f"Deep neural attention patterns indicate {tr_ai_prob*100:.1f}% AI generation likelihood",
        "strength": "strong"
    })

    # Neural Waveform Signals
    for s in neural_score_data["signals"]:
        evidence.append({
            "signal": s["signal"],
            "direction": s["dir"],
            "value": wave_metrics.get("perplexity", 0),
            "contribution": round(neural_bias, 3),
            "description": s["desc"],
            "strength": "strong"
        })

    # Stylometric Evidence
    if hasattr(_model, "feature_importances_") or hasattr(_model, "estimator_"):
        evidence.append({
            "signal": "Stylometric Tree Model (XGBoost)",
            "direction": "AI" if sty_ai_prob > 0.5 else "Human",
            "value": round(sty_ai_prob * 100, 1),
            "contribution": round(sty_logit, 3),
            "description": f"Ensemble tree stylometry indicates {sty_ai_prob*100:.1f}% AI probability",
            "strength": "strong" if abs(sty_ai_prob - 0.5) > 0.25 else "moderate"
        })
    elif hasattr(_model, "coef_"):
        coefs = _model.coef_[0]
        scaled_vals = feature_scaled[0]
        contributions = coefs * scaled_vals

        for i, fname in enumerate(FEATURE_NAMES):
            c = float(contributions[i])
            if abs(c) > 0.15:
                direction = "AI" if c > 0 else "Human"
                desc = f"{'High' if scaled_vals[i] > 0 else 'Low'} {fname.replace('_', ' ')}"
                evidence.append({
                    "signal": fname,
                    "direction": direction,
                    "value": round(float(features.get(fname, 0)), 4),
                    "contribution": round(c, 4),
                    "description": desc,
                    "strength": "strong" if abs(c) > 0.4 else "moderate"
                })

    evidence.sort(key=lambda x: abs(x.get("contribution", 0)), reverse=True)

    # Confidence Classification
    spread = abs(ai_percentage - 50.0)
    if spread > 28:
        confidence = "high"
    elif spread > 14:
        confidence = "medium"
    else:
        confidence = "low"

    if ai_percentage >= 70:
        verdict = f"High probability of AI Generation ({ai_percentage}% AI likelihood) verified across transformer representations and neural token paths."
    elif human_percentage >= 70:
        verdict = f"Strong signature of Human Authorship ({human_percentage}% Human likelihood) verified by genuine semantic surprise and stylometric variance."
    else:
        verdict = f"Mixed signature ({human_percentage}% Human / {ai_percentage}% AI) — text contains blended characteristics."

    return {
        "human_probability": human_percentage,
        "ai_probability": ai_percentage,
        "confidence": confidence,
        "evidence": evidence[:12],
        "features": features,
        "neural_metrics": wave_metrics,
        "transformer_metrics": tr_res,
        "verdict": verdict,
        "model_architecture": "Triple Ensemble Engine: Fine-Tuned BERT Sequence Head + DistilGPT-2 Neural Waveform + 23-Dim Stylometry"
    }
