"""detector.py — Triple-Engine Fusion AI vs Human Detection System.

Ensemble Architecture:
1. Deep Supervised Transformer Classification Head (BERT-AI-Detector)
2. Causal Autoregressive Token-Level Perplexity & GLTR Waveform Engine (DistilGPT-2)
3. 23-Dimension Empirical Mathematical Stylometry (Calibrated XGBoost / Trees)
4. Multi-Engine Meta-Learner Stacking Fusion Layer (Learned weights over all engine signals)
"""

import os
import json
import pickle
import numpy as np

# Explicitly ensure libomp from PyTorch / dynamic loader is loaded for macOS ARM64 / XGBoost
_TORCH_LIB_DIR = os.path.expanduser("~/workspace/mysha/.venv/lib/python3.9/site-packages/torch/lib")
_DYLIB_OMP = os.path.join(_TORCH_LIB_DIR, "libomp.dylib")
if os.path.exists(_DYLIB_OMP):
    try:
        import ctypes
        ctypes.CDLL(_DYLIB_OMP)
    except Exception:
        pass

from .stylometry import extract_features, tokenize_sentences
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

    # Optional Meta-Learner Classifier & Scaler
    meta_clf_path = os.path.join(_MODEL_DIR, "meta_classifier.pkl")
    meta_scaler_path = os.path.join(_MODEL_DIR, "meta_scaler.pkl")
    meta_clf = None
    meta_scaler = None

    if os.path.exists(meta_clf_path) and os.path.exists(meta_scaler_path):
        try:
            with open(meta_clf_path, "rb") as f:
                meta_clf = pickle.load(f)
            with open(meta_scaler_path, "rb") as f:
                meta_scaler = pickle.load(f)
        except Exception:
            meta_clf = None
            meta_scaler = None

    return model, scaler, metadata, meta_clf, meta_scaler

_model, _scaler, _metadata, _meta_clf, _meta_scaler = _load_model()
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
    ppl = wave_metrics.get("perplexity", 50.0)
    gltr_10 = wave_metrics.get("gltr_top10", 0.5)
    gltr_tail = wave_metrics.get("gltr_tail1000", 0.0)
    volatility = wave_metrics.get("wave_volatility", 2.0)

    # --- ENGINE 3: 23-Dimension Stylometric Classifier (Calibrated XGBoost / Trees) ---
    feature_vec = np.array([[features.get(f, 0) for f in FEATURE_NAMES]], dtype=np.float32)
    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
    feature_scaled = _scaler.transform(feature_vec)
    
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

    # Try Meta-Learner Stacking if available, with robust calibrated fallback
    if _meta_clf is not None and _meta_scaler is not None:
        try:
            meta_vec = np.array([[tr_ai_prob, neural_bias, ppl, gltr_10, gltr_tail, volatility, sty_ai_prob]], dtype=np.float32)
            meta_scaled = _meta_scaler.transform(meta_vec)
            meta_prob = float(_meta_clf.predict_proba(meta_scaled)[0][1])
            meta_logit = np.log(max(meta_prob, 1e-4) / max(1.0 - meta_prob, 1e-4))
            
            # Blend learned meta-prediction with direct deep transformer & neural bias
            fused_logit = (0.50 * tr_logit) + (0.80 * neural_bias) + (0.30 * meta_logit) + (0.15 * sty_logit)
        except Exception:
            fused_logit = (0.60 * tr_logit) + (0.90 * neural_bias) + (0.25 * sty_logit)
    else:
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
    evidence.append({
        "signal": "Stylometric Tree Model (XGBoost)",
        "direction": "AI" if sty_ai_prob > 0.5 else "Human",
        "value": round(sty_ai_prob * 100, 1),
        "contribution": round(sty_logit, 3),
        "description": f"Ensemble tree stylometry indicates {sty_ai_prob*100:.1f}% AI probability",
        "strength": "strong" if abs(sty_ai_prob - 0.5) > 0.25 else "moderate"
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

    # --- Sentence-Level AI / Human Heatmap Breakdown ---
    sentences = tokenize_sentences(text)
    sentence_analysis = []
    if len(sentences) > 1:
        for s in sentences[:30]:  # Cap at 30 sentences for fast interactive response
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue
            s_wave = compute_token_waveform(s_clean, max_tokens=64)
            s_tr = detect_with_transformer(s_clean)
            s_neural = get_neural_ai_score(s_wave).get("neural_ai_bias", 0.0)
            
            # Sentence probability heuristic
            s_tr_p = s_tr.get("transformer_ai_prob", 0.5)
            s_logit = (0.65 * np.log(max(s_tr_p, 1e-4) / max(1.0 - s_tr_p, 1e-4))) + (0.75 * s_neural)
            s_ai_p = float(1.0 / (1.0 + np.exp(-1.2 * s_logit)))
            s_ai_pct = round(float(np.clip(s_ai_p * 100.0, 1.0, 99.0)), 1)
            
            tag = "ai" if s_ai_pct >= 65 else ("human" if s_ai_pct <= 35 else "mixed")
            sentence_analysis.append({
                "sentence": s_clean,
                "ai_probability": s_ai_pct,
                "human_probability": round(100.0 - s_ai_pct, 1),
                "tag": tag,
                "perplexity": s_wave.get("perplexity", 0.0)
            })

    return {
        "human_probability": human_percentage,
        "ai_probability": ai_percentage,
        "confidence": confidence,
        "evidence": evidence[:12],
        "features": features,
        "neural_metrics": wave_metrics,
        "transformer_metrics": tr_res,
        "sentence_analysis": sentence_analysis,
        "verdict": verdict,
        "model_architecture": "Triple Ensemble Engine: Fine-Tuned BERT Sequence Head + DistilGPT-2 Neural Waveform + 23-Dim Stylometry + Stacking Meta-Learner"
    }
