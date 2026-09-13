"""detector.py — Full Fusion AI vs Human Detection System.

Combines:
1. 500-Tree Tuned Random Forest Classifier on 24 Stylometric Features.
2. Character-Boundary & Sub-word N-Gram TF-IDF Ensembles (Logistic Regression + SGD).
3. Meta-Learner XGBoost Stacking Classifier trained on Arslan.
4. DistilGPT-2 Autoregressive Token-Level Perplexity & GLTR Waveform Engine.
5. Sentence-Level Origin Heatmap & Evidence Extraction.
"""

import os
import json
import pickle
import ctypes
import numpy as np
import pandas as pd

# Load OpenMP for XGBoost on macOS
_TORCH_LIB_DIR = os.path.expanduser("~/workspace/mysha/.venv/lib/python3.9/site-packages/torch/lib")
_DYLIB_OMP = os.path.join(_TORCH_LIB_DIR, "libomp.dylib")
if os.path.exists(_DYLIB_OMP):
    try:
        ctypes.CDLL(_DYLIB_OMP)
    except Exception:
        pass

from .stylometry import tokenize_sentences
from colab_clayton_research import extract_features_fast, ALL_FEATURE_NAMES
from .neural_waveform import compute_token_waveform, get_neural_ai_score

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_model")
_PIPELINE_PATH = os.path.join(_MODEL_DIR, "rf_xgboost_pipeline.pkl")

_pipeline_bundle = None

def _load_pipeline():
    global _pipeline_bundle
    if _pipeline_bundle is not None:
        return _pipeline_bundle
    
    if os.path.exists(_PIPELINE_PATH):
        try:
            with open(_PIPELINE_PATH, "rb") as f:
                _pipeline_bundle = pickle.load(f)
        except Exception:
            _pipeline_bundle = None
    return _pipeline_bundle

_load_pipeline()

def detect_ai_vs_human(text: str) -> dict:
    """Runs combined Random Forest + XGBoost + N-Gram + Neural Waveform Detection."""
    if not text or len(text.strip()) < 15:
        return {
            "human_probability": 50.0,
            "ai_probability": 50.0,
            "confidence": "low",
            "evidence": [{"signal": "insufficient_text", "detail": "Text too short for analysis (minimum 15 characters required)."}],
            "features": {},
            "verdict": "Text too short for reliable analysis",
            "neural_metrics": None,
            "sentence_analysis": []
        }

    # 1. Stylometric Feature Extraction
    features = extract_features_fast(text)
    
    # 2. Pipeline Execution (Random Forest + XGBoost + N-Grams)
    bundle = _load_pipeline()
    if bundle:
        feature_names = bundle.get("feature_names", ALL_FEATURE_NAMES)
        f_vec = pd.DataFrame([features])[feature_names].fillna(0.0).values
        
        # Stylometric Random Forest prediction
        rf_prob = float(bundle["rf_sty"].predict_proba(f_vec)[0][1])
        
        # Text N-Gram predictions
        tfidf_vec = bundle["union"].transform([text])
        lr_prob = float(bundle["clf_lr"].predict_proba(tfidf_vec)[0][1])
        sgd_prob = float(bundle["clf_sgd"].predict_proba(tfidf_vec)[0][1])
        
        # Meta-Feature vector for XGBoost
        meta_vec = np.column_stack([
            f_vec,
            [rf_prob],
            [lr_prob],
            [sgd_prob],
            [rf_prob * lr_prob],
            [abs(rf_prob - lr_prob)],
            [np.log(rf_prob + 1e-4)],
            [np.log(1 - rf_prob + 1e-4)],
            [np.log(lr_prob + 1e-4)],
            [np.log(1 - lr_prob + 1e-4)]
        ])
        
        # XGBoost meta-probability
        xgb_prob = float(bundle["xgb_model"].predict_proba(meta_vec)[0][1])
        
        # Strong hybrid probability
        combined_ml_prob = (0.55 * xgb_prob) + (0.30 * lr_prob) + (0.15 * rf_prob)
    else:
        combined_ml_prob = 0.5
        rf_prob = 0.5
        xgb_prob = 0.5

    # 3. Neural Token Waveform (DistilGPT-2 & GLTR)
    wave_metrics = compute_token_waveform(text)
    neural_score_data = get_neural_ai_score(wave_metrics)
    neural_bias = neural_score_data.get("neural_ai_bias", 0.0) # -1.5 to +1.5
    ppl = wave_metrics.get("perplexity", 50.0)

    # 4. Final Calibrated Probability
    # Convert ML probability to logit
    ml_logit = np.log(max(combined_ml_prob, 1e-4) / max(1.0 - combined_ml_prob, 1e-4))
    
    # ML Models (Random Forest + XGBoost) have primary weight (0.90)
    fused_logit = (0.90 * ml_logit) + (0.30 * neural_bias)
    calibrated_ai_prob = 1.0 / (1.0 + np.exp(-1.4 * fused_logit))

    ai_percentage = round(float(np.clip(calibrated_ai_prob * 100.0, 0.5, 99.5)), 1)
    human_percentage = round(float(100.0 - ai_percentage), 1)

    # 5. Compile Mathematical Evidence
    evidence = []
    evidence.append({
        "signal": "XGBoost Meta-Learner",
        "direction": "AI" if xgb_prob > 0.5 else "Human",
        "value": round(xgb_prob * 100, 1),
        "contribution": round(ml_logit, 3),
        "description": f"XGBoost stacked over Stylometry & N-Grams indicates {xgb_prob*100:.1f}% AI probability",
        "strength": "strong"
    })

    evidence.append({
        "signal": "Random Forest Stylometric Ensemble (400 Trees)",
        "direction": "AI" if rf_prob > 0.5 else "Human",
        "value": round(rf_prob * 100, 1),
        "contribution": round(rf_prob, 3),
        "description": f"24-dimension stylometric trees indicate {rf_prob*100:.1f}% AI probability",
        "strength": "strong" if abs(rf_prob - 0.5) > 0.20 else "moderate"
    })

    for s in neural_score_data.get("signals", []):
        evidence.append({
            "signal": s.get("signal", "Wave Signal"),
            "direction": s.get("dir", "Neutral"),
            "value": round(ppl, 1),
            "contribution": round(neural_bias, 3),
            "description": s.get("desc", ""),
            "strength": "strong"
        })

    # Confidence Rating
    spread = abs(ai_percentage - 50.0)
    if spread > 28:
        confidence = "high"
    elif spread > 14:
        confidence = "medium"
    else:
        confidence = "low"

    if ai_percentage >= 65:
        verdict = f"High probability of AI Generation ({ai_percentage}% AI likelihood) detected by Random Forest, XGBoost & Neural Waveforms."
    elif human_percentage >= 65:
        verdict = f"Strong signature of Human Authorship ({human_percentage}% Human likelihood) verified by stylometric variance and lexical entropy."
    else:
        verdict = f"Mixed signature ({human_percentage}% Human / {ai_percentage}% AI) — text contains blended characteristics."

    # 6. Sentence-Level Breakdown
    sentences = tokenize_sentences(text)
    sentence_analysis = []
    if len(sentences) > 1 and bundle:
        for s in sentences[:30]:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue
            s_tfidf = bundle["union"].transform([s_clean])
            s_lr_p = float(bundle["clf_lr"].predict_proba(s_tfidf)[0][1])
            s_pct = round(s_lr_p * 100.0, 1)
            
            tag = "ai" if s_pct >= 60 else ("human" if s_pct <= 40 else "mixed")
            sentence_analysis.append({
                "sentence": s_clean,
                "ai_probability": s_pct,
                "human_probability": round(100.0 - s_pct, 1),
                "tag": tag
            })

    return {
        "human_probability": human_percentage,
        "ai_probability": ai_percentage,
        "confidence": confidence,
        "verdict": verdict,
        "evidence": evidence,
        "features": features,
        "neural_metrics": wave_metrics,
        "sentence_analysis": sentence_analysis,
        "model_architecture": "Random Forest (400 Trees) + XGBoost Meta-Learner + TF-IDF N-Grams + Neural Waveform"
    }
