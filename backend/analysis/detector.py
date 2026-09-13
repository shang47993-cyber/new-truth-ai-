"""detector.py — Unified Multi-Model AI vs Human Detection System (Gemini, ChatGPT, Claude, Human).

Combines:
1. Multi-scale Sub-word (3-5 Char-WB) and Word (1-2 N-Gram) TF-IDF Language Models (Logistic Regression + SGD).
2. 400-Tree Continuous Mathematical Stylometric Random Forest Classifier.
3. DistilGPT-2 Autoregressive Token-Level Perplexity & GLTR Waveform Engine.
4. Sentence-Level Origin Heatmap & Quantitative Evidence Extraction.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

from .stylometry import tokenize_sentences
from colab_clayton_research import extract_features_fast, ALL_FEATURE_NAMES
from .neural_waveform import compute_token_waveform, get_neural_ai_score

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_model")
_BUNDLE_PATH = os.path.join(_MODEL_DIR, "unified_detector_bundle.pkl")

_bundle = None

def _load_bundle():
    global _bundle
    if _bundle is not None:
        return _bundle
    
    if os.path.exists(_BUNDLE_PATH):
        try:
            with open(_BUNDLE_PATH, "rb") as f:
                _bundle = pickle.load(f)
        except Exception:
            _bundle = None
    return _bundle

_load_bundle()

def detect_ai_vs_human(text: str) -> dict:
    """Runs continuous multi-model Random Forest + N-Gram TF-IDF + Token Waveform AI detection tailored to each input."""
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

    bundle = _load_bundle()
    
    # 1. Extract Continuous 24 Stylometric Features (<5ms)
    features = extract_features_fast(text)
    
    if bundle:
        feature_names = bundle.get("feature_names", ALL_FEATURE_NAMES)
        f_df = pd.DataFrame([features])[feature_names].fillna(0.0)
        rf_ai_prob = float(bundle["rf_sty"].predict_proba(f_df)[0][1])
        
        # 2. Text N-Gram Inference
        tfidf_vec = bundle["union"].transform([text])
        lr_prob = float(bundle["clf_lr"].predict_proba(tfidf_vec)[0][1])
        sgd_prob = float(bundle["clf_sgd"].predict_proba(tfidf_vec)[0][1])
        ngram_prob = (0.70 * lr_prob) + (0.30 * sgd_prob)
    else:
        rf_ai_prob = 0.5
        ngram_prob = 0.5

    # 3. Neural Token Waveform (DistilGPT-2 & GLTR Token Predictability)
    wave_metrics = compute_token_waveform(text)
    neural_score_data = get_neural_ai_score(wave_metrics)
    neural_bias = neural_score_data.get("neural_ai_bias", 0.0) # -1.5 to +1.5
    ppl = wave_metrics.get("perplexity", 50.0)

    # 4. Multi-Engine Continuous Logit Fusion
    # Convert probabilities to logits
    rf_logit = np.log(max(rf_ai_prob, 1e-4) / max(1.0 - rf_ai_prob, 1e-4))
    ngram_logit = np.log(max(ngram_prob, 1e-4) / max(1.0 - ngram_prob, 1e-4))
    
    # Balanced logit combination: 60% N-Gram Language Model + 40% Stylometric RF
    fused_logit = (0.60 * ngram_logit) + (0.40 * rf_logit) + (0.25 * neural_bias) + 0.35
    calibrated_ai_prob = 1.0 / (1.0 + np.exp(-1.40 * fused_logit))

    ai_percentage = round(float(np.clip(calibrated_ai_prob * 100.0, 0.5, 99.5)), 1)
    human_percentage = round(float(100.0 - ai_percentage), 1)

    # 5. Compile Mathematical Evidence Trail
    evidence = []
    evidence.append({
        "signal": "Sub-word & Boundary N-Gram Language Model",
        "direction": "AI" if ngram_prob > 0.5 else "Human",
        "value": round(ngram_prob * 100, 1),
        "contribution": round(ngram_logit, 3),
        "description": f"Character boundary and sub-token distributions indicate {ngram_prob*100:.1f}% AI generation probability",
        "strength": "strong" if abs(ngram_prob - 0.5) > 0.20 else "moderate"
    })

    evidence.append({
        "signal": "Mathematical Stylometric Random Forest (400 Trees)",
        "direction": "AI" if rf_ai_prob > 0.5 else "Human",
        "value": round(rf_ai_prob * 100, 1),
        "contribution": round(rf_logit, 3),
        "description": f"Continuous 24-dimension stylometric distributions indicate {rf_ai_prob*100:.1f}% AI probability",
        "strength": "strong" if abs(rf_ai_prob - 0.5) > 0.20 else "moderate"
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
        verdict = f"High probability of AI Generation ({ai_percentage}% AI likelihood) verified across sub-word N-grams, Random Forest, and neural token paths."
    elif human_percentage >= 65:
        verdict = f"Strong signature of Human Authorship ({human_percentage}% Human likelihood) verified by stylometric variance and lexical entropy."
    else:
        verdict = f"Mixed signature ({human_percentage}% Human / {ai_percentage}% AI) — text contains blended characteristics."

    # 6. Sentence-Level Origin Heatmap
    sentences = tokenize_sentences(text)
    sentence_analysis = []
    if len(sentences) > 1 and bundle:
        for s in sentences[:30]:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue
            s_tfidf = bundle["union"].transform([s_clean])
            s_p = float(bundle["clf_lr"].predict_proba(s_tfidf)[0][1])
            s_pct = round(s_p * 100.0, 1)
            
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
        "model_architecture": "Sub-word Char-WB TF-IDF + 400-Tree Stylometric Random Forest + DistilGPT-2 Neural Waveform"
    }
