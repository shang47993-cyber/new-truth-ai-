"""train_classifier.py — Supervised calibration trainer & multi-engine meta-learner.

Trains and evaluates:
1. Multi-Engine Meta-Learner (Stacking Classifier fusing Transformer, Waveform GLTR, and Stylometry)
2. Hyperparameter-tuned XGBoost & Random Forest Stylometric Classifiers
3. Probability Calibration with Isotonic & Sigmoid scaling
4. Out-of-fold cross-validation metrics
"""

import os
import sys
import ctypes

# Explicitly ensure libomp from PyTorch is loaded for macOS ARM64 / XGBoost
_TORCH_LIB_DIR = os.path.expanduser("~/workspace/mysha/.venv/lib/python3.9/site-packages/torch/lib")
_DYLIB_OMP = os.path.join(_TORCH_LIB_DIR, "libomp.dylib")
if os.path.exists(_DYLIB_OMP):
    try:
        ctypes.CDLL(_DYLIB_OMP)
    except Exception:
        pass

if os.path.exists(_TORCH_LIB_DIR):
    current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _TORCH_LIB_DIR not in current_dyld:
        os.environ["DYLD_LIBRARY_PATH"] = f"{_TORCH_LIB_DIR}:{current_dyld}" if current_dyld else _TORCH_LIB_DIR

import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from backend.analysis.stylometry import extract_features
from backend.analysis.neural_waveform import compute_token_waveform, get_neural_ai_score
from backend.analysis.transformer_detector import detect_with_transformer
from backend.analysis.calibration_data_v3 import CALIBRATION_SAMPLES

FEATURE_NAMES = [
    "type_token_ratio",
    "hapax_legomena_ratio",
    "dis_legomena_ratio",
    "yules_k",
    "simpsons_diversity",
    "honores_r",
    "entropy",
    "avg_sentence_length",
    "std_sentence_length",
    "cv_sentence_length",
    "avg_word_length",
    "std_word_length",
    "punctuation_ratio",
    "comma_ratio",
    "semicolon_ratio",
    "exclamation_ratio",
    "question_ratio",
    "function_word_ratio",
    "zipf_coefficient",
    "zipf_r_squared",
    "burstiness",
    "cv_paragraph_length",
    "contraction_ratio",
]

META_FEATURE_NAMES = [
    "tr_ai_prob",
    "neural_bias",
    "perplexity",
    "gltr_top10",
    "gltr_tail1000",
    "wave_volatility",
    "sty_ai_prob"
]

def build_multi_engine_dataset():
    """Extracts features across all 3 engines for every calibration sample."""
    print("Extracting multi-engine features across calibration dataset...")
    X_sty = []
    X_meta = []
    y = []

    for idx, s in enumerate(CALIBRATION_SAMPLES):
        text = s["text"]
        label = 1 if s["label"] == "ai" else 0
        
        # 1. Stylometry
        feats = extract_features(text)
        if "error" in feats:
            continue
        vec_sty = [feats.get(f, 0.0) for f in FEATURE_NAMES]
        
        # 2. Neural Waveform & GLTR
        wave = compute_token_waveform(text)
        neural_score = get_neural_ai_score(wave)
        neural_bias = neural_score.get("neural_ai_bias", 0.0)
        ppl = wave.get("perplexity", 50.0)
        gltr_10 = wave.get("gltr_top10", 0.5)
        gltr_tail = wave.get("gltr_tail1000", 0.0)
        vol = wave.get("wave_volatility", 2.0)

        # 3. Transformer Head
        tr_res = detect_with_transformer(text)
        tr_ai_p = tr_res.get("transformer_ai_prob", 0.5)

        X_sty.append(vec_sty)
        # Raw meta features before stylometric probability is filled in
        X_meta.append([tr_ai_p, neural_bias, ppl, gltr_10, gltr_tail, vol])
        y.append(label)

    X_sty = np.array(X_sty, dtype=np.float32)
    X_meta_raw = np.array(X_meta, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X_sty, X_meta_raw, y

def train_and_optimize():
    print("=" * 70)
    print("MYSHA AI DETECTOR — ADVANCED MULTI-ENGINE META-LEARNER TRAINING")
    print("=" * 70)

    X_sty, X_meta_raw, y = build_multi_engine_dataset()
    print(f"Loaded {len(y)} samples ({np.sum(y == 0)} Human, {np.sum(y == 1)} AI).")

    scaler_sty = StandardScaler()
    X_sty_scaled = scaler_sty.fit_transform(X_sty)

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # -------------------------------------------------------------
    # 1. Hyperparameter Tuning for Stylometric XGBoost & RF
    # -------------------------------------------------------------
    print("\n[1/4] Tuning Stylometric Tree Classifiers...")
    xgb_param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.08, 0.12],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [0.5, 1.0]
    }
    xgb_grid = GridSearchCV(
        estimator=xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        param_grid=xgb_param_grid,
        cv=cv_inner,
        scoring="f1",
        n_jobs=-1
    )
    xgb_grid.fit(X_sty_scaled, y)
    best_xgb = xgb_grid.best_estimator_

    calibrated_xgb = CalibratedClassifierCV(estimator=best_xgb, method="sigmoid", cv=3)
    calibrated_xgb.fit(X_sty_scaled, y)

    # Get out-of-fold probability predictions for stylometry
    sty_oof_probs = cross_val_predict(calibrated_xgb, X_sty_scaled, y, cv=cv_outer, method="predict_proba")[:, 1]
    
    acc_sty = accuracy_score(y, (sty_oof_probs > 0.5).astype(int))
    f1_sty = f1_score(y, (sty_oof_probs > 0.5).astype(int))
    print(f"  → Stylometric XGBoost 5-Fold CV Accuracy: {acc_sty:.1%}, F1: {f1_sty:.3f}")

    # -------------------------------------------------------------
    # 2. Build Multi-Engine Feature Matrix
    # -------------------------------------------------------------
    print("\n[2/4] Constructing Full Multi-Engine Feature Matrix (Transformer + GLTR + Stylometry)...")
    # Combine raw meta features with stylometric probability
    X_meta = np.hstack([X_meta_raw, sty_oof_probs.reshape(-1, 1)])
    
    scaler_meta = StandardScaler()
    X_meta_scaled = scaler_meta.fit_transform(X_meta)

    # -------------------------------------------------------------
    # 3. Train & Evaluate Meta-Learner Classifier (Stacking)
    # -------------------------------------------------------------
    print("\n[3/4] Training Multi-Engine Meta-Learner (Logistic Regression / Stacking)...")
    meta_param_grid = {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"]
    }
    meta_grid = GridSearchCV(
        estimator=LogisticRegression(random_state=42, class_weight="balanced"),
        param_grid=meta_param_grid,
        cv=cv_inner,
        scoring="roc_auc",
        n_jobs=-1
    )
    meta_grid.fit(X_meta_scaled, y)
    best_meta_clf = meta_grid.best_estimator_

    # Evaluate Meta-Learner with Out-of-Fold Cross Validation
    meta_oof_probs = cross_val_predict(best_meta_clf, X_meta_scaled, y, cv=cv_outer, method="predict_proba")[:, 1]
    meta_oof_preds = (meta_oof_probs > 0.5).astype(int)

    acc_meta = accuracy_score(y, meta_oof_preds)
    f1_meta = f1_score(y, meta_oof_preds)
    roc_meta = roc_auc_score(y, meta_oof_probs)

    print(f"  ★ Multi-Engine Meta-Learner 5-Fold CV Accuracy: {acc_meta:.1%}")
    print(f"  ★ Multi-Engine Meta-Learner 5-Fold CV F1:       {f1_meta:.3f}")
    print(f"  ★ Multi-Engine Meta-Learner 5-Fold CV ROC-AUC:  {roc_meta:.3f}")

    # Meta-Learner Feature Weights
    best_meta_clf.fit(X_meta_scaled, y)
    meta_coefs = best_meta_clf.coef_[0]
    print("\n  Meta-Learner Learned Weights per Engine Signal:")
    for fname, w in zip(META_FEATURE_NAMES, meta_coefs):
        print(f"    - {fname:18s}: {w:+.4f}")

    # -------------------------------------------------------------
    # 4. Serialize Models, Scalers and Metadata
    # -------------------------------------------------------------
    print("\n[4/4] Serializing Model Artifacts...")
    out_dir = os.path.join(os.path.dirname(__file__), "trained_model")
    os.makedirs(out_dir, exist_ok=True)

    # Stylometric classifier
    with open(os.path.join(out_dir, "classifier.pkl"), "wb") as f:
        pickle.dump(calibrated_xgb, f)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler_sty, f)

    # Meta-Learner model & scaler
    with open(os.path.join(out_dir, "meta_classifier.pkl"), "wb") as f:
        pickle.dump(best_meta_clf, f)
    with open(os.path.join(out_dir, "meta_scaler.pkl"), "wb") as f:
        pickle.dump(scaler_meta, f)

    meta_info = {
        "model_type": "Multi-Engine Stacking Meta-Learner (Transformer + GLTR + XGBoost Stylometry)",
        "cv_accuracy_meta": float(acc_meta),
        "cv_f1_meta": float(f1_meta),
        "cv_roc_auc_meta": float(roc_meta),
        "cv_accuracy_sty": float(acc_sty),
        "feature_names": FEATURE_NAMES,
        "meta_feature_names": META_FEATURE_NAMES,
        "meta_weights": {f: round(float(w), 4) for f, w in zip(META_FEATURE_NAMES, meta_coefs)},
        "sample_count": len(y)
    }

    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"All artifacts successfully exported to {out_dir}")
    return meta_info

if __name__ == "__main__":
    train_and_optimize()
