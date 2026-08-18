"""train_classifier.py — Supervised calibration trainer for Stylometric AI Detection.

Trains and evaluates:
1. Random Forest (Clayton's baseline ~85%)
2. XGBoost Classifier (Clayton's top performer ~87%)
3. Calibrated Probabilistic Ensemble with feature scaling and persistence.
"""

import os
import sys

# Ensure libomp from torch / sklearn is visible for macOS dynamic loader
_TORCH_LIB = os.path.expanduser("~/workspace/mysha/.venv/lib/python3.9/site-packages/sklearn/.dylibs")
if os.path.exists(_TORCH_LIB):
    current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _TORCH_LIB not in current_dyld:
        os.environ["DYLD_LIBRARY_PATH"] = f"{_TORCH_LIB}:{current_dyld}" if current_dyld else _TORCH_LIB

import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from backend.analysis.stylometry import extract_features
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

def train_and_export():
    X = []
    y = []
    for s in CALIBRATION_SAMPLES:
        feats = extract_features(s["text"])
        if "error" in feats:
            continue
        vec = [feats.get(f, 0) for f in FEATURE_NAMES]
        X.append(vec)
        y.append(1 if s["label"] == "ai" else 0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Feature scaling (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )

    # 2. XGBoost Classifier (Clayton's refined choice)
    xgb_base = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    
    # Wrap in Probability Calibration
    calibrated_xgb = CalibratedClassifierCV(estimator=xgb_base, method="sigmoid", cv=3)

    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_rf = cross_val_predict(rf_clf, X_scaled, y, cv=cv)
    y_pred_xgb = cross_val_predict(calibrated_xgb, X_scaled, y, cv=cv)

    acc_rf = accuracy_score(y, y_pred_rf)
    acc_xgb = accuracy_score(y, y_pred_xgb)
    f1_xgb = f1_score(y, y_pred_xgb)

    print(f"Random Forest 5-Fold CV Accuracy: {acc_rf:.1%}")
    print(f"Calibrated XGBoost 5-Fold CV Accuracy: {acc_xgb:.1%} (F1: {f1_xgb:.3f})")

    # Fit final calibrated model on all training data
    calibrated_xgb.fit(X_scaled, y)
    rf_clf.fit(X_scaled, y)

    # Feature Importances from base XGB / RF
    xgb_base.fit(X_scaled, y)
    importances = xgb_base.feature_importances_
    feat_importance_dict = {
        FEATURE_NAMES[i]: round(float(importances[i]), 4) for i in range(len(FEATURE_NAMES))
    }

    out_dir = os.path.join(os.path.dirname(__file__), "trained_model")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "classifier.pkl"), "wb") as f:
        pickle.dump(calibrated_xgb, f)
    with open(os.path.join(out_dir, "rf_baseline.pkl"), "wb") as f:
        pickle.dump(rf_clf, f)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "model_type": "Calibrated XGBoost Classifier (with Random Forest Baseline)",
        "cv_accuracy_xgb": float(acc_xgb),
        "cv_accuracy_rf": float(acc_rf),
        "feature_names": FEATURE_NAMES,
        "feature_importances": feat_importance_dict,
        "sample_count": len(X)
    }

    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Models and scalers serialized successfully to {out_dir}")

if __name__ == "__main__":
    train_and_export()
