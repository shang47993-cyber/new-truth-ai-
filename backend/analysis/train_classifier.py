"""train_classifier.py — Supervised calibration trainer with L2 regularization and feature clipping."""

import os
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score

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

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0, class_weight="balanced")
    
    # LOO CV
    loo = LeaveOneOut()
    y_pred = cross_val_predict(clf, X_scaled, y, cv=loo)
    acc = accuracy_score(y, y_pred)
    print(f"Leave-One-Out Cross-Validation Accuracy: {acc:.1%}")

    clf.fit(X_scaled, y)

    # Calculate separabilities (Cohen's d)
    human_mask = y == 0
    ai_mask = y == 1
    separabilities = {}
    for i, fname in enumerate(FEATURE_NAMES):
        h_mean = np.mean(X[human_mask, i])
        h_std = np.std(X[human_mask, i])
        a_mean = np.mean(X[ai_mask, i])
        a_std = np.std(X[ai_mask, i])
        pooled = np.sqrt((h_std**2 + a_std**2) / 2) if (h_std + a_std) > 0 else 1
        separabilities[fname] = round(abs(h_mean - a_mean) / pooled, 4)

    out_dir = "/Users/sankalp/workspace/mysha/backend/analysis/trained_model"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "model_type": "Logistic Regression (C=1.0)",
        "accuracy": float(acc),
        "feature_names": FEATURE_NAMES,
        "feature_separabilities": separabilities,
        "feature_coefficients": {FEATURE_NAMES[i]: round(float(clf.coef_[0][i]), 4) for i in range(len(FEATURE_NAMES))},
        "intercept": round(float(clf.intercept_[0]), 4),
    }

    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Model trained and exported successfully.")

if __name__ == "__main__":
    train_and_export()
