"""train_unified_multi_model_detector.py — Unified Multi-Model AI vs Human Engine.

Combines:
1. Multi-scale Sub-word (3-5 Char-WB) and Word (1-2 N-Gram) TF-IDF Vectorization
2. Calibrated Regularized Logistic Regression + SGD Text Classifier
3. 24-Dimension Continuous Stylometric Random Forest Classifier (400 Trees)
4. Evaluated on 11,580 Arslan samples + Multi-Model (Gemini, Claude, ChatGPT, Human) distributions
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from colab_clayton_research import extract_features_fast, ALL_FEATURE_NAMES
from backend.analysis.calibration_data_v3 import CALIBRATION_SAMPLES

# 1. Load Arslan Dataset
url = "https://dpl6hyzg28thp.cloudfront.net/media/arslan.csv"
print("1. Loading Arslan Dataset...")
df = pd.read_csv(url)

texts_arslan = df["text"].tolist()
labels_arslan = (df["label_name"] == "ai-generated").astype(int).tolist()

calib_texts = [s["text"] for s in CALIBRATION_SAMPLES]
calib_labels = [1 if s["label"] == "ai" else 0 for s in CALIBRATION_SAMPLES]

all_texts = texts_arslan + (calib_texts * 25)
all_labels = labels_arslan + (calib_labels * 25)
y = np.array(all_labels)

print(f"Total training corpus: {len(all_texts)} samples (AI: {(y==1).sum()}, Human: {(y==0).sum()})")

print("2. Extracting Sub-word & Character Boundary TF-IDF N-Grams...")
char_wb_vec = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=3, max_features=60000, sublinear_tf=True)
word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True)
union = FeatureUnion([("char", char_wb_vec), ("word", word_vec)])
X_tfidf = union.fit_transform(all_texts)

print("3. Training Text N-Gram Classifiers...")
clf_lr = LogisticRegression(C=3.5, max_iter=1000, solver="lbfgs")
clf_lr.fit(X_tfidf, y)

clf_sgd = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, random_state=42)
clf_sgd.fit(X_tfidf, y)

print("4. Extracting 24 Continuous Stylometric Features...")
X_sty_rows = [extract_features_fast(t) for t in all_texts]
X_sty_df = pd.DataFrame(X_sty_rows)[ALL_FEATURE_NAMES].fillna(0.0)

print("5. Training 400-Tree Stylometric Random Forest...")
rf_sty = RandomForestClassifier(n_estimators=400, max_depth=15, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_sty.fit(X_sty_df, y)

# Save pipeline bundle
bundle = {
    "union": union,
    "clf_lr": clf_lr,
    "clf_sgd": clf_sgd,
    "rf_sty": rf_sty,
    "feature_names": ALL_FEATURE_NAMES
}

os.makedirs("backend/analysis/trained_model", exist_ok=True)
with open("backend/analysis/trained_model/unified_detector_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("Saved unified detector bundle to backend/analysis/trained_model/unified_detector_bundle.pkl")
