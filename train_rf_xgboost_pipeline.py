import os
import sys
import ctypes

_TORCH_LIB_DIR = os.path.expanduser("~/workspace/mysha/.venv/lib/python3.9/site-packages/torch/lib")
_DYLIB_OMP = os.path.join(_TORCH_LIB_DIR, "libomp.dylib")
if os.path.exists(_DYLIB_OMP):
    ctypes.CDLL(_DYLIB_OMP)

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from colab_clayton_research import extract_features_fast, ALL_FEATURE_NAMES
from backend.analysis.calibration_data_v3 import CALIBRATION_SAMPLES

url = "https://dpl6hyzg28thp.cloudfront.net/media/arslan.csv"
print("Loading Arslan Dataset (11,580 samples) + Multi-Domain Creative Fiction/Memoir Samples...")
df = pd.read_csv(url)

# Label: 1 = AI-Generated, 0 = Human-written
arslan_texts = df["text"].tolist()
arslan_labels = (df["label_name"] == "ai-generated").astype(int).tolist()

calib_texts = [s["text"] for s in CALIBRATION_SAMPLES]
calib_labels = [1 if s["label"] == "ai" else 0 for s in CALIBRATION_SAMPLES]

# Balance formal and creative fiction/memoir domains
all_texts = arslan_texts + (calib_texts * 60)
all_labels = arslan_labels + (calib_labels * 60)
y = np.array(all_labels)

print(f"Total training corpus: {len(all_texts)} samples across formal & creative domains")

print("1. Extracting 24 Stylometric Features...")
sty_rows = [extract_features_fast(t) for t in all_texts]
X_sty_df = pd.DataFrame(sty_rows)[ALL_FEATURE_NAMES].fillna(0.0)
X_sty = X_sty_df.values

print("2. Extracting Word & Sub-word Character Boundary TF-IDF N-Grams...")
word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
char_vec = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=4, max_features=40000, sublinear_tf=True)

union = FeatureUnion([
    ("word", word_vec),
    ("char", char_vec)
])

X_tfidf = union.fit_transform(all_texts)
print(f"TF-IDF Matrix shape: {X_tfidf.shape}")

print("3. Training 500-Tree Tuned Multi-Domain Random Forest...")
rf_sty = RandomForestClassifier(n_estimators=450, max_depth=16, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_sty.fit(X_sty, y)

clf_lr = LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs")
clf_lr.fit(X_tfidf, y)

clf_sgd = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, random_state=42)
clf_sgd.fit(X_tfidf, y)

print("4. Training Meta-Learner XGBoost on RF + Stylometry + N-Gram Logits...")
rf_probs = rf_sty.predict_proba(X_sty)[:, 1]
lr_probs = clf_lr.predict_proba(X_tfidf)[:, 1]
sgd_probs = clf_sgd.predict_proba(X_tfidf)[:, 1]

X_meta = np.column_stack([
    X_sty,
    rf_probs,
    lr_probs,
    sgd_probs,
    rf_probs * lr_probs,
    np.abs(rf_probs - lr_probs),
    np.log(rf_probs + 1e-4),
    np.log(1 - rf_probs + 1e-4),
    np.log(lr_probs + 1e-4),
    np.log(1 - lr_probs + 1e-4)
])

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_meta, y)

final_preds_proba = xgb_model.predict_proba(X_meta)[:, 1]
final_preds = (final_preds_proba >= 0.5).astype(int)

print(f"★ Pipeline Accuracy: {accuracy_score(y, final_preds)*100:.2f}% | ROC-AUC: {roc_auc_score(y, final_preds_proba):.4f}")

# Save complete bundle
pipeline_bundle = {
    "union": union,
    "rf_sty": rf_sty,
    "clf_lr": clf_lr,
    "clf_sgd": clf_sgd,
    "xgb_model": xgb_model,
    "feature_names": ALL_FEATURE_NAMES
}

os.makedirs("backend/analysis/trained_model", exist_ok=True)
with open("backend/analysis/trained_model/rf_xgboost_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline_bundle, f)

print("Saved updated multi-domain pipeline to backend/analysis/trained_model/rf_xgboost_pipeline.pkl")
