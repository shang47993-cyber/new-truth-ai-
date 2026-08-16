"""authorship.py — Multi-Author Segmentation and Statistical Clustering Engine."""

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .stylometry import extract_features, tokenize_words, tokenize_sentences


def segment_text(text: str, n_segments: int = 8) -> list[dict]:
    """Dynamically slices text into rolling segments for multi-author detection."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) >= 4:
        return [{"text": p, "label": f"Paragraph {i+1}", "index": i} for i, p in enumerate(paragraphs)]

    sentences = tokenize_sentences(text)
    if len(sentences) >= 6:
        chunk_size = max(2, len(sentences) // n_segments)
        segments = []
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            if chunk:
                segments.append({
                    "text": ". ".join(chunk),
                    "label": f"Sentences {i+1}-{i+len(chunk)}",
                    "index": len(segments)
                })
        return segments

    words = tokenize_words(text)
    chunk_size = max(40, len(words) // n_segments)
    raw_words = text.split()
    segments = []
    for i in range(0, len(raw_words), chunk_size):
        chunk = raw_words[i:i + chunk_size]
        if len(chunk) >= 20:
            segments.append({
                "text": " ".join(chunk),
                "label": f"Words {i+1}-{i+len(chunk)}",
                "index": len(segments)
            })
    return segments


def detect_authors(text: str) -> dict:
    """Analyzes text for multi-author signatures via K-Means and Hierarchical Clustering."""
    words = tokenize_words(text)
    if len(words) < 80:
        return {
            "predicted_authors": 1,
            "confidence": "low",
            "verdict": "Text too short for multi-author segmentation (minimum 80 words required).",
            "evidence": [],
            "segments": [],
            "dendrogram_data": None,
            "style_shifts": []
        }

    segments = segment_text(text)
    if len(segments) < 3:
        return {
            "predicted_authors": 1,
            "confidence": "low",
            "verdict": "Single author profile detected (insufficient segment variance).",
            "evidence": [],
            "segments": [{"label": s["label"], "features": {}} for s in segments],
            "dendrogram_data": None,
            "style_shifts": []
        }

    key_features = [
        "type_token_ratio", "hapax_legomena_ratio", "yules_k",
        "avg_sentence_length", "cv_sentence_length", "avg_word_length",
        "std_word_length", "punctuation_ratio", "comma_ratio",
        "function_word_ratio", "contraction_ratio", "entropy"
    ]

    feature_matrix = []
    valid_segments = []
    for seg in segments:
        feats = extract_features(seg["text"])
        if "error" not in feats:
            feature_matrix.append([feats.get(f, 0) for f in key_features])
            valid_segments.append(seg)

    if len(feature_matrix) < 3:
        return {
            "predicted_authors": 1,
            "confidence": "low",
            "verdict": "Single consistent writing style.",
            "evidence": [],
            "segments": [],
            "dendrogram_data": None,
            "style_shifts": []
        }

    X = np.array(feature_matrix)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate K-Means across k in [2, min(5, N-1)]
    max_k = min(5, len(X) - 1)
    silhouette_scores = {}
    cluster_assignments = {}

    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            silhouette_scores[k] = round(float(sil), 4)
            cluster_assignments[k] = labels.tolist()
        except Exception:
            continue

    best_k = 1
    best_sil = 0.0
    if silhouette_scores:
        candidate_k = max(silhouette_scores, key=silhouette_scores.get)
        candidate_sil = silhouette_scores[candidate_k]
        # Require clear separation for multi-author declaration (> 0.28)
        if candidate_sil >= 0.28:
            best_k = candidate_k
            best_sil = candidate_sil

    # Hierarchical Ward Linkage
    try:
        distances = pdist(X_scaled, metric='euclidean')
        linkage_matrix = linkage(distances, method='ward')
        dendrogram_data = {
            "linkage_matrix": linkage_matrix.tolist(),
            "labels": [s["label"] for s in valid_segments],
            "n_segments": len(X)
        }
        if best_k > 1:
            hier_labels = fcluster(linkage_matrix, t=best_k, criterion='maxclust').tolist()
        else:
            hier_labels = [1] * len(X)
    except Exception:
        dendrogram_data = None
        hier_labels = [1] * len(X)

    # Style Shift Outlier Detection
    style_shifts = []
    if len(X) > 2:
        diffs = np.linalg.norm(np.diff(X_scaled, axis=0), axis=1)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs) if len(diffs) > 1 else 1.0
        for i, d in enumerate(diffs):
            if d > mean_diff + 1.5 * std_diff:
                style_shifts.append({
                    "position": i + 1,
                    "between": f"{valid_segments[i]['label']} → {valid_segments[i+1]['label']}",
                    "shift_magnitude": round(float(d), 2),
                    "description": f"Significant style disruption between {valid_segments[i]['label']} and {valid_segments[i+1]['label']}"
                })

    # Segment details for UI
    assigned_clusters = cluster_assignments.get(best_k, [0] * len(X))
    seg_details = []
    for i, seg in enumerate(valid_segments):
        seg_details.append({
            "label": seg["label"],
            "features": {key_features[j]: round(float(X[i, j]), 3) for j in range(len(key_features))},
            "cluster": int(assigned_clusters[i]) if i < len(assigned_clusters) else 0,
            "hier_cluster": int(hier_labels[i]) if i < len(hier_labels) else 1
        })

    confidence = "high" if best_sil > 0.45 else "medium" if best_sil > 0.28 else "high" if best_k == 1 else "low"
    
    if best_k == 1:
        verdict = f"Consistent single-author profile throughout the text (uniform stylistic coherence)."
    else:
        verdict = f"Distinct authorship shifts detected: likely composed by {best_k} separate contributors."

    return {
        "predicted_authors": best_k,
        "confidence": confidence,
        "verdict": verdict,
        "silhouette_scores": silhouette_scores,
        "best_silhouette": round(best_sil, 3),
        "segments": seg_details,
        "dendrogram_data": dendrogram_data,
        "style_shifts": style_shifts
    }
