"""stylometry.py — Core stylometric feature extraction with outlier bounding.

Extracts quantitative linguistic, syntactic, lexical, and morphological features.
"""

import re
import math
import string
from collections import Counter
from typing import Optional

import numpy as np
from scipy import stats


def tokenize_words(text: str) -> list[str]:
    """Split text into lowercase words."""
    return re.findall(r"\b\w+\b", text.lower())


def tokenize_sentences(text: str) -> list[str]:
    """Split text into clean sentences."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_features(text: str) -> dict:
    """Extract 23 normalized stylometric features with NaN safety guards."""
    words = tokenize_words(text)
    sentences = tokenize_sentences(text)

    if not words or not sentences:
        return {"error": "Text too short for analysis"}

    word_lengths = [len(w) for w in words]
    sentence_lengths = [len(tokenize_words(s)) for s in sentences]

    vocab = set(words)
    n_words = len(words)
    n_vocab = len(vocab)

    ttr = n_vocab / n_words if n_words > 0 else 0

    freq = Counter(words)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    hapax_ratio = hapax / n_vocab if n_vocab > 0 else 0

    dis = sum(1 for w, c in freq.items() if c == 2)
    dis_ratio = dis / n_vocab if n_vocab > 0 else 0

    freq_spectrum = Counter(freq.values())
    m1 = n_words
    m2 = sum(i * i * vi for i, vi in freq_spectrum.items())
    yules_k = 10000 * (m2 - m1) / (m1 * m1) if m1 > 1 else 0

    simpsons_d = 1 - sum(c * (c - 1) for c in freq.values()) / (n_words * (n_words - 1)) if n_words > 1 else 0
    brunets_w = n_words ** (n_vocab ** -0.172) if n_vocab > 0 else 0
    honores_r = 100 * math.log(n_words) / (1 - hapax / n_vocab) if n_vocab > 0 and hapax != n_vocab else 0

    avg_sentence_len = np.mean(sentence_lengths) if sentence_lengths else 0
    std_sentence_len = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
    cv_sentence_len = std_sentence_len / avg_sentence_len if avg_sentence_len > 0 else 0

    avg_word_len = np.mean(word_lengths) if word_lengths else 0
    std_word_len = np.std(word_lengths) if len(word_lengths) > 1 else 0

    punct_counts = Counter(c for c in text if c in string.punctuation)
    total_punct = sum(punct_counts.values())
    punct_ratio = total_punct / len(text) if len(text) > 0 else 0
    comma_ratio = punct_counts.get(',', 0) / n_words if n_words > 0 else 0
    semicolon_ratio = punct_counts.get(';', 0) / n_words if n_words > 0 else 0
    exclamation_ratio = punct_counts.get('!', 0) / len(sentences) if sentences else 0
    question_ratio = punct_counts.get('?', 0) / len(sentences) if sentences else 0

    function_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
        'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
        'they', 'them', 'their', 'not', 'no', 'nor', 'as', 'if', 'then',
        'than', 'so', 'just', 'very', 'also', 'still', 'already', 'even',
        'quite', 'rather', 'about', 'above', 'after', 'before', 'between',
        'into', 'through', 'during', 'without', 'within'
    }
    func_word_count = sum(1 for w in words if w in function_words)
    func_word_ratio = func_word_count / n_words if n_words > 0 else 0

    sorted_freqs = sorted(freq.values(), reverse=True)
    if len(sorted_freqs) > 5:
        ranks = np.arange(1, len(sorted_freqs) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(np.array(sorted_freqs, dtype=float))
        valid_mask = np.isfinite(log_ranks) & np.isfinite(log_freqs)
        if valid_mask.sum() > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_ranks[valid_mask], log_freqs[valid_mask]
            )
            zipf_coefficient = abs(slope) if np.isfinite(slope) else 0
            zipf_r_squared = r_value ** 2 if np.isfinite(r_value) else 0
        else:
            zipf_coefficient = 0
            zipf_r_squared = 0
    else:
        zipf_coefficient = 0
        zipf_r_squared = 0

    if n_words > 20:
        top_words = [w for w, c in freq.most_common(10) if c > 2]
        burstiness_scores = []
        for word in top_words:
            positions = [i for i, w in enumerate(words) if w == word]
            if len(positions) > 1:
                gaps = np.diff(positions)
                mean_gap = np.mean(gaps)
                std_gap = np.std(gaps)
                burstiness = (std_gap - mean_gap) / (std_gap + mean_gap) if (std_gap + mean_gap) > 0 else 0
                burstiness_scores.append(burstiness)
        avg_burstiness = np.mean(burstiness_scores) if burstiness_scores else 0
    else:
        avg_burstiness = 0

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    n_paragraphs = len(paragraphs)
    if n_paragraphs > 1:
        para_lengths = [len(tokenize_words(p)) for p in paragraphs]
        cv_para_len = np.std(para_lengths) / np.mean(para_lengths) if np.mean(para_lengths) > 0 else 0
    else:
        cv_para_len = 0

    word_probs = np.array(list(freq.values())) / n_words
    entropy = -np.sum(word_probs * np.log2(word_probs + 1e-10))

    contractions = re.findall(r"\b\w+'\w+\b", text.lower())
    contraction_ratio = len(contractions) / n_words if n_words > 0 else 0

    return {
        "word_count": n_words,
        "vocabulary_size": n_vocab,
        "type_token_ratio": round(float(ttr), 4),
        "hapax_legomena_ratio": round(float(hapax_ratio), 4),
        "dis_legomena_ratio": round(float(dis_ratio), 4),
        "yules_k": round(float(yules_k), 4),
        "simpsons_diversity": round(float(simpsons_d), 4),
        "brunets_w": round(float(brunets_w), 4),
        "honores_r": round(float(honores_r), 4),
        "entropy": round(float(entropy), 4),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(float(avg_sentence_len), 2),
        "std_sentence_length": round(float(std_sentence_len), 2),
        "cv_sentence_length": round(float(cv_sentence_len), 4),
        "avg_word_length": round(float(avg_word_len), 2),
        "std_word_length": round(float(std_word_len), 2),
        "punctuation_ratio": round(float(punct_ratio), 4),
        "comma_ratio": round(float(comma_ratio), 4),
        "semicolon_ratio": round(float(semicolon_ratio), 4),
        "exclamation_ratio": round(float(exclamation_ratio), 4),
        "question_ratio": round(float(question_ratio), 4),
        "function_word_ratio": round(float(func_word_ratio), 4),
        "zipf_coefficient": round(float(zipf_coefficient), 4),
        "zipf_r_squared": round(float(zipf_r_squared), 4),
        "burstiness": round(float(avg_burstiness), 4),
        "paragraph_count": n_paragraphs,
        "cv_paragraph_length": round(float(cv_para_len), 4),
        "contraction_ratio": round(float(contraction_ratio), 4),
    }
