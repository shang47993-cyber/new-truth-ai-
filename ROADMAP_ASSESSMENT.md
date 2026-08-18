# Mysha AI Detection Engine — Project Assessment & Roadmap

## 1. Executive Summary & Assessment of Current Implementation (V1–V3)

Based on empirical evaluation across iterations (V1 through V3) and comparison with SOTA research benchmarks (SeqXGPT paradigm):

| Dimension | Current Baseline (V1–V3 Calibrated Stylometry + DistilGPT-2) | Strengths | Limitations / Ceiling | Target Architecture (SeqXGPT Paradigm) | Target Performance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AI vs. Human (Essay / Formal Text)** | Calibrated Logistic Regression (23 features) + DistilGPT-2 ONNX Perplexity | High precision on formal essays, news, academic, blogs; transparent signal breakdown | Relies on standard surface feature distributions | Multi-model token log-prob waveforms + Calibrated Neural Head | **>95% Macro-F1** |
| **AI vs. Human (Creative Writing & Dialogue)** | Stylometric + Perplexity | Explains statistical markers | **60%–75% (Lands in "Uncertain")**. Fails when LLMs are prompted for literary voice, short punchy dialogue, or casual cadence. | Token-level cross-model probability vectors (cross-model predictability curves) | **>92%+** across fiction & dialogue |
| **Authorship Detection (Number of Authors)** | Unsupervised K-Means ($k \in [2, 6]$) with Silhouette Scoring & Rolling 50-word Chunks | Good when two distinctly styled long texts are concatenated | **65%–70%** (Reliable only on long texts $> 500$ words; 50-word chunks suffer high noise) | Hybrid Stylometric + Semantic Embeddings (Sentence-Transformers) | Multi-author cluster voting with semantic normalization |
| **Style-Shift Boundary Detection** | Euclidean distance outlier detection ($> 2.5\sigma$) | Identifies abrupt register shifts (formal $\leftrightarrow$ casual) | Conflates topical/semantic shifts with genuine stylistic shifts | Sentence-level 1D CNN + CRF boundary sequence tagger | Precise sentence-by-sentence boundary detection |

---

## 2. Why Mathematical Stylometry Alone Cannot Reach SOTA (The Ceiling)

Current statistical stylometry evaluates surface features:
- **Lexical richness:** TTR, Yule’s K, Simpson’s Index
- **Sentence length variation:** $\text{CV} = \sigma / \mu$
- **Surface markers:** Contraction ratios, punctuation frequencies, Zipf’s law fit

### The Fundamental Vulnerability
Modern LLMs (Claude 3.5 Sonnet, GPT-4o, etc.) adapt surface features dynamically when prompted for voice, fiction, or personality:
1. Average sentence length drops to $\approx 9$ words.
2. Contractions appear naturally.
3. Sentence variation matches human writing distributions.

Consequently, detectors relying solely on surface stylometry misclassify human literary prose as AI or AI-crafted fiction as human.

---

## 3. Required Architecture: The SeqXGPT Multi-Model Waveform Paradigm

```
[ Input Text ]
      │
      ├──► 1. Multi-LLM Perplexity Extraction (GPT-2 + TinyLlama / GPT-Neo via ONNX)
      │       └── Extracts token log-probability streams across models [M models × N tokens]
      │
      ├──► 2. Stylometric & Morphological Embeddings (23 dimensions)
      │
      └──► 3. Hybrid Neural Classifier (1D CNN + Transformer + CRF)
              ├── Sentence-level boundary prediction (Classifies sentence-by-sentence)
              └── Document-level calibrated probability + Multi-author cluster voting
```

### Key Components to Implement
1. **Multi-Model Perplexity Vectors:** Query compact models in parallel via ONNX to extract token log-probability signals as waveforms. AI-generated text exhibits a distinct cross-model predictability curve absent in human writing.
2. **Sentence-Level Token Alignment:** Evaluate token-by-token across sentences rather than document averages to identify hybrid/collaborative documents (e.g., human-drafted, AI-polished).
3. **Decoupled Semantic vs. Stylistic Authorship:** Combine stylometric vectors with semantic embeddings (Sentence-Transformers) to ensure topic changes do not trigger false authorship boundaries.

---

## 4. Local Codebase & Implementation Mapping

- **Codebase Root:** `/Users/sankalp/workspace/mysha`
- **Core Files:**
  - `backend/analysis/detector.py` (Main pipeline entrypoint)
  - `backend/analysis/neural_waveform.py` (Perplexity / waveform extraction)
  - `backend/analysis/stylometry.py` (23-feature extractor)
  - `backend/analysis/authorship.py` (Clustering & boundary detection)
  - `backend/analysis/transformer_detector.py` (ONNX model wrapper)
  - `backend/analysis/train_classifier.py` (Classifier training & calibration)
  - `app.py` (Flask REST API backend)
  - `frontend/` (Templates & UI)
