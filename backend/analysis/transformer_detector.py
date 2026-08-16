"""transformer_detector.py — Pre-trained Supervised Deep Transformer AI Text Detection Head.

Uses fine-tuned deep bidirectional representations (BERT / RoBERTa) to detect subtle
machine-generated semantic artifacts, syntactic regularities, and latent model signatures.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

_bert_model = None
_bert_tokenizer = None

def _load_transformer_head():
    global _bert_model, _bert_tokenizer
    if _bert_model is not None:
        return

    model_id = "followsci/bert-ai-text-detector"
    logger.info(f"Loading Deep Transformer Classifier Head: {model_id}...")
    _bert_tokenizer = AutoTokenizer.from_pretrained(model_id)
    _bert_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    _bert_model.eval()
    logger.info("Deep Transformer Classifier Head ready.")

def detect_with_transformer(text: str) -> dict:
    """Classifies text using the fine-tuned deep transformer detector."""
    _load_transformer_head()
    
    if not text or len(text.strip()) < 10:
        return {"transformer_ai_prob": 0.5, "transformer_human_prob": 0.5, "confidence": "low"}

    inputs = _bert_tokenizer(text[:2000], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

    # Model id2label: {0: Human, 1: AI}
    human_p = float(probs[0])
    ai_p = float(probs[1])

    return {
        "transformer_human_prob": round(human_p, 4),
        "transformer_ai_prob": round(ai_p, 4),
        "ai_pct": round(ai_p * 100, 1),
        "human_pct": round(human_p * 100, 1)
    }
