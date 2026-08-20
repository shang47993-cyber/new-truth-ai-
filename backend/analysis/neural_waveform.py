"""neural_waveform.py — SeqXGPT & GLTR Neural Perplexity Waveform Engine."""

import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

def _load_neural_engine():
    global _model, _tokenizer
    if _model is not None:
        return
    
    logger.info("Loading DistilGPT-2 Neural Waveform Engine...")
    model_name = "distilgpt2"
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(model_name)
    _model.eval()
    logger.info("Neural Waveform Engine loaded successfully.")

def compute_token_waveform(text: str, max_tokens: int = 128) -> dict:
    """Computes token-level log probability wave, GLTR rank distributions, and volatility metrics."""
    _load_neural_engine()

    if not text or len(text.strip()) < 10:
        return {
            "perplexity": 0.0,
            "mean_log_prob": 0.0,
            "std_log_prob": 0.0,
            "frac_low_entropy": 0.0,
            "frac_high_entropy": 0.0,
            "wave_volatility": 0.0,
            "gltr_top10": 0.0,
            "gltr_top100": 0.0,
            "gltr_tail1000": 0.0,
            "token_waveform": [],
            "n_tokens": 0,
            "error": "Text too short for neural waveform analysis"
        }

    inputs = _tokenizer(text[:1200], return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    n_tokens = input_ids.size(1)
    if n_tokens < 4:
        return {
            "perplexity": 0.0,
            "mean_log_prob": 0.0,
            "std_log_prob": 0.0,
            "frac_low_entropy": 0.0,
            "frac_high_entropy": 0.0,
            "wave_volatility": 0.0,
            "gltr_top10": 0.0,
            "gltr_top100": 0.0,
            "gltr_tail1000": 0.0,
            "token_waveform": [],
            "n_tokens": n_tokens,
            "error": "Insufficient token count"
        }

    with torch.no_grad():
        outputs = _model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # [1, seq_len-1, vocab_size]
        targets = input_ids[:, 1:]          # [1, seq_len-1]

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        waveform_values = target_log_probs[0].cpu().numpy().tolist()

        # Compute GLTR token ranks (rank of target token in predicted distribution)
        # Sort descending by logit to find rank
        sorted_indices = torch.argsort(logits[0], dim=-1, descending=True)
        # Find where the target matches sorted_indices
        target_expanded = targets[0].unsqueeze(-1)
        ranks = (sorted_indices == target_expanded).nonzero(as_tuple=False)[:, 1].cpu().numpy()

    wave_arr = np.array(waveform_values)
    mean_lp = float(np.mean(wave_arr))
    std_lp = float(np.std(wave_arr))
    ppl = float(np.exp(-mean_lp))

    # Classical entropy ratios
    frac_low_entropy = float(np.mean(wave_arr > -1.2))
    frac_high_entropy = float(np.mean(wave_arr < -4.5))
    wave_diffs = np.diff(wave_arr) if len(wave_arr) > 1 else np.array([0.0])
    wave_volatility = float(np.std(wave_diffs))

    # GLTR Distribution Bins:
    # Top 10 (Green: highly predictable AI tokens)
    # Top 100 (Yellow: moderately predictable)
    # Tail > 1000 (Red/Purple: unexpected human tokens)
    gltr_top10 = float(np.mean(ranks < 10))
    gltr_top100 = float(np.mean(ranks < 100))
    gltr_tail1000 = float(np.mean(ranks >= 1000))

    tokens = [_tokenizer.decode([token_id]) for token_id in input_ids[0][1:].tolist()]
    token_details = [
        {
            "token": t,
            "log_prob": round(float(lp), 3),
            "prob": round(float(np.exp(lp)), 4),
            "rank": int(r)
        }
        for t, lp, r in zip(tokens, waveform_values, ranks)
    ]

    return {
        "perplexity": round(ppl, 2),
        "mean_log_prob": round(mean_lp, 4),
        "std_log_prob": round(std_lp, 4),
        "frac_low_entropy": round(frac_low_entropy, 4),
        "frac_high_entropy": round(frac_high_entropy, 4),
        "wave_volatility": round(wave_volatility, 4),
        "gltr_top10": round(gltr_top10, 4),
        "gltr_top100": round(gltr_top100, 4),
        "gltr_tail1000": round(gltr_tail1000, 4),
        "token_waveform": token_details[:60],
        "n_tokens": len(waveform_values)
    }

def get_neural_ai_score(wave_metrics: dict) -> dict:
    """Interprets neural waveform & GLTR rank characteristics to compute neural AI likelihood contribution."""
    if "error" in wave_metrics or wave_metrics.get("n_tokens", 0) < 4:
        return {"neural_ai_bias": 0.0, "signals": []}

    ppl = wave_metrics["perplexity"]
    frac_low = wave_metrics["frac_low_entropy"]
    volatility = wave_metrics["wave_volatility"]
    gltr_top10 = wave_metrics.get("gltr_top10", 0.0)
    gltr_tail = wave_metrics.get("gltr_tail1000", 0.0)

    signals = []
    neural_bias = 0.0

    # Decisive Perplexity thresholds
    if ppl < 25.0:
        neural_bias += 1.4
        signals.append({"signal": "Ultra-Low Perplexity", "desc": f"Tokens match canonical LLM generation (PPL={ppl:.1f})", "dir": "AI"})
    elif ppl < 40.0:
        neural_bias += 0.85
        signals.append({"signal": "Low Perplexity", "desc": f"Language conforms to standard model paths (PPL={ppl:.1f})", "dir": "AI"})
    elif ppl > 100.0:
        neural_bias -= 1.3
        signals.append({"signal": "High Perplexity", "desc": f"Distinct human stylistic surprise & rich phrasing (PPL={ppl:.1f})", "dir": "Human"})
    elif ppl > 65.0:
        neural_bias -= 0.75
        signals.append({"signal": "Elevated Perplexity", "desc": f"Vocabulary and phrasing diverge from model predictions (PPL={ppl:.1f})", "dir": "Human"})

    # GLTR Rank Bins (GLTR Top-10 vs Tail)
    if gltr_top10 > 0.65:
        neural_bias += 0.7
        signals.append({"signal": "GLTR High Top-10 Density", "desc": f"{gltr_top10:.0%} of tokens are in the model's top 10 choices", "dir": "AI"})
    elif gltr_top10 < 0.40:
        neural_bias -= 0.6
        signals.append({"signal": "GLTR Low Top-10 Density", "desc": f"Only {gltr_top10:.0%} of tokens in model top 10 (rich human variance)", "dir": "Human"})

    if gltr_tail > 0.08:
        neural_bias -= 0.8
        signals.append({"signal": "GLTR Heavy Tail Spikes", "desc": f"{gltr_tail:.0%} of tokens fall outside top 1000 (human idiosyncrasy)", "dir": "Human"})

    # Waveform volatility
    if volatility < 1.6:
        neural_bias += 0.4
        signals.append({"signal": "Uniform Token Waveform", "desc": "Smooth, artificial token transitions across sequence", "dir": "AI"})
    elif volatility > 2.6:
        neural_bias -= 0.4
        signals.append({"signal": "Volatile Token Waveform", "desc": "Human burstiness and irregular rhythm spikes", "dir": "Human"})

    return {
        "neural_ai_bias": round(float(neural_bias), 3),
        "signals": signals
    }
