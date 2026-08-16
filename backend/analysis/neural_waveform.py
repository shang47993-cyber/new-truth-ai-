"""neural_waveform.py — SeqXGPT-inspired Token-level Neural Perplexity Waveform Engine."""

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
    """Computes token-level log probability wave and metrics."""
    _load_neural_engine()

    if not text or len(text.strip()) < 10:
        return {
            "perplexity": 0.0,
            "mean_log_prob": 0.0,
            "std_log_prob": 0.0,
            "frac_low_entropy": 0.0,
            "frac_high_entropy": 0.0,
            "wave_volatility": 0.0,
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
            "token_waveform": [],
            "n_tokens": n_tokens,
            "error": "Insufficient token count"
        }

    with torch.no_grad():
        outputs = _model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        waveform_values = target_log_probs[0].cpu().numpy().tolist()

    wave_arr = np.array(waveform_values)
    mean_lp = float(np.mean(wave_arr))
    std_lp = float(np.std(wave_arr))
    ppl = float(np.exp(-mean_lp))

    frac_low_entropy = float(np.mean(wave_arr > -1.2))
    frac_high_entropy = float(np.mean(wave_arr < -4.5))
    wave_diffs = np.diff(wave_arr) if len(wave_arr) > 1 else np.array([0.0])
    wave_volatility = float(np.std(wave_diffs))

    tokens = [_tokenizer.decode([token_id]) for token_id in input_ids[0][1:].tolist()]
    token_details = [
        {"token": t, "log_prob": round(float(lp), 3), "prob": round(float(np.exp(lp)), 4)}
        for t, lp in zip(tokens, waveform_values)
    ]

    return {
        "perplexity": round(ppl, 2),
        "mean_log_prob": round(mean_lp, 4),
        "std_log_prob": round(std_lp, 4),
        "frac_low_entropy": round(frac_low_entropy, 4),
        "frac_high_entropy": round(frac_high_entropy, 4),
        "wave_volatility": round(wave_volatility, 4),
        "token_waveform": token_details[:60],
        "n_tokens": len(waveform_values)
    }

def get_neural_ai_score(wave_metrics: dict) -> dict:
    """Interprets neural waveform characteristics to compute neural AI likelihood contribution."""
    if "error" in wave_metrics or wave_metrics.get("n_tokens", 0) < 4:
        return {"neural_ai_bias": 0.0, "signals": []}

    ppl = wave_metrics["perplexity"]
    frac_low = wave_metrics["frac_low_entropy"]
    volatility = wave_metrics["wave_volatility"]

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

    # High predictability density
    if frac_low > 0.40:
        neural_bias += 0.6
        signals.append({"signal": "High Predictability Density", "desc": f"{frac_low:.0%} of tokens match greedy model paths", "dir": "AI"})
    elif frac_low < 0.20:
        neural_bias -= 0.5
        signals.append({"signal": "Low Predictability Density", "desc": f"Only {frac_low:.0%} of tokens follow typical LLM paths", "dir": "Human"})

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
