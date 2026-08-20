"""test_pipeline.py — End-to-end regression & accuracy test suite for Mysha Detection Pipeline."""

import sys
import unittest
import numpy as np

from backend.analysis.stylometry import extract_features
from backend.analysis.neural_waveform import compute_token_waveform, get_neural_ai_score
from backend.analysis.transformer_detector import detect_with_transformer
from backend.analysis.detector import detect_ai_vs_human
from backend.analysis.authorship import detect_authors

class TestMyshaPipeline(unittest.TestCase):

    def test_stylometry_feature_extraction(self):
        text = "This is a clean sample sentence designed for testing lexical richness and syntactic variance."
        feats = extract_features(text)
        self.assertNotIn("error", feats)
        self.assertIn("type_token_ratio", feats)
        self.assertIn("zipf_r_squared", feats)
        self.assertIn("yules_k", feats)
        self.assertGreater(feats["type_token_ratio"], 0.0)

    def test_neural_waveform_and_gltr(self):
        text = "Artificial intelligence and machine learning architectures operate on gradient descent optimization."
        wave = compute_token_waveform(text)
        self.assertNotIn("error", wave)
        self.assertIn("gltr_top10", wave)
        self.assertIn("gltr_tail1000", wave)
        self.assertIn("perplexity", wave)
        self.assertGreaterEqual(wave["gltr_top10"], 0.0)
        self.assertLessEqual(wave["gltr_top10"], 1.0)
        
        score_data = get_neural_ai_score(wave)
        self.assertIn("neural_ai_bias", score_data)

    def test_transformer_head(self):
        text = "Deep neural networks learn hierarchical representations directly from raw text inputs."
        tr = detect_with_transformer(text)
        self.assertIn("transformer_ai_prob", tr)
        self.assertIn("transformer_human_prob", tr)
        self.assertAlmostEqual(tr["transformer_ai_prob"] + tr["transformer_human_prob"], 1.0, places=3)

    def test_ai_detection_formal_ai(self):
        ai_text = (
            "The rapid advancement of artificial intelligence has fundamentally transformed how we approach "
            "problem-solving across industries. Machine learning algorithms now process vast datasets with "
            "unprecedented efficiency, enabling organizations to extract meaningful insights from complex "
            "information streams. This technological evolution represents a paradigm shift in computational capabilities."
        )
        res = detect_ai_vs_human(ai_text)
        self.assertGreaterEqual(res["ai_probability"], 80.0)
        self.assertEqual(res["confidence"], "high")
        self.assertIn("High probability of AI Generation", res["verdict"])
        self.assertIsNotNone(res["evidence"])

    def test_ai_detection_human_narrative(self):
        human_text = (
            "The funeral was on a Wednesday, which my grandmother would have hated. She always said "
            "Wednesdays were for laundry and nothing else. We buried her in the blue dress she wore to "
            "my cousin's wedding. My aunt chose it. My mother disagreed but said nothing, which is how "
            "my mother disagrees with everything — silently, completely, and forever."
        )
        res = detect_ai_vs_human(human_text)
        self.assertGreaterEqual(res["human_probability"], 80.0)
        self.assertEqual(res["confidence"], "high")
        self.assertIn("Strong signature of Human Authorship", res["verdict"])

    def test_authorship_clustering(self):
        single_author_text = (
            "Paragraph one is written in a simple and direct manner without complicated words.\n\n"
            "Paragraph two maintains the exact same tone and vocabulary without any shift.\n\n"
            "Paragraph three concludes with the exact same sentence length and simple style.\n\n"
            "Paragraph four wraps up the short document consistently."
        )
        auth = detect_authors(single_author_text)
        self.assertIn("predicted_authors", auth)
        self.assertGreaterEqual(auth["predicted_authors"], 1)

if __name__ == "__main__":
    unittest.main()
