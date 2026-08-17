import importlib
import unittest

import numpy as np


parser = importlib.import_module("src.01_parser")
sw = importlib.import_module("src.05_sw_bted")
from src.node import CapstoneNode


class DummyNLP:
    """Minimal NLP double for parser tests that do not require spaCy models."""

    def __call__(self, text):
        return []


class ParserRegressionTests(unittest.TestCase):
    def test_clean_text_removes_bullet_noise_and_normalizes_space(self):
        self.assertEqual(parser.clean_text("  •  Uses\tAI  *  OCR  "), "Uses AI OCR")

    def test_group_header_detection_is_bounded(self):
        self.assertTrue(parser.is_group_header("Verification Officer:"))
        self.assertTrue(parser.is_group_header("1. Applicant"))
        self.assertFalse(parser.is_group_header("This is a long sentence with more than eight tokens here"))

    def test_initial_tree_has_four_domains_and_preserves_requirement_metadata(self):
        sections = {
            "english_title": "Resilience Housing",
            "vietnamese_title": "Nha o ben vung",
            "context": ["Housing assistance context"],
            "functional_requirement": ["Verification Officer:", "Uses AI to extract text"],
            "nonfunctional_requirement": ["Performance: Response time is low"],
            "proposed_solutions": [],
            "applied_theory": ["React frontend", "Agile methodology"],
            "products": [],
            "proposed_tasks": ["Implement OCR"],
        }
        root = parser.build_initial_tree("SU26SE082", sections, DummyNLP())
        self.assertEqual(root.depth, 1)
        self.assertEqual(root.label, "SU26SE082")
        self.assertEqual([node.label for node in root.children], [
            "D1_BUSINESS_CONTEXT", "D2_FUNCTIONAL",
            "D3_TECHNICAL_REALIZATION", "D4_EXECUTION_PLANNING",
        ])
        functional = root.children[1]
        requirement = functional.children[0]
        self.assertEqual(requirement.raw_text, "Uses AI to extract text")
        self.assertEqual(requirement.normalized_text, "Verification Officer Uses AI to extract text")
        self.assertEqual(requirement.feature_label, "Verification Officer")


class CostEngineRegressionTests(unittest.TestCase):
    def make_node(self, depth, label, schema_class=None, embedding=None):
        return CapstoneNode(
            label=label,
            schema_class=schema_class or label,
            depth=depth,
            embedding=embedding,
        )

    def test_beta_accepts_per_layer_dictionary(self):
        model = sw.SWCostModel(alpha=1.0, beta={"T2": 0.0, "T3": 0.9, "T4": 0.8})
        left = self.make_node(3, "actor", "IntentMatching", [1.0, 0.0])
        right = self.make_node(3, "actor", "IntentMatching", [0.0, 1.0])
        self.assertAlmostEqual(model.w_rep(left, right), 1.8, places=6)

    def test_replace_cost_is_bounded_by_delete_plus_insert(self):
        model = sw.SWCostModel(alpha=1.0, beta={"T2": 0.0, "T3": 0.9, "T4": 0.8})
        for depth in (2, 3, 4):
            left = self.make_node(depth, "left", "IntentMatching", [1.0, 0.0])
            right = self.make_node(depth, "right", "IntentMatching", [0.0, 1.0])
            self.assertLessEqual(
                model.w_rep(left, right),
                model.w_del(left) + model.w_ins(right) + 1e-9,
            )

    def test_canonical_reproduction_artifact_has_expected_metrics(self):
        import json
        from pathlib import Path

        path = Path(__file__).parents[1] / "reports" / "canonical_reproduction_138.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for source in ("historical_source", "current_source"):
            row = data[source]
            self.assertEqual(row["n_pairs"], 138)
            self.assertEqual(row["positive"], 38)
            self.assertAlmostEqual(row["cv_f1_mean"], 0.9498, places=4)
            self.assertAlmostEqual(row["cv_f1_std"], 0.0253, places=4)


if __name__ == "__main__":
    unittest.main()
