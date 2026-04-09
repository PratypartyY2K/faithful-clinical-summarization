from __future__ import annotations

import unittest

from scripts.create_dummy_dataset import PATIENT_PROFILES, build_claims


class DummyDatasetTest(unittest.TestCase):
    def test_nli_claim_schema_contains_three_labels(self) -> None:
        claims = build_claims(PATIENT_PROFILES[0], label_schema="nli")
        label_names = {claim["label_name"] for claim in claims}
        self.assertEqual(label_names, {"contradiction", "neutral", "entailment"})


if __name__ == "__main__":
    unittest.main()
