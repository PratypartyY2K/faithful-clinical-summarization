from __future__ import annotations

import unittest
from unittest.mock import patch

from src.preprocessing.claim_extractor import extract_claims


class ClaimExtractorBackendTest(unittest.TestCase):
    def test_extract_claims_uses_heuristic_backend_by_default(self) -> None:
        claims = extract_claims("Patient seen for hypertension with report of morning headaches.")
        self.assertGreaterEqual(len(claims), 2)

    def test_extract_claims_dispatches_to_llm_backend(self) -> None:
        with patch(
            "src.preprocessing.llm_claim_extractor.extract_claims_with_openai",
            return_value=["The patient has hypertension."],
        ) as mocked:
            claims = extract_claims(
                "Patient seen for hypertension.",
                backend="llm",
                llm_model="gpt-4.1-mini",
            )
        mocked.assert_called_once()
        self.assertEqual(claims, ["The patient has hypertension."])


if __name__ == "__main__":
    unittest.main()
