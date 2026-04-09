from __future__ import annotations

import unittest

from src.preprocessing.claim_extractor import split_into_claims


class ClaimExtractorTest(unittest.TestCase):
    def test_extracts_atomic_claims_from_summary_sections(self) -> None:
        summary = (
            "Patient seen for hypertension with report of morning headaches. "
            "Relevant finding: blood pressure 152/94. "
            "Plan is to continue lisinopril at 10 mg daily and repeat blood pressure log in 2 weeks."
        )

        claims = split_into_claims(summary)

        self.assertIn("Patient seen for hypertension.", claims)
        self.assertIn("Reported morning headaches.", claims)
        self.assertIn("Relevant finding: blood pressure 152/94.", claims)
        self.assertIn("Continue lisinopril at 10 mg daily.", claims)
        self.assertIn("Repeat blood pressure log in 2 weeks.", claims)


if __name__ == "__main__":
    unittest.main()
