from __future__ import annotations

import unittest

from src.preprocessing.mimiciii_discharge import build_raw_example, parse_note_sections


class MIMICIIIIngestionTest(unittest.TestCase):
    def test_parse_note_sections_handles_inline_and_block_headings(self) -> None:
        text = """
        Admission Date: 2118-06-02
        Discharge Date: 2118-06-14

        HISTORY OF PRESENT ILLNESS:
        Progressive dyspnea for three days.

        PAST MEDICAL HISTORY:
        COPD.

        Brief Hospital Course: Required intubation and ICU stay.

        DISCHARGE MEDICATIONS:
        Albuterol inhaler.
        """

        sections = parse_note_sections(text)

        self.assertEqual(sections["history of present illness"], "Progressive dyspnea for three days.")
        self.assertEqual(sections["past medical history"], "COPD.")
        self.assertEqual(sections["brief hospital course"], "Required intubation and ICU stay.")
        self.assertEqual(sections["discharge medications"], "Albuterol inhaler.")

    def test_build_raw_example_removes_deidentification_and_skips_medication_targets(self) -> None:
        row = {
            "SUBJECT_ID": "10",
            "HADM_ID": "20",
            "CHARTDATE": "2118-06-14",
            "CATEGORY": "Discharge summary",
            "DESCRIPTION": "Report",
            "TEXT": """
            HISTORY OF PRESENT ILLNESS:
            Patient presented to [**Hospital1 18**] with progressive dyspnea and worsening cough.
            She had severe wheezing and increasing oxygen requirements prior to admission.

            BRIEF HOSPITAL COURSE:
            Treated with steroids, bronchodilators, and oxygen with gradual improvement over several days.
            Respiratory status stabilized and the patient was discharged after successful weaning from supplemental support.

            DISCHARGE MEDICATIONS:
            1. Albuterol inhaler.
            2. Prednisone taper.

            FOLLOW-UP PLANS:
            Follow up with pulmonology in one week.

            Dictated By: Someone
            JOB#: 1234
            """,
        }

        example = build_raw_example(row, min_source_chars=20, min_target_chars=20)

        self.assertIsNotNone(example)
        assert example is not None
        self.assertNotIn("[**", str(example["dialogue"]))
        self.assertNotIn("[**", str(example["summary"]))
        self.assertNotIn("dictated by", str(example["summary"]).lower())
        self.assertEqual(example["metadata"]["target_sections"], ["brief hospital course"])
        self.assertNotIn("albuterol", str(example["summary"]).lower())
        self.assertNotIn("follow up with pulmonology", str(example["summary"]).lower())

    def test_build_raw_example_rejects_terse_status_only_targets(self) -> None:
        row = {
            "SUBJECT_ID": "11",
            "HADM_ID": "21",
            "CHARTDATE": "2118-06-14",
            "CATEGORY": "Discharge summary",
            "DESCRIPTION": "Report",
            "TEXT": """
            HISTORY OF PRESENT ILLNESS:
            Patient presented with chest pain and dyspnea requiring hospital admission for evaluation.
            The symptoms were severe enough to require urgent inpatient monitoring and treatment.

            DISCHARGE CONDITION:
            Good

            DISCHARGE DISPOSITION:
            Home
            """,
        }

        example = build_raw_example(row, min_source_chars=20, min_target_chars=20)

        self.assertIsNone(example)

    def test_build_raw_example_extracts_source_and_target_sections(self) -> None:
        row = {
            "SUBJECT_ID": "13702",
            "HADM_ID": "107527",
            "CHARTDATE": "2118-06-14",
            "CATEGORY": "Discharge summary",
            "DESCRIPTION": "Report",
            "TEXT": """
            HISTORY OF PRESENT ILLNESS:
            This is an 81-year-old female with progressive dyspnea over three days.
            She failed outpatient therapy and presented to the emergency department.

            PAST MEDICAL HISTORY:
            COPD, angina, hypothyroidism, depression.

            PHYSICAL EXAM:
            Tachycardic with diffuse wheezing and increased work of breathing.

            BRIEF HOSPITAL COURSE:
            Treated for COPD exacerbation with steroids, bronchodilators, antibiotics,
            and later intubation for respiratory failure before eventual extubation.
            She improved steadily after ICU management and was discharged once breathing returned closer to baseline.

            DISCHARGE MEDICATIONS:
            Levothyroxine, citalopram, aspirin, inhalers, and levofloxacin.

            FOLLOW-UP PLANS:
            Follow up with pulmonology and primary care within two weeks.
            """,
        }

        example = build_raw_example(row, min_source_chars=50, min_target_chars=50)

        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example["example_id"], "mimiciii-107527")
        self.assertEqual(example["claims"], [])
        self.assertIn("progressive dyspnea", str(example["dialogue"]).lower())
        self.assertIn("treated for copd exacerbation", str(example["summary"]).lower())
        self.assertEqual(example["metadata"]["source_sections"][0], "history of present illness")
        self.assertNotIn("past medical history", example["metadata"]["source_sections"])
        self.assertEqual(example["metadata"]["target_sections"], ["brief hospital course"])
        self.assertNotIn("levofloxacin", str(example["summary"]).lower())


if __name__ == "__main__":
    unittest.main()
