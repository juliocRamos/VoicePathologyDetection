import unittest

import pandas as pd

from classes.experiment.runners.cross_database_experiment_runner import (
    CrossDatabaseExperimentRunner,
)
from main import (
    build_hupa_manifest_config,
    build_cross_hupa_manifest_config,
    build_cross_svd_manifest_config,
    build_svd_manifest_config,
)


class CrossDatabaseCohortTests(unittest.TestCase):
    def test_both_cross_database_cohorts_are_adult_only(self) -> None:
        hupa_config = build_cross_hupa_manifest_config()
        svd_config = build_cross_svd_manifest_config()

        self.assertTrue(hupa_config.adults_only)
        self.assertTrue(svd_config.adults_only)
        self.assertEqual(hupa_config.minimum_age, 18.0)
        self.assertEqual(svd_config.minimum_age, 18.0)
        self.assertEqual(svd_config.vowels, ("a",))
        self.assertEqual(svd_config.conditions, ("n",))

    def test_independent_cohorts_match_cross_protocol(self) -> None:
        hupa_config = build_hupa_manifest_config()
        svd_config = build_svd_manifest_config()

        self.assertTrue(hupa_config.adults_only)
        self.assertEqual(hupa_config.minimum_age, 18.0)
        self.assertTrue(svd_config.adults_only)
        self.assertEqual(svd_config.minimum_age, 18.0)
        self.assertEqual(svd_config.vowels, ("a",))
        self.assertEqual(svd_config.conditions, ("n",))

    def test_cross_database_cohort_rejects_invalid_ages(self) -> None:
        cohort = pd.DataFrame({
            "age": [17.0, None, 40.0],
            "label": ["healthy", "pathological", "healthy"],
        })

        with self.assertRaisesRegex(
            ValueError,
            "without valid age",
        ):
            CrossDatabaseExperimentRunner._validate_adult_cohort(
                cohort=cohort,
                database="TEST",
            )

        with self.assertRaisesRegex(
            ValueError,
            "younger than 18",
        ):
            CrossDatabaseExperimentRunner._validate_adult_cohort(
                cohort=cohort.dropna(subset=["age"]),
                database="TEST",
            )

    def test_demographic_summary_is_stratified_and_auditable(
        self,
    ) -> None:
        cohort = pd.DataFrame({
            "speaker_id": ["s1", "s2", "s3"],
            "age": [20.0, 30.0, 40.0],
            "sex": ["female", "male", None],
            "label": ["healthy", "healthy", "pathological"],
        })

        summary = (
            CrossDatabaseExperimentRunner._demographic_summary_row(
                cohort=cohort,
                database="TEST",
                label="all",
            )
        )

        self.assertEqual(summary["n_samples"], 3)
        self.assertEqual(summary["n_speakers"], 3)
        self.assertEqual(summary["age_median"], 30.0)
        self.assertEqual(summary["female_samples"], 1)
        self.assertEqual(summary["male_samples"], 1)
        self.assertEqual(summary["sex_missing_or_other"], 1)


if __name__ == "__main__":
    unittest.main()
