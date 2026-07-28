from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from classes.dataset.preparation.hupa_training_manifest_builder import (
    HUPATrainingManifestBuilder,
    HUPATrainingManifestConfig,
)
from classes.dataset.preparation.svd_training_manifest_builder import (
    SVDTrainingManifestBuilder,
    SVDTrainingManifestConfig,
)


class TrainingManifestBuilderTests(unittest.TestCase):
    def test_svd_builder_filters_and_consolidates_diagnostic_copies(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate_a = self._write(root / "p1.nsp", b"same-audio")
            duplicate_b = self._write(root / "p2.nsp", b"same-audio")
            healthy = self._write(root / "healthy.nsp", b"healthy")
            child = self._write(root / "child.nsp", b"child")
            short = self._write(root / "short.nsp", b"short")
            wrong_vowel = self._write(root / "wrong.nsp", b"wrong")

            rows = [
                self._svd_row(
                    "p1",
                    duplicate_a,
                    "101",
                    "speaker-p",
                    "pathological",
                    "Group A",
                    age=40,
                ),
                self._svd_row(
                    "p2",
                    duplicate_b,
                    "101",
                    "speaker-p",
                    "pathological",
                    "Group B",
                    age=40,
                ),
                self._svd_row(
                    "h1",
                    healthy,
                    "202",
                    "speaker-h",
                    "healthy",
                    "healthy",
                    age=35,
                ),
                self._svd_row(
                    "child",
                    child,
                    "303",
                    "speaker-child",
                    "pathological",
                    "Group C",
                    age=17,
                ),
                self._svd_row(
                    "short",
                    short,
                    "404",
                    "speaker-short",
                    "pathological",
                    "Group D",
                    age=30,
                    duration=0.3,
                ),
                self._svd_row(
                    "wrong",
                    wrong_vowel,
                    "505",
                    "speaker-wrong",
                    "pathological",
                    "Group E",
                    age=30,
                    vowel="i",
                ),
                self._svd_row(
                    "broken",
                    root / "missing.nsp",
                    "606",
                    "speaker-broken",
                    "pathological",
                    "Group F",
                    age=30,
                    audio_status="error",
                ),
            ]

            result = SVDTrainingManifestBuilder(
                SVDTrainingManifestConfig()
            ).build(pd.DataFrame(rows))

            self.assertEqual(len(result.training_manifest), 2)
            self.assertEqual(len(result.duplicate_groups), 2)
            self.assertFalse(
                result.training_manifest["file_sha256"]
                .duplicated()
                .any()
            )
            self.assertEqual(
                set(result.excluded_samples["exclusion_reason"]),
                {
                    "audio_read_error",
                    "vowel_not_selected",
                    "age_below_minimum",
                    "duration_below_minimum",
                },
            )

            pathological = result.training_manifest.loc[
                result.training_manifest["label"].eq("pathological")
            ].iloc[0]
            self.assertEqual(
                pathological["pathology_group"],
                "Group A | Group B",
            )
            self.assertEqual(pathological["source_count"], 2)

    def test_svd_builder_rejects_inconsistent_acoustic_copies(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write(root / "p1.nsp", b"first")
            second = self._write(root / "p2.nsp", b"second")
            healthy = self._write(root / "h.nsp", b"healthy")
            rows = [
                self._svd_row(
                    "p1",
                    first,
                    "101",
                    "speaker-p",
                    "pathological",
                    "Group A",
                    age=40,
                ),
                self._svd_row(
                    "p2",
                    second,
                    "101",
                    "speaker-p",
                    "pathological",
                    "Group B",
                    age=40,
                ),
                self._svd_row(
                    "h",
                    healthy,
                    "202",
                    "speaker-h",
                    "healthy",
                    "healthy",
                    age=35,
                ),
            ]

            with self.assertRaisesRegex(
                ValueError,
                "Different hashes",
            ):
                SVDTrainingManifestBuilder(
                    SVDTrainingManifestConfig()
                ).build(pd.DataFrame(rows))

    def test_hupa_builder_reports_conflicting_duplicate_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate_a = self._write(root / "p1.wav", b"same-audio")
            duplicate_b = self._write(root / "p2.wav", b"same-audio")
            healthy = self._write(root / "h1.wav", b"healthy")

            rows = [
                self._hupa_row(
                    "p1",
                    duplicate_a,
                    "pathological",
                    age=40,
                ),
                self._hupa_row(
                    "p2",
                    duplicate_b,
                    "pathological",
                    age=60,
                ),
                self._hupa_row(
                    "h1",
                    healthy,
                    "healthy",
                    age=30,
                ),
            ]

            result = HUPATrainingManifestBuilder(
                HUPATrainingManifestConfig()
            ).build(pd.DataFrame(rows))

            self.assertEqual(len(result.training_manifest), 2)
            self.assertEqual(len(result.duplicate_groups), 2)

            pathological = result.training_manifest.loc[
                result.training_manifest["label"].eq("pathological")
            ].iloc[0]
            self.assertTrue(pd.isna(pathological["age"]))
            self.assertIn(
                "age",
                pathological["metadata_conflict_columns"],
            )
            self.assertEqual(pathological["source_count"], 2)

    def test_hupa_adult_cohort_excludes_duplicate_age_conflicts(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate_a = self._write(root / "p1.wav", b"same-audio")
            duplicate_b = self._write(root / "p2.wav", b"same-audio")
            pathological = self._write(
                root / "p3.wav",
                b"pathological",
            )
            healthy = self._write(root / "h1.wav", b"healthy")
            rows = [
                self._hupa_row(
                    "p1",
                    duplicate_a,
                    "pathological",
                    age=40,
                ),
                self._hupa_row(
                    "p2",
                    duplicate_b,
                    "pathological",
                    age=60,
                ),
                self._hupa_row(
                    "p3",
                    pathological,
                    "pathological",
                    age=50,
                ),
                self._hupa_row(
                    "h1",
                    healthy,
                    "healthy",
                    age=30,
                ),
            ]

            result = HUPATrainingManifestBuilder(
                HUPATrainingManifestConfig(
                    adults_only=True,
                    minimum_age=18,
                )
            ).build(pd.DataFrame(rows))

            self.assertEqual(len(result.training_manifest), 2)
            self.assertTrue(
                pd.to_numeric(
                    result.training_manifest["age"],
                    errors="coerce",
                ).ge(18).all()
            )
            self.assertEqual(
                set(result.excluded_samples["exclusion_reason"]),
                {"conflicting_age_for_duplicate_audio"},
            )
            self.assertEqual(len(result.excluded_samples), 2)

    def test_hupa_builder_rejects_identical_audio_with_two_labels(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write(root / "p.wav", b"same")
            second = self._write(root / "h.wav", b"same")
            rows = [
                self._hupa_row(
                    "p",
                    first,
                    "pathological",
                    age=40,
                ),
                self._hupa_row(
                    "h",
                    second,
                    "healthy",
                    age=40,
                ),
            ]

            with self.assertRaisesRegex(
                ValueError,
                "conflicting binary labels",
            ):
                HUPATrainingManifestBuilder(
                    HUPATrainingManifestConfig()
                ).build(pd.DataFrame(rows))

    @staticmethod
    def _write(path: Path, content: bytes) -> Path:
        path.write_bytes(content)
        return path

    @staticmethod
    def _svd_row(
        sample_id: str,
        filepath: Path,
        recording_id: str,
        speaker_id: str,
        label: str,
        pathology_group: str,
        age: float,
        duration: float = 1.0,
        vowel: str = "a",
        condition: str = "n",
        audio_status: str = "ok",
    ) -> dict:
        return {
            "sample_id": sample_id,
            "base": "SVD",
            "filepath": str(filepath),
            "relative_path": filepath.name,
            "recording_id": recording_id,
            "speaker_id": speaker_id,
            "label": label,
            "vowel": vowel,
            "condition": condition,
            "pathology_group": pathology_group,
            "pathology_group_key": pathology_group.lower(),
            "pathology": pathology_group,
            "age": age,
            "duration": duration,
            "audio_read_status": audio_status,
            "audio_read_error": (
                "could not read" if audio_status != "ok" else None
            ),
        }

    @staticmethod
    def _hupa_row(
        sample_id: str,
        filepath: Path,
        label: str,
        age: float,
    ) -> dict:
        return {
            "sample_id": sample_id,
            "base": "HUPA",
            "filepath": str(filepath),
            "relative_path": filepath.name,
            "filename": filepath.name,
            "file_stem": filepath.stem,
            "file_key": filepath.stem,
            "speaker_id": sample_id,
            "speaker_id_source": "sample_id_assumption",
            "label": label,
            "vowel": "a",
            "pitch": None,
            "samplerate": 16_000,
            "duration": 1.0,
            "channels": 1,
            "audio_read_status": "ok",
            "audio_read_error": None,
            "age": age,
            "sex": "female",
            "pathology_code": "x",
            "pathology": label,
        }


if __name__ == "__main__":
    unittest.main()
