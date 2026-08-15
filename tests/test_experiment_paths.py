from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from classes.experiment.path_manager.experiment_paths import (
    ExperimentPaths,
)


class ExperimentPathsTests(unittest.TestCase):
    def test_dataset_outputs_are_outside_dataset_source_root(self) -> None:
        with TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"

            paths = ExperimentPaths.create(
                data_root=data_root,
                dataset_name="SVD",
                experiment_name="a_normal",
            )

            self.assertEqual(
                paths.root_dir.parents[1],
                data_root / "SVD",
            )
            self.assertTrue(paths.manifests_dir.exists())
            self.assertTrue(paths.reports_dir.exists())

    def test_opens_existing_experiment_without_new_timestamp(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "existing"
            root.mkdir()

            paths = ExperimentPaths.open_existing(
                root_dir=root,
                dataset_name="SVD",
                experiment_name="resume",
            )

            self.assertEqual(paths.root_dir, root.resolve())
            self.assertTrue(paths.features_dir.exists())


if __name__ == "__main__":
    unittest.main()
