from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


@dataclass
class ExperimentPaths:
    root_dir: Path
    dataset_name: str
    experiment_name: str

    @classmethod
    def create(
        cls,
        data_root: Path,
        dataset_name: str,
        experiment_name: str,
    ) -> "ExperimentPaths":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_name = (
            f"{timestamp}_"
            f"{slugify(dataset_name)}_"
            f"{slugify(experiment_name)}"
        )

        dataset_dir = slugify(dataset_name).upper()
        root_dir = (
            data_root
            / dataset_dir
            / "experiments"
            / run_name
        )

        paths = cls(
            root_dir=root_dir,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
        )

        paths.make_dirs()

        return paths

    @classmethod
    def open_existing(
        cls,
        root_dir: str | Path,
        dataset_name: str,
        experiment_name: str,
    ) -> "ExperimentPaths":
        root = Path(root_dir).resolve()

        if not root.is_dir():
            raise FileNotFoundError(
                f"Resume experiment root does not exist: {root}"
            )

        paths = cls(
            root_dir=root,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
        )
        paths.make_dirs()
        return paths

    @property
    def manifests_dir(self) -> Path:
        return self.root_dir / "manifests"

    @property
    def profiles_dir(self) -> Path:
        return self.root_dir / "profiles"

    @property
    def features_dir(self) -> Path:
        return self.root_dir / "features"

    @property
    def figures_dir(self) -> Path:
        return self.root_dir / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.root_dir / "reports"

    @property
    def config_path(self) -> Path:
        return self.root_dir / "config.json"

    def make_dirs(self) -> None:
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
