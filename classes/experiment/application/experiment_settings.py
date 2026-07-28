from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSettings:
    project_root: Path
    data_root: Path
    hupa_root: Path
    svd_root: Path

    @classmethod
    def default(
        cls,
        project_root: str | Path | None = None,
    ) -> ExperimentSettings:
        resolved_project_root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[3]
        )

        return cls(
            project_root=resolved_project_root,
            data_root=resolved_project_root / "data",
            hupa_root=Path(
                "/mnt/d/masters_degree/datasets/hupa/"
                "BDAtos HUPA Segmentada"
            ),
            svd_root=Path(
                "/mnt/d/masters_degree/datasets/svd"
            ),
        )
