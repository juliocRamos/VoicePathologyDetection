from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import shutil
import subprocess
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import StandardScaler

from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.compute_backend_runtime import (
    TORCH_DETERMINISTIC_ALGORITHMS,
    TORCH_DETERMINISTIC_WARN_ONLY,
)
from classes.experiment.training.training_plan import (
    FeatureScenario,
    ModelSpec,
)


class ExperimentalProtocolWriter:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        output_dir: str | Path,
    ) -> None:
        self.output_dir = Path(output_dir)

    def write(
        self,
        config: TrainingConfig,
        feature_scenarios: list[FeatureScenario],
        model_specs: list[ModelSpec],
    ) -> str:
        hash_payload = {
            "schema_version": self.SCHEMA_VERSION,
            "protocol_version": config.protocol_version,
            "training_config": self._normalize(asdict(config)),
            "feature_scenarios": [
                self._normalize(asdict(scenario))
                for scenario in feature_scenarios
            ],
            "models": [
                {
                    "name": spec.name,
                    "estimator": self._normalize(spec.estimator),
                    "parameter_grid": self._normalize(
                        spec.param_grid
                    ),
                    "uses_balanced_sample_weight": (
                        spec.use_balanced_sample_weight
                    ),
                    "rationale": spec.rationale,
                }
                for spec in model_specs
            ],
            "shared_preprocessing": (
                self._shared_preprocessing(
                    config=config,
                    model_specs=model_specs,
                )
            ),
            "selection_policy": {
                "primary_metric": config.scoring,
                "within_family": "one_standard_error",
                "minimum_score_tolerance": (
                    config.selection_score_tolerance
                ),
                "global_family_order": None,
                "global_tie_breakers": [
                    "lower_cv_standard_deviation",
                    "smaller_absolute_training_cv_gap",
                    "fewer_selected_features",
                    "higher_cv_score",
                ],
                "primary_evaluation": (
                    "single_global_champion_selected_by_training_cv"
                ),
                "secondary_family_comparison": (
                    "training_cv_selected_svm_and_mlp_champions"
                ),
                "candidate_ranking_source": "training_cv",
            },
            "seed_policy": {
                "master_seed": config.random_state,
                "holdout_seed": config.random_state,
                "inner_cv_seed": config.random_state,
                "nested_outer_seeds": [
                    config.random_state + repeat
                    for repeat in range(
                        config.nested_cv_repeats
                    )
                ],
                "model_initialization_seed": config.random_state,
                "grouped_bootstrap_seed": config.random_state,
                "holdout_repetitions": 1,
                "seed_selection": (
                    "Seeds are prespecified; no run or seed is selected "
                    "from holdout performance."
                ),
            },
            "determinism_policy": {
                "torch_deterministic_algorithms": (
                    TORCH_DETERMINISTIC_ALGORITHMS
                ),
                "warn_only_for_unsupported_operations": (
                    TORCH_DETERMINISTIC_WARN_ONLY
                ),
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
        }
        canonical_payload = json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        protocol_hash = sha256(
            canonical_payload.encode("utf-8")
        ).hexdigest()
        artifact = {
            **hash_payload,
            "protocol_hash": protocol_hash,
            "execution_environment": (
                self._execution_environment(config)
            ),
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (
            self.output_dir / "experimental_protocol.json"
        ).write_text(
            json.dumps(
                artifact,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (
            self.output_dir / "experimental_protocol.md"
        ).write_text(
            self._build_markdown(artifact),
            encoding="utf-8",
        )
        return protocol_hash

    @classmethod
    def _execution_environment(
        cls,
        config: TrainingConfig,
    ) -> dict[str, Any]:
        distribution_names = (
            "numpy",
            "pandas",
            "scikit-learn",
            "torch",
            "skorch",
            "cuml-cu13",
            "cupy-cuda13x",
            "cuda-toolkit",
        )
        packages: dict[str, str | None] = {}

        for distribution_name in distribution_names:
            try:
                packages[distribution_name] = version(
                    distribution_name
                )
            except PackageNotFoundError:
                packages[distribution_name] = None

        environment: dict[str, Any] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        }

        if not config.compute_backend.uses_cuda:
            return environment

        try:
            import torch

            cuda_environment: dict[str, Any] = {
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "device_count": torch.cuda.device_count(),
            }

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(
                    device
                )
                cuda_environment.update({
                    "device_index": device,
                    "device_name": properties.name,
                    "compute_capability": (
                        f"{properties.major}.{properties.minor}"
                    ),
                    "total_memory_bytes": (
                        properties.total_memory
                    ),
                })

            environment["cuda"] = cuda_environment
        except (ImportError, RuntimeError) as exc:
            environment["cuda"] = {
                "inspection_error": (
                    f"{type(exc).__name__}: {exc}"
                )
            }

        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is not None:
            try:
                completed = subprocess.run(
                    [
                        nvidia_smi,
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=5,
                )
                environment["nvidia_smi"] = (
                    completed.stdout.strip()
                    if completed.returncode == 0
                    else completed.stderr.strip()
                )
            except (
                OSError,
                subprocess.SubprocessError,
            ) as exc:
                environment["nvidia_smi"] = (
                    f"{type(exc).__name__}: {exc}"
                )

        return environment

    @staticmethod
    def _shared_preprocessing(
        config: TrainingConfig,
        model_specs: list[ModelSpec],
    ) -> dict[str, Any]:
        if not config.eligible_for_final_reporting:
            return {
                "enforced": False,
                "rationale": (
                    "Shared preprocessing is enforced only for the "
                    "confirmatory GPU protocol."
                ),
            }

        expected_percentiles = [10, 25, 50]

        for spec in model_specs:
            for grid in spec.param_grid:
                strategies = grid.get("imputer__strategy", [])
                scalers = grid.get("scaler", [])
                selectors = grid.get("selector", [])
                percentiles = [
                    selector.percentile
                    for selector in selectors
                    if isinstance(
                        selector,
                        SelectPercentile,
                    )
                ]

                if (
                    strategies != ["median"]
                    or len(scalers) != 1
                    or not isinstance(
                        scalers[0],
                        StandardScaler,
                    )
                    or percentiles != expected_percentiles
                    or len(selectors) != len(percentiles)
                ):
                    raise ValueError(
                        "Confirmatory GPU models must share median "
                        "imputation, StandardScaler, and feature "
                        "selection percentiles 10, 25, and 50."
                    )

        return {
            "enforced": True,
            "imputation": "median",
            "scaling": "StandardScaler",
            "feature_selection_percentiles": expected_percentiles,
            "rationale": (
                "The same preprocessing search space is used by all "
                "confirmatory GPU model families."
            ),
        }

    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value

        if value is None or isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, dict):
            return {
                str(key): cls._normalize(item)
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: str(pair[0]),
                )
            }

        if isinstance(value, set):
            return [
                cls._normalize(item)
                for item in sorted(value, key=str)
            ]

        if isinstance(value, (list, tuple)):
            return [cls._normalize(item) for item in value]

        if isinstance(value, BaseEstimator):
            return {
                "type": cls._qualified_name(type(value)),
                "parameters": cls._normalize(
                    value.get_params(deep=False)
                ),
            }

        if isinstance(value, type) or callable(value):
            return cls._qualified_name(value)

        return {
            "type": cls._qualified_name(type(value)),
        }

    @staticmethod
    def _qualified_name(value: Any) -> str:
        module = getattr(value, "__module__", "")
        qualified_name = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", type(value).__name__),
        )
        return (
            f"{module}.{qualified_name}"
            if module
            else str(qualified_name)
        )

    @staticmethod
    def _build_markdown(
        artifact: dict[str, Any],
    ) -> str:
        shared_preprocessing = artifact[
            "shared_preprocessing"
        ]
        preprocessing_description = (
            "All confirmatory model families use median imputation, "
            "`StandardScaler`, and feature-selection percentiles "
            "`10`, `25`, and `50` inside grouped cross-validation."
            if shared_preprocessing.get("enforced")
            else shared_preprocessing["rationale"]
        )
        lines = [
            "# Experimental protocol",
            "",
            f"- Version: `{artifact['protocol_version']}`",
            f"- Hash: `{artifact['protocol_hash']}`",
            (
                "- Backend: `"
                f"{artifact['training_config']['compute_backend']}`"
            ),
            (
                "- Eligible for final reporting: `"
                f"{artifact['training_config']['eligible_for_final_reporting']}`"
            ),
            "",
            "## Shared preprocessing",
            "",
            preprocessing_description,
            "",
            "## Selection",
            "",
            "Balanced accuracy is the primary metric. The one-standard-"
            "error rule is applied within each family. Global comparison "
            "does not impose a model-family order; equivalent candidates "
            "are resolved by CV stability, training–CV gap, selected "
            "feature count, and CV score.",
            "",
            "The global champion remains the single primary result. "
            "For a prespecified secondary comparison, one SVM and one "
            "MLP champion selected exclusively on training CV are "
            "refitted on the full training partition and evaluated on "
            "the same holdout. Candidate rankings use training CV, not "
            "holdout performance.",
            "",
            "## Seeds and runtime budget",
            "",
            (
                f"The holdout, inner CV, model initialization, and grouped "
                f"bootstrap use the prespecified master seed "
                f"`{artifact['seed_policy']['master_seed']}`. Repeated "
                f"nested CV uses outer seeds "
                f"`{artifact['seed_policy']['nested_outer_seeds']}`. "
                "All repeated estimates are reported; no best seed is "
                "selected."
            ),
            "",
            (
                "The final holdout is evaluated once. Repeated nested "
                "grouped CV provides the mean and variability across "
                "partitions without multiplying full holdout executions."
            ),
            "",
            "## Execution environment",
            "",
            "```json",
            json.dumps(
                artifact["execution_environment"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Models",
            "",
        ]

        for model in artifact["models"]:
            lines.extend([
                f"### `{model['name']}`",
                "",
                model["rationale"],
                "",
                "```json",
                json.dumps(
                    model["parameter_grid"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
            ])

        return "\n".join(lines) + "\n"
