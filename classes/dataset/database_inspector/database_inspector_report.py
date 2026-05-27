from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseInspectorReport:
    root_dir: Path
    total_files: int
    audio_files: int
    metadata_files: int
    extensions: dict
    audio_examples: list[str]
    metadata_examples: list[str]
