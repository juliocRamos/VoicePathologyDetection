from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DatasetAdapter(ABC):
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    @abstractmethod
    def build_manifest(self) -> pd.DataFrame:
        """Translate a physical dataset into a raw manifest."""
