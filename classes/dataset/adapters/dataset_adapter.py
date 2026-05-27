from pathlib import Path
import pandas as pd

class DatasetAdapter:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def build_manifest(self) -> pd.DataFrame:
        raise NotImplementedError