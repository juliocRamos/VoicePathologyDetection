from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

@dataclass
class AudioSample:
    sample_id: str
    base: str
    filepath: Path
    signal: np.ndarray
    sr: int

    label: Optional[str] = None
    speaker_id: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[float] = None
    pathology: Optional[str] = None
    pathology_code: Optional[str] = None
    vowel: Optional[str] = None
    pitch: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.signal) / self.sr if self.sr > 0 else 0.0