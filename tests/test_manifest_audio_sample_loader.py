from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from classes.audio_sample.audio_loader.audio_file_reader import (
    AudioFileReader,
    LoadedAudio,
)
from classes.audio_sample.audio_loader.manifest_audio_sample_loader import (
    ManifestAudioSampleLoader,
)


class StubAudioFileReader(AudioFileReader):
    def read(
        self,
        filepath: str | Path,
    ) -> LoadedAudio:
        return LoadedAudio(
            signal=np.ones(1_600, dtype=np.float32),
            sample_rate=16_000,
            filepath=Path(filepath),
        )


class ManifestAudioSampleLoaderTests(unittest.TestCase):
    def test_manifest_mapping_is_separate_from_preprocessing(
        self,
    ) -> None:
        row = pd.Series({
            "sample_id": "sample-1",
            "base": "TEST",
            "filepath": "/tmp/sample.wav",
            "label": "pathological",
            "speaker_id": "speaker-1",
            "age": np.float64(42),
            "condition": "n",
        })
        loader = ManifestAudioSampleLoader(
            audio_reader=StubAudioFileReader()
        )

        sample = loader.load(row)

        self.assertEqual(sample.sample_id, "sample-1")
        self.assertEqual(sample.speaker_id, "speaker-1")
        self.assertEqual(sample.age, 42.0)
        self.assertEqual(sample.duration, 0.1)
        self.assertEqual(sample.metadata["condition"], "n")


if __name__ == "__main__":
    unittest.main()
