from pathlib import Path
import unittest

import numpy as np

from classes.audio_sample.audio_loader.preprocessing.audio_preprocess_config import (
    AudioPreprocessConfig,
)
from classes.audio_sample.audio_loader.preprocessing.audio_preprocessor import (
    AudioPreprocessor,
)
from classes.audio_sample.audio_sample import AudioSample


class AudioPreprocessorTests(unittest.TestCase):
    def test_short_valid_signal_is_not_a_cohort_error(self) -> None:
        sample = AudioSample(
            sample_id="short",
            base="TEST",
            filepath=Path("short.wav"),
            signal=np.ones(1_600, dtype=np.float32),
            sr=16_000,
        )
        preprocessor = AudioPreprocessor(
            AudioPreprocessConfig(
                target_sr=16_000,
                remove_dc=False,
                normalize_rms=False,
            )
        )

        processed = preprocessor.process(sample)

        self.assertEqual(len(processed.signal), 1_600)
        self.assertAlmostEqual(processed.duration, 0.1)

    def test_empty_signal_is_a_technical_error(self) -> None:
        sample = AudioSample(
            sample_id="empty",
            base="TEST",
            filepath=Path("empty.wav"),
            signal=np.array([], dtype=np.float32),
            sr=16_000,
        )
        preprocessor = AudioPreprocessor(
            AudioPreprocessConfig()
        )

        with self.assertRaisesRegex(
            ValueError,
            "empty audio signal",
        ):
            preprocessor.process(sample)


if __name__ == "__main__":
    unittest.main()
