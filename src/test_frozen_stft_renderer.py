"""Pixel-parity regression for the frozen per-window STFT renderer."""

import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from frozen_stft_renderer import render_stft_pngs
from fusion_engine import PROJECT_ROOT


TRIAL_ID = "1_20160518_trial17"
RAW_SAMPLE = PROJECT_ROOT / "data/output/fear" / f"{TRIAL_ID}.mat"
REFERENCE_DIRECTORY = PROJECT_ROOT / "data/processed/crnn_dataset/validation/fear"


class FrozenSTFTRendererTests(unittest.TestCase):
    def test_validation_pngs_are_reproduced_pixel_identically(self):
        self.assertTrue(RAW_SAMPLE.is_file(), f"Missing validation EEG: {RAW_SAMPLE}")
        references = sorted(
            REFERENCE_DIRECTORY.glob(f"{TRIAL_ID}_window_*_STFT_RGB.png")
        )
        self.assertEqual(len(references), 36, "Expected all 36 saved validation STFT windows")
        eeg = loadmat(RAW_SAMPLE)["EEG"]
        window_ids = [
            int(re.search(r"_window_(\d+)_", path.name).group(1))
            for path in references
        ]

        with tempfile.TemporaryDirectory(prefix="frozen_stft_parity_") as output_directory:
            generated = render_stft_pngs(
                eeg,
                output_directory,
                recording_id=TRIAL_ID,
                window_ids=window_ids,
            )
            self.assertEqual(len(generated), len(references))
            for reference, output in zip(references, generated):
                with Image.open(reference) as expected, Image.open(output) as actual:
                    self.assertEqual(expected.size, (1024, 1024))
                    self.assertEqual(expected.mode, "RGBA")
                    self.assertEqual(actual.size, expected.size)
                    self.assertEqual(actual.mode, expected.mode)
                    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
