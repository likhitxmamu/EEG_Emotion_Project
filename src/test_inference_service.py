"""Input-validation and automatic-rendering checks for the frozen service."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from scipy.io import savemat

import inference_service as inference_module
from inference_service import InferenceService, _load_eeg_input


class InferenceServiceValidationTests(unittest.TestCase):
    def setUp(self):
        self.service = InferenceService(device="cpu")
        self.valid_eeg = np.zeros((62, 400), dtype=np.float32)

    def assert_error_code(self, result, expected):
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], expected)
        self.assertTrue(result["error"]["message"])

    def test_invalid_numeric_eeg_is_rejected_cleanly(self):
        cases = (
            (np.zeros((61, 400), dtype=np.float32), "channel count"),
            (np.zeros((62, 399), dtype=np.float32), "sample count"),
            (np.full((62, 400), np.nan, dtype=np.float32), "NaN"),
            (np.full((62, 400), np.inf, dtype=np.float32), "infinity"),
        )
        for eeg, description in cases:
            with self.subTest(description=description):
                self.assert_error_code(self.service.predict({"eeg": eeg}), "invalid_eeg")

    def test_mat_file_errors_and_label_free_input(self):
        with tempfile.TemporaryDirectory(prefix="eeg_service_mat_") as directory:
            malformed = Path(directory) / "malformed.mat"
            malformed.write_bytes(b"not a MATLAB file")
            self.assert_error_code(
                self.service.predict({"mat_path": str(malformed)}), "malformed_mat"
            )

            missing_eeg = Path(directory) / "missing_eeg.mat"
            savemat(missing_eeg, {"other": np.zeros((62, 400), dtype=np.float32)})
            self.assert_error_code(
                self.service.predict({"mat_path": str(missing_eeg)}),
                "missing_eeg_variable",
            )

            no_label = Path(directory) / "eeg_only.mat"
            savemat(no_label, {"EEG": self.valid_eeg})
            eeg, input_type, source_path = _load_eeg_input({"mat_path": str(no_label)})
            self.assertEqual(eeg.shape, (62, 400))
            self.assertEqual(input_type, "mat")
            self.assertEqual(source_path, no_label)

    def test_invalid_png_shape_and_channel_mode_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="eeg_service_png_") as directory:
            bad_size = Path(directory) / "bad_size.png"
            Image.new("RGBA", (16, 16)).save(bad_size)
            self.assert_error_code(
                self.service.predict(
                    {"eeg": self.valid_eeg, "stft_image_paths": [str(bad_size)]}
                ),
                "invalid_stft_image",
            )

            bad_mode = Path(directory) / "bad_mode.png"
            Image.new("RGB", (1024, 1024)).save(bad_mode)
            self.assert_error_code(
                self.service.predict(
                    {"eeg": self.valid_eeg, "stft_image_paths": [str(bad_mode)]}
                ),
                "invalid_stft_image",
            )

    def test_automatic_rendering_uses_validated_temporary_pngs(self):
        observed = {}

        def fake_renderer(eeg, output_directory, recording_id):
            self.assertEqual(eeg.shape, (62, 400))
            path = Path(output_directory) / f"{recording_id}_window_0001_STFT_RGB.png"
            Image.new("RGBA", (1024, 1024)).save(path)
            observed["path"] = path
            return [path]

        def capture_inference(*args, **kwargs):
            paths = args[4]
            window_ids = args[5]
            self.assertEqual(window_ids, [1])
            self.assertTrue(paths[0].is_file())
            observed["path_during_inference"] = paths[0]
            self.assertEqual(kwargs["stft_source"], "frozen_stft_renderer")
            return {"ok": True, "prediction": "fear"}

        with mock.patch.object(inference_module, "render_stft_pngs", side_effect=fake_renderer):
            with mock.patch.object(
                self.service,
                "_predict_with_stft_paths",
                side_effect=capture_inference,
            ):
                result = self.service.predict({"eeg": self.valid_eeg})

        self.assertEqual(result["prediction"], "fear")
        self.assertTrue(observed["path_during_inference"].is_relative_to(observed["path"].parent))
        self.assertFalse(observed["path"].exists(), "Temporary STFT files should be cleaned up")

    def test_explicit_reference_pngs_remain_supported(self):
        with tempfile.TemporaryDirectory(prefix="eeg_service_reference_") as directory:
            image_path = Path(directory) / "window.png"
            Image.new("RGBA", (1024, 1024)).save(image_path)

            def capture_inference(*args, **kwargs):
                self.assertEqual(args[4], [image_path])
                self.assertEqual(kwargs["stft_source"], "provided_reference_images")
                return {"ok": True, "prediction": "fear"}

            with mock.patch.object(
                self.service,
                "_predict_with_stft_paths",
                side_effect=capture_inference,
            ):
                result = self.service.predict(
                    {"eeg": self.valid_eeg, "stft_image_paths": [str(image_path)]}
                )
        self.assertEqual(result["prediction"], "fear")

    def test_renderer_failure_has_machine_readable_error(self):
        with mock.patch.object(
            inference_module,
            "render_stft_pngs",
            side_effect=RuntimeError("simulated rendering failure"),
        ):
            result = self.service.predict({"eeg": self.valid_eeg})
        self.assert_error_code(result, "stft_generation_failed")

    def test_missing_checkpoint_has_machine_readable_error(self):
        with tempfile.TemporaryDirectory(prefix="eeg_service_checkpoint_") as directory:
            image_path = Path(directory) / "window.png"
            Image.new("RGBA", (1024, 1024)).save(image_path)
            with mock.patch.object(
                inference_module,
                "load_models",
                side_effect=FileNotFoundError("Required checkpoint not found: frozen.pth"),
            ):
                result = self.service.predict(
                    {"eeg": self.valid_eeg, "stft_image_paths": [str(image_path)]}
                )
        self.assert_error_code(result, "missing_checkpoint")


if __name__ == "__main__":
    unittest.main(verbosity=2)
