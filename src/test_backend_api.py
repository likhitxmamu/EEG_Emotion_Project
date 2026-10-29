"""Contract tests for the WSGI EEG prediction API."""

import csv
import io
import json
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from wsgiref.util import setup_testing_defaults

_SOURCE_DIRECTORY = str(Path(__file__).resolve().parent)
if _SOURCE_DIRECTORY not in sys.path:
    sys.path.insert(0, _SOURCE_DIRECTORY)

import numpy as np
from scipy.io import savemat

import inference_service as inference_module
from backend_api import LOCAL_FRONTEND_ORIGIN, create_app
from frozen_stft_renderer import render_stft_pngs
from fusion_engine import CLASS_NAMES, MODEL_NAMES, PROJECT_ROOT
from inference_service import InferenceService


TRIAL_ID = "1_20160518_trial17"
RAW_SAMPLE = PROJECT_ROOT / "data/output/fear" / f"{TRIAL_ID}.mat"
STFT_REFERENCE_DIR = PROJECT_ROOT / "data/processed/crnn_dataset/validation/fear"
VALIDATION_PREDICTIONS = (
    PROJECT_ROOT / "models/fusion/adaptive_run_20261027_132228/validation_predictions.csv"
)
TOLERANCE = 1e-5


def _mat_bytes(values):
    stream = io.BytesIO()
    savemat(stream, values)
    return stream.getvalue()


def _multipart_body(filename, file_bytes, boundary="ApiContractBoundary"):
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("ascii")
    return body + file_bytes + f"\r\n--{boundary}--\r\n".encode("ascii"), boundary


def _call_wsgi(app, method="GET", path="/", body=b"", content_type="", origin=None):
    environ = {}
    setup_testing_defaults(environ)
    environ.update(
        {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": io.BytesIO(body),
        }
    )
    if origin is not None:
        environ["HTTP_ORIGIN"] = origin
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = int(status.split(" ", 1)[0])
        captured["headers"] = dict(headers)

    response_body = b"".join(app(environ, start_response))
    captured["json"] = json.loads(response_body.decode("utf-8")) if response_body else None
    captured["body"] = response_body
    return captured


def _upload(app, filename, file_bytes):
    body, boundary = _multipart_body(filename, file_bytes)
    return _call_wsgi(
        app,
        method="POST",
        path="/predict",
        body=body,
        content_type=f"multipart/form-data; boundary={boundary}",
    )


def _successful_service_result():
    probabilities = {name: value for name, value in zip(CLASS_NAMES, (0.1, 0.2, 0.3, 0.4))}
    models = {}
    for index, model_name in enumerate(MODEL_NAMES):
        models[model_name] = {
            "prediction": "sad",
            "confidence": 0.4,
            "confidence_type": "not_calibrated",
            "confidence_description": "Frozen model confidence description.",
            "probabilities": probabilities,
            "window_count": 12 + index,
            "window_outputs": [{"logits": [1, 2, 3, 4]}],
        }
    return {
        "ok": True,
        "prediction": "sad",
        "confidence": 0.4,
        "adaptive_confidence_score": 0.4,
        "confidence_type": "frozen_adaptive_confidence_score",
        "confidence_description": "Frozen adaptive confidence score; not calibrated.",
        "class_order": list(CLASS_NAMES),
        "probabilities": probabilities,
        "models": models,
        "adaptive_fusion": {
            "reliability_prior_weights": {name: 0.25 for name in MODEL_NAMES},
            "trial_weights": {name: 0.25 for name in MODEL_NAMES},
            "normalized_entropy_by_model": {name: 0.8 for name in MODEL_NAMES},
            "certainty_gate_by_model": {name: 0.2 for name in MODEL_NAMES},
        },
        "input": {
            "source_file": "/private/tmp/internal/upload.mat",
            "raw_window_count": 15,
            "stft_window_count": 16,
        },
    }


class ServiceSpy:
    def __init__(self, result):
        self.result = result
        self.requests = []
        self.file_bytes_seen = []

    def predict(self, request):
        self.requests.append(dict(request))
        mat_path = Path(request["mat_path"])
        self.file_bytes_seen.append(mat_path.read_bytes())
        self.path_seen = mat_path
        return self.result


class BackendAPITests(unittest.TestCase):
    def test_health_succeeds_without_calling_inference(self):
        service = ServiceSpy(_successful_service_result())
        response = _call_wsgi(create_app(service), method="GET", path="/health")
        self.assertEqual(response["status"], 200)
        self.assertEqual(response["json"], {"ok": True, "status": "ok"})
        self.assertEqual(service.requests, [])

    def test_cors_is_limited_to_the_local_frontend_origin(self):
        app = create_app(ServiceSpy(_successful_service_result()))
        allowed = _call_wsgi(
            app,
            method="GET",
            path="/health",
            origin=LOCAL_FRONTEND_ORIGIN,
        )
        self.assertEqual(
            allowed["headers"].get("Access-Control-Allow-Origin"),
            LOCAL_FRONTEND_ORIGIN,
        )
        self.assertEqual(allowed["headers"].get("Vary"), "Origin")

        blocked = _call_wsgi(
            app,
            method="GET",
            path="/health",
            origin="https://example.invalid",
        )
        self.assertNotIn("Access-Control-Allow-Origin", blocked["headers"])

    def test_local_frontend_preflight_allows_only_the_existing_predict_contract(self):
        response = _call_wsgi(
            create_app(ServiceSpy(_successful_service_result())),
            method="OPTIONS",
            path="/predict",
            origin=LOCAL_FRONTEND_ORIGIN,
        )
        self.assertEqual(response["status"], 204)
        self.assertIsNone(response["json"])
        self.assertEqual(
            response["headers"].get("Access-Control-Allow-Origin"),
            LOCAL_FRONTEND_ORIGIN,
        )
        self.assertEqual(response["headers"].get("Access-Control-Allow-Methods"), "POST, OPTIONS")
        self.assertEqual(response["headers"].get("Access-Control-Allow-Headers"), "Content-Type")

    def test_valid_mat_upload_reaches_service_and_preserves_public_values(self):
        mat_content = _mat_bytes({"EEG": np.zeros((62, 400), dtype=np.float32)})
        service_result = _successful_service_result()
        service = ServiceSpy(service_result)
        response = _upload(create_app(service), "session.mat", mat_content)

        self.assertEqual(response["status"], 200)
        self.assertEqual(len(service.requests), 1)
        self.assertEqual(set(service.requests[0]), {"mat_path"})
        self.assertEqual(service.file_bytes_seen, [mat_content])
        self.assertFalse(service.path_seen.exists(), "Uploaded temporary files must be removed")

        payload = response["json"]
        self.assertTrue(payload["ok"])
        for field in (
            "prediction",
            "confidence",
            "class_probabilities",
            "models",
            "window_counts",
            "adaptive_fusion",
        ):
            self.assertIn(field, payload)
        self.assertEqual(payload["prediction"], service_result["prediction"])
        self.assertEqual(payload["confidence"], service_result["confidence"])
        self.assertEqual(
            payload["adaptive_confidence_score"],
            service_result["adaptive_confidence_score"],
        )
        self.assertEqual(payload["class_order"], list(CLASS_NAMES))
        self.assertEqual(payload["class_probabilities"], service_result["probabilities"])
        for model_name in MODEL_NAMES:
            self.assertEqual(
                payload["models"][model_name]["class_probabilities"],
                service_result["models"][model_name]["probabilities"],
            )
            self.assertEqual(
                payload["models"][model_name]["window_count"],
                service_result["models"][model_name]["window_count"],
            )
        self.assertEqual(
            payload["adaptive_fusion"]["reliability_prior_weights"],
            service_result["adaptive_fusion"]["reliability_prior_weights"],
        )
        self.assertEqual(
            payload["adaptive_fusion"]["weights"],
            service_result["adaptive_fusion"]["trial_weights"],
        )
        self.assertEqual(
            payload["adaptive_fusion"]["normalized_entropy_by_model"],
            service_result["adaptive_fusion"]["normalized_entropy_by_model"],
        )
        self.assertEqual(
            payload["adaptive_fusion"]["certainty_gate_by_model"],
            service_result["adaptive_fusion"]["certainty_gate_by_model"],
        )
        response_text = response["body"].decode("utf-8")
        self.assertNotIn("/private/tmp", response_text)
        self.assertNotIn("source_file", response_text)
        self.assertNotIn("window_outputs", response_text)

    def test_non_mat_upload_returns_a_structured_error_without_inference(self):
        service = ServiceSpy(_successful_service_result())
        response = _upload(create_app(service), "session.csv", b"a,b,c\n")
        self.assertEqual(response["status"], 415)
        self.assertEqual(
            response["json"],
            {
                "ok": False,
                "error": {
                    "code": "unsupported_file_type",
                    "message": "Only MATLAB .mat files are accepted.",
                },
            },
        )
        self.assertEqual(service.requests, [])

    def test_missing_eeg_variable_returns_safe_structured_error(self):
        mat_content = _mat_bytes({"other": np.zeros((62, 400), dtype=np.float32)})
        response = _upload(create_app(InferenceService(device="cpu")), "missing.mat", mat_content)
        self.assertEqual(response["status"], 422)
        self.assertEqual(response["json"]["error"]["code"], "missing_eeg_variable")
        self.assertEqual(
            response["json"]["error"]["message"],
            "The MATLAB file must contain an EEG variable.",
        )
        self.assertNotIn("missing.mat", response["body"].decode("utf-8"))

    def test_malformed_mat_returns_safe_structured_error(self):
        response = _upload(
            create_app(InferenceService(device="cpu")), "malformed.mat", b"not a MAT file"
        )
        self.assertEqual(response["status"], 422)
        self.assertEqual(response["json"]["error"]["code"], "malformed_mat")
        self.assertEqual(
            response["json"]["error"]["message"],
            "The uploaded file could not be read as a MATLAB recording.",
        )

    def test_service_failure_is_sanitized(self):
        failure = {
            "ok": False,
            "error": {
                "code": "model_loading_failed",
                "message": "secret /Users/private/checkpoint.pth traceback ...",
            },
        }
        response = _upload(create_app(ServiceSpy(failure)), "session.mat", b"mat-bytes")
        self.assertEqual(response["status"], 503)
        self.assertEqual(
            response["json"],
            {
                "ok": False,
                "error": {
                    "code": "model_unavailable",
                    "message": "The inference service is temporarily unavailable.",
                },
            },
        )
        response_text = response["body"].decode("utf-8")
        self.assertNotIn("checkpoint.pth", response_text)
        self.assertNotIn("traceback", response_text)
        self.assertNotIn("Users/private", response_text)

    def test_raised_inference_exception_is_sanitized(self):
        class RaisingService:
            def predict(self, _request):
                raise RuntimeError("secret /private/checkpoint.pth traceback")

        response = _upload(create_app(RaisingService()), "session.mat", b"mat-bytes")
        self.assertEqual(response["status"], 500)
        self.assertEqual(
            response["json"],
            {
                "ok": False,
                "error": {
                    "code": "inference_failed",
                    "message": "The prediction could not be completed.",
                },
            },
        )
        response_text = response["body"].decode("utf-8")
        self.assertNotIn("checkpoint.pth", response_text)
        self.assertNotIn("traceback", response_text)

    def test_validation_upload_preserves_frozen_outputs(self):
        reference_paths = sorted(
            STFT_REFERENCE_DIR.glob(f"{TRIAL_ID}_window_*_STFT_RGB.png")
        )
        self.assertEqual(len(reference_paths), 36)
        window_ids = [
            int(re.search(r"_window_(\d+)_", path.name).group(1))
            for path in reference_paths
        ]
        with VALIDATION_PREDICTIONS.open(newline="", encoding="utf-8") as handle:
            reference = next(
                row for row in csv.DictReader(handle) if row["trial_id"] == TRIAL_ID
            )

        def render_reference_windows(eeg, output_directory, recording_id):
            return render_stft_pngs(
                eeg,
                output_directory,
                recording_id=recording_id,
                window_ids=window_ids,
            )

        app = create_app(InferenceService(device="cpu"))
        created_temp_directories = []
        real_temporary_directory = tempfile.TemporaryDirectory

        def tracked_temporary_directory(*args, **kwargs):
            directory = real_temporary_directory(*args, **kwargs)
            created_temp_directories.append(Path(directory.name))
            return directory

        with mock.patch.object(
            tempfile,
            "TemporaryDirectory",
            side_effect=tracked_temporary_directory,
        ):
            with mock.patch.object(
                inference_module,
                "render_stft_pngs",
                side_effect=render_reference_windows,
            ):
                upload_bytes = RAW_SAMPLE.read_bytes()
                response = _upload(app, "validation.mat", upload_bytes)
                repeated_response = _upload(app, "validation.mat", upload_bytes)

        self.assertEqual(response["json"], repeated_response["json"])
        self.assertGreaterEqual(len(created_temp_directories), 4)
        for directory in created_temp_directories:
            self.assertFalse(directory.exists(), f"Temporary directory was not removed: {directory}")

        self.assertEqual(response["status"], 200)
        payload = response["json"]
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["prediction"], reference["adaptive_prediction"])
        self.assertTrue(
            math.isclose(
                payload["confidence"],
                float(reference["adaptive_confidence_score_not_calibrated"]),
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            )
        )
        expected_probabilities = json.loads(reference["adaptive_probabilities_json"])
        self.assertEqual(set(payload["class_probabilities"]), set(CLASS_NAMES))
        for class_name in CLASS_NAMES:
            self.assertTrue(
                math.isclose(
                    payload["class_probabilities"][class_name],
                    expected_probabilities[class_name],
                    rel_tol=0.0,
                    abs_tol=TOLERANCE,
                )
            )
        expected_weights = json.loads(reference["adaptive_trial_weights_json"])
        for model_name in MODEL_NAMES:
            self.assertEqual(
                payload["models"][model_name]["prediction"],
                reference[
                    f"{model_name.replace(' ', '_').replace('+', 'plus')}_prediction"
                ],
            )
            self.assertEqual(
                payload["models"][model_name]["window_count"],
                int(reference[
                    f"{model_name.replace(' ', '_').replace('+', 'plus')}_window_count"
                ]),
            )
            expected_model_probabilities = json.loads(
                reference[
                    f"{model_name.replace(' ', '_').replace('+', 'plus')}_probabilities_json"
                ]
            )
            for class_name in CLASS_NAMES:
                self.assertTrue(
                    math.isclose(
                        payload["models"][model_name]["class_probabilities"][class_name],
                        expected_model_probabilities[class_name],
                        rel_tol=0.0,
                        abs_tol=TOLERANCE,
                    )
                )
            self.assertTrue(
                math.isclose(
                    payload["adaptive_fusion"]["weights"][model_name],
                    expected_weights[model_name],
                    rel_tol=0.0,
                    abs_tol=TOLERANCE,
                )
            )
        self.assertEqual(payload["window_counts"]["stft"], 36)
        self.assertNotIn(str(PROJECT_ROOT), response["body"].decode("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
