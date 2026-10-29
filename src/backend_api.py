"""Small WSGI HTTP wrapper around the frozen EEG inference service.

The API accepts one multipart MATLAB file upload and delegates all EEG
validation, STFT rendering, model inference, and fusion to InferenceService.
"""

import json
import logging
import os
import sys
import tempfile
from email.message import Message
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from pathlib import Path
from wsgiref.simple_server import make_server

_SOURCE_DIRECTORY = str(Path(__file__).resolve().parent)
if _SOURCE_DIRECTORY not in sys.path:
    sys.path.insert(0, _SOURCE_DIRECTORY)

from inference_service import CLASS_NAMES, InferenceService


LOGGER = logging.getLogger("eeg_emotion.api")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
LOCAL_FRONTEND_ORIGIN = "http://127.0.0.1:8080"


class APIRequestError(ValueError):
    """An expected request failure with a safe public error response."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = int(status)
        self.code = str(code)
        self.public_message = str(message)


_SERVICE_ERROR_RESPONSES = {
    "invalid_eeg": (
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "invalid_eeg",
        "The EEG recording must contain 62 finite channels and enough samples.",
    ),
    "malformed_mat": (
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "malformed_mat",
        "The uploaded file could not be read as a MATLAB recording.",
    ),
    "missing_eeg_variable": (
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "missing_eeg_variable",
        "The MATLAB file must contain an EEG variable.",
    ),
    "unsupported_format": (
        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        "unsupported_file_type",
        "Upload a MATLAB .mat file.",
    ),
    "input_file_error": (
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "invalid_mat",
        "The uploaded MATLAB recording could not be processed.",
    ),
    "missing_input": (
        HTTPStatus.BAD_REQUEST,
        "missing_file",
        "Upload a MATLAB file in the 'file' field.",
    ),
    "invalid_input": (
        HTTPStatus.BAD_REQUEST,
        "invalid_request",
        "The request could not be processed.",
    ),
    "missing_checkpoint": (
        HTTPStatus.SERVICE_UNAVAILABLE,
        "model_unavailable",
        "The inference service is temporarily unavailable.",
    ),
    "model_loading_failed": (
        HTTPStatus.SERVICE_UNAVAILABLE,
        "model_unavailable",
        "The inference service is temporarily unavailable.",
    ),
    "preprocessing_failed": (
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "inference_failed",
        "The recording could not be processed.",
    ),
    "stft_generation_failed": (
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "inference_failed",
        "The recording could not be processed.",
    ),
    "invalid_stft_image": (
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "inference_failed",
        "The recording could not be processed.",
    ),
    "inference_failed": (
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "inference_failed",
        "The prediction could not be completed.",
    ),
}


def _json_response(start_response, status, payload, extra_headers=()):
    body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
        ("X-Content-Type-Options", "nosniff"),
    ]
    headers.extend(extra_headers)
    start_response(f"{int(status)} {HTTPStatus(int(status)).phrase}", headers)
    return [body]


def _error_payload(code, message):
    return {"ok": False, "error": {"code": code, "message": message}}


def _read_body(environ, max_upload_bytes):
    content_length = environ.get("CONTENT_LENGTH", "")
    try:
        length = int(content_length)
    except (TypeError, ValueError) as exc:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST,
            "invalid_content_length",
            "The upload request is incomplete.",
        ) from exc
    if length <= 0:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST, "empty_request", "Upload a non-empty MATLAB file."
        )
    if length > max_upload_bytes:
        raise APIRequestError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "file_too_large",
            "The uploaded file exceeds the configured size limit.",
        )

    stream = environ.get("wsgi.input")
    if stream is None:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST,
            "invalid_request",
            "The upload request is incomplete.",
        )
    chunks = []
    remaining = length
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if remaining:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST,
            "incomplete_upload",
            "The upload request ended before the file was complete.",
        )
    return b"".join(chunks)


def _extract_mat_upload(environ, max_upload_bytes):
    content_type = environ.get("CONTENT_TYPE", "")
    if "\r" in content_type or "\n" in content_type:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST,
            "invalid_multipart",
            "The upload must use multipart form data.",
        )
    content_header = Message()
    content_header["Content-Type"] = content_type
    if content_header.get_content_type().lower() != "multipart/form-data":
        raise APIRequestError(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            "unsupported_media_type",
            "Upload the MATLAB file as multipart form data.",
        )
    boundary = content_header.get_param("boundary")
    if not isinstance(boundary, str) or not boundary or len(boundary) > 200:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST,
            "invalid_multipart",
            "The upload must use a valid multipart boundary.",
        )

    body = _read_body(environ, max_upload_bytes)
    try:
        headers = (
            "MIME-Version: 1.0\r\nContent-Type: " + content_type + "\r\n\r\n"
        ).encode("ascii")
        message = BytesParser(policy=email_policy).parsebytes(headers + body)
    except Exception as exc:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST,
            "invalid_multipart",
            "The multipart upload could not be read.",
        ) from exc
    if not message.is_multipart():
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST,
            "invalid_multipart",
            "The upload must contain a file part.",
        )

    uploads = []
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        if part.get_param("name", header="content-disposition") != "file":
            continue
        uploads.append((part.get_filename(), part.get_payload(decode=True)))
    if not uploads:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST,
            "missing_file",
            "Upload a MATLAB file in the 'file' field.",
        )
    if len(uploads) != 1:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST,
            "multiple_files",
            "Upload exactly one MATLAB file in the 'file' field.",
        )

    filename, file_bytes = uploads[0]
    if not filename:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST,
            "missing_file",
            "Upload a named MATLAB file in the 'file' field.",
        )
    if Path(filename).suffix.lower() != ".mat":
        raise APIRequestError(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            "unsupported_file_type",
            "Only MATLAB .mat files are accepted.",
        )
    if not file_bytes:
        raise APIRequestError(
            HTTPStatus.BAD_REQUEST, "empty_file", "The uploaded file is empty."
        )
    if len(file_bytes) > max_upload_bytes:
        raise APIRequestError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "file_too_large",
            "The uploaded file exceeds the configured size limit.",
        )
    return file_bytes


def _ordered_probability_map(value):
    if not isinstance(value, dict) or set(value) != set(CLASS_NAMES):
        raise ValueError("Inference service returned an invalid probability map")
    return {name: float(value[name]) for name in CLASS_NAMES}


def _public_model_results(value):
    if not isinstance(value, dict):
        raise ValueError("Inference service returned invalid model results")
    results = {}
    for model_name, item in value.items():
        if not isinstance(item, dict):
            raise ValueError("Inference service returned invalid model results")
        results[str(model_name)] = {
            "prediction": str(item["prediction"]),
            "confidence": float(item["confidence"]),
            "confidence_type": str(item.get("confidence_type", "not_calibrated")),
            "confidence_description": str(item.get("confidence_description", "")),
            "class_probabilities": _ordered_probability_map(item["probabilities"]),
            "window_count": int(item["window_count"]),
        }
    return results


def _public_fusion_information(value):
    if not isinstance(value, dict):
        raise ValueError("Inference service returned invalid fusion information")
    fields = (
        ("reliability_prior_weights", "reliability_prior_weights"),
        ("trial_weights", "weights"),
        ("normalized_entropy_by_model", "normalized_entropy_by_model"),
        ("certainty_gate_by_model", "certainty_gate_by_model"),
    )
    return {output_name: dict(value[input_name]) for input_name, output_name in fields}


def _public_success_response(result):
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise ValueError("Inference service returned an invalid success response")
    models = _public_model_results(result["models"])
    input_metadata = result.get("input", {})
    return {
        "ok": True,
        "prediction": str(result["prediction"]),
        "confidence": float(result["confidence"]),
        "adaptive_confidence_score": float(
            result.get("adaptive_confidence_score", result["confidence"])
        ),
        "confidence_type": str(result.get("confidence_type", "frozen_adaptive_confidence_score")),
        "confidence_description": str(result.get("confidence_description", "")),
        "class_order": list(CLASS_NAMES),
        "class_probabilities": _ordered_probability_map(result["probabilities"]),
        "models": models,
        "window_counts": {
            "by_model": {name: item["window_count"] for name, item in models.items()},
            "raw": int(input_metadata.get("raw_window_count", 0)),
            "stft": int(input_metadata.get("stft_window_count", 0)),
        },
        "adaptive_fusion": _public_fusion_information(result["adaptive_fusion"]),
    }


class EEGInferenceAPI:
    """WSGI application exposing ``GET /health`` and ``POST /predict``."""

    def __init__(self, inference_service=None, max_upload_bytes=MAX_UPLOAD_BYTES):
        if int(max_upload_bytes) <= 0:
            raise ValueError("max_upload_bytes must be positive")
        self.inference_service = (
            inference_service if inference_service is not None else InferenceService()
        )
        self.max_upload_bytes = int(max_upload_bytes)

    def __call__(self, environ, start_response):
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = environ.get("PATH_INFO", "/")
        if path == "/health":
            if method != "GET":
                return _json_response(
                    start_response,
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    _error_payload("method_not_allowed", "Use GET for this endpoint."),
                    [("Allow", "GET")],
                )
            # This route reports process availability only. It does not call
            # predict(), load checkpoints, or run model inference.
            return _json_response(
                start_response,
                HTTPStatus.OK,
                {"ok": True, "status": "ok"},
            )
        if path != "/predict":
            return _json_response(
                start_response,
                HTTPStatus.NOT_FOUND,
                _error_payload("not_found", "Endpoint not found."),
            )
        if method != "POST":
            return _json_response(
                start_response,
                HTTPStatus.METHOD_NOT_ALLOWED,
                _error_payload("method_not_allowed", "Use POST for this endpoint."),
                [("Allow", "POST")],
            )

        try:
            file_bytes = _extract_mat_upload(environ, self.max_upload_bytes)
            with tempfile.TemporaryDirectory(prefix="eeg_api_upload_") as temp_directory:
                mat_path = Path(temp_directory) / "upload.mat"
                mat_path.write_bytes(file_bytes)
                result = self.inference_service.predict({"mat_path": str(mat_path)})
            if not isinstance(result, dict) or result.get("ok") is not True:
                return self._service_error_response(start_response, result)
            payload = _public_success_response(result)
            return _json_response(start_response, HTTPStatus.OK, payload)
        except APIRequestError as exc:
            return _json_response(
                start_response,
                exc.status,
                _error_payload(exc.code, exc.public_message),
            )
        except Exception:
            # Internal exception details stay in server logs and never enter
            # the response body.
            LOGGER.exception("Prediction API request failed")
            return _json_response(
                start_response,
                HTTPStatus.INTERNAL_SERVER_ERROR,
                _error_payload("inference_failed", "The prediction could not be completed."),
            )

    @staticmethod
    def _service_error_response(start_response, result):
        error = result.get("error", {}) if isinstance(result, dict) else {}
        service_code = error.get("code") if isinstance(error, dict) else None
        status, public_code, message = _SERVICE_ERROR_RESPONSES.get(
            service_code,
            (
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "inference_failed",
                "The prediction could not be completed.",
            ),
        )
        LOGGER.warning("Prediction service returned public error code=%s", public_code)
        return _json_response(start_response, status, _error_payload(public_code, message))


class LocalFrontendCORSMiddleware:
    """Allow browser requests from the project's exact local frontend origin."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        origin = environ.get("HTTP_ORIGIN")
        if origin != LOCAL_FRONTEND_ORIGIN:
            return self.app(environ, start_response)

        path = environ.get("PATH_INFO", "/")
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        if method == "OPTIONS" and path in ("/health", "/predict"):
            allowed_methods = "GET, OPTIONS" if path == "/health" else "POST, OPTIONS"
            start_response(
                "204 No Content",
                [
                    ("Access-Control-Allow-Origin", origin),
                    ("Access-Control-Allow-Methods", allowed_methods),
                    ("Access-Control-Allow-Headers", "Content-Type"),
                    ("Access-Control-Max-Age", "600"),
                    ("Vary", "Origin"),
                    ("Content-Length", "0"),
                ],
            )
            return []

        def cors_start_response(status, headers, exc_info=None):
            response_headers = list(headers)
            response_headers.extend(
                (
                    ("Access-Control-Allow-Origin", origin),
                    ("Vary", "Origin"),
                )
            )
            return start_response(status, response_headers, exc_info)

        return self.app(environ, cors_start_response)


def create_app(inference_service=None, max_upload_bytes=MAX_UPLOAD_BYTES):
    """Create the WSGI app; the service argument supports isolated API tests."""
    app = EEGInferenceAPI(inference_service, max_upload_bytes=max_upload_bytes)
    return LocalFrontendCORSMiddleware(app)


def main():
    host = os.environ.get("EEG_API_HOST", DEFAULT_HOST)
    try:
        port = int(os.environ.get("EEG_API_PORT", str(DEFAULT_PORT)))
    except ValueError as exc:
        raise SystemExit("EEG_API_PORT must be an integer") from exc
    if not 0 < port < 65536:
        raise SystemExit("EEG_API_PORT must be between 1 and 65535")
    with make_server(host, port, application) as server:
        LOGGER.info("EEG inference API listening on http://%s:%d", host, port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            LOGGER.info("EEG inference API stopped")


application = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
