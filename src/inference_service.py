"""Reusable inference service for the frozen four-model EEG pipeline.

Requests provide either a numeric 62-channel EEG array or a MATLAB file with
an ``EEG`` variable. Frozen-format STFT windows are rendered automatically
unless matching PNGs are explicitly supplied for a reference-data regression.
"""

import json
import logging
import re
import sys
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

# Existing project modules import one another by their top-level names. Add
# src so this entry point works both as a script and as ``src.inference_service``.
_SOURCE_DIRECTORY = str(Path(__file__).resolve().parent)
if _SOURCE_DIRECTORY not in sys.path:
    sys.path.insert(0, _SOURCE_DIRECTORY)

from adaptive_fusion import FROZEN_RELIABILITY_PRIOR, fuse_adaptively
from frozen_stft_renderer import render_stft_pngs
from fusion_engine import (
    CLASS_NAMES,
    FROZEN_CONFIG,
    FROZEN_CONFIG_PATH,
    aggregate_model_trial,
    infer_raw_recording,
    infer_stft_image_paths,
    load_models,
)


LOGGER = logging.getLogger("eeg_emotion.inference")
_STFT_NAME_RE = re.compile(
    r"^(?P<trial>.+)_window_(?P<window>\d+)_STFT_RGB$", re.IGNORECASE
)
_RAW_CONFIG = FROZEN_CONFIG["common_raw_eeg_preprocessing"]
_EXPECTED_CHANNELS = int(_RAW_CONFIG["channels"])
_WINDOW_SAMPLES = int(_RAW_CONFIG["windowing"]["window_samples"])
_HOP_SAMPLES = int(_RAW_CONFIG["windowing"]["hop_samples"])
_SAMPLING_RATE_HZ = int(_RAW_CONFIG["sampling_rate_hz"])
_STFT_IMAGE_CONFIG = FROZEN_CONFIG["input_assets"]["stft_images"]
_STFT_IMAGE_DIMS = tuple(int(value) for value in _STFT_IMAGE_CONFIG["image_dimensions_px"])
_STFT_IMAGE_MODE = str(_STFT_IMAGE_CONFIG["image_mode"])


class UnsupportedFormatError(ValueError):
    """Raised for an input file type outside the service contract."""


class MissingInferenceInputError(ValueError):
    """Raised when an essential input is missing."""


class InvalidEEGError(ValueError):
    """Raised when numeric EEG data violates the frozen input contract."""


class MalformedMATFileError(ValueError):
    """Raised when a MATLAB file cannot be read."""


class MissingEEGVariableError(ValueError):
    """Raised when a readable MATLAB file does not contain ``EEG``."""


class InvalidSTFTImageError(ValueError):
    """Raised when an STFT raster does not match the frozen image contract."""


class STFTGenerationError(RuntimeError):
    """Raised when frozen-format STFT rasters cannot be generated."""


class MissingModelCheckpointError(FileNotFoundError):
    """Raised when a frozen model checkpoint is absent."""


class ModelLoadingError(RuntimeError):
    """Raised when frozen configuration or model loading fails."""


class PreprocessingError(RuntimeError):
    """Raised when frozen raw EEG preprocessing cannot be completed."""


class InferenceExecutionError(RuntimeError):
    """Raised when a loaded model cannot produce valid inference output."""


def _error(code, message, recording_id=None):
    return {
        "ok": False,
        "recording_id": recording_id,
        "error": {"code": code, "message": str(message)},
    }


def _device_from_frozen_policy():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _probability_map(values):
    return {name: float(values[index]) for index, name in enumerate(CLASS_NAMES)}


def _coerce_eeg(values):
    try:
        eeg = np.asarray(values)
    except Exception as exc:
        raise InvalidEEGError(f"EEG input cannot be converted to an array: {exc}") from exc
    if eeg.dtype.kind not in "iuf" or np.iscomplexobj(eeg):
        raise InvalidEEGError("EEG input must contain real numeric values")
    if eeg.ndim != 2 or eeg.shape[0] != _EXPECTED_CHANNELS:
        raise InvalidEEGError(
            f"Expected EEG shape ({_EXPECTED_CHANNELS}, samples); received {eeg.shape}"
        )
    if eeg.shape[1] < _WINDOW_SAMPLES:
        raise InvalidEEGError(
            f"EEG recording must contain at least {_WINDOW_SAMPLES} samples; received {eeg.shape[1]}"
        )
    if not np.isfinite(eeg).all():
        raise InvalidEEGError("EEG input contains NaN or infinite values")
    return eeg


def _load_eeg_input(request):
    has_array = "eeg" in request and request["eeg"] is not None
    has_mat_path = "mat_path" in request and request["mat_path"] is not None
    if not has_array and not has_mat_path:
        raise MissingInferenceInputError("Provide either an 'eeg' array or a 'mat_path'")
    if has_array == has_mat_path:
        raise ValueError("Provide exactly one of 'eeg' or 'mat_path'")
    if has_array:
        return _coerce_eeg(request["eeg"]), "array", None

    raw_path = request["mat_path"]
    if not isinstance(raw_path, (str, Path)):
        raise ValueError("'mat_path' must be a filesystem path")
    path = Path(raw_path).expanduser()
    if path.suffix.lower() != ".mat":
        raise UnsupportedFormatError("Raw recording files must use the MATLAB .mat format")
    if not path.is_file():
        raise FileNotFoundError(f"EEG recording file not found: {path}")
    try:
        mat = loadmat(path)
    except Exception as exc:
        raise MalformedMATFileError(
            f"Cannot read MATLAB EEG recording {path.name}: {exc}"
        ) from exc
    if "EEG" not in mat:
        raise MissingEEGVariableError(
            f"MAT file does not contain the required 'EEG' variable: {path.name}"
        )
    # Labels are never required or read by inference.
    return _coerce_eeg(mat["EEG"]), "mat", path


def _load_stft_paths(provided, expected_recording_id=None):
    if isinstance(provided, (str, Path)) or provided is None:
        raise ValueError("'stft_image_paths' must be a sequence of PNG paths")
    try:
        paths = [Path(value).expanduser() for value in provided]
    except (TypeError, ValueError) as exc:
        raise ValueError("'stft_image_paths' must be a sequence of PNG paths") from exc
    if not paths:
        raise MissingInferenceInputError("At least one STFT PNG must be available")

    for path in paths:
        if path.suffix.lower() != ".png":
            raise UnsupportedFormatError(f"STFT inputs must be PNG files: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(f"STFT image not found: {path}")
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise InvalidSTFTImageError(f"STFT image is not a PNG file: {path.name}")
                if image.size != _STFT_IMAGE_DIMS:
                    raise InvalidSTFTImageError(
                        f"Frozen STFT inputs must be {_STFT_IMAGE_DIMS[0]}x{_STFT_IMAGE_DIMS[1]}; "
                        f"{path.name} is {image.size}"
                    )
                if image.mode != _STFT_IMAGE_MODE:
                    raise InvalidSTFTImageError(
                        f"Frozen STFT inputs must use {_STFT_IMAGE_MODE} mode; "
                        f"{path.name} is {image.mode}"
                    )
                image.verify()
        except InvalidSTFTImageError:
            raise
        except Exception as exc:
            raise InvalidSTFTImageError(
                f"Cannot read STFT PNG {path.name}: {exc}"
            ) from exc

    # Preserve numeric filename order independently of the caller's order.
    parsed = []
    for path in paths:
        match = _STFT_NAME_RE.fullmatch(path.stem)
        parsed.append(
            (match.group("trial"), int(match.group("window"))) if match else None
        )
    if any(value is not None for value in parsed) and not all(value is not None for value in parsed):
        raise ValueError(
            "STFT image names must either all use the recording/window naming pattern or none do"
        )
    if all(value is not None for value in parsed):
        trial_ids = {value[0] for value in parsed}
        if len(trial_ids) != 1:
            raise ValueError("STFT image paths contain windows from more than one recording")
        image_trial_id = next(iter(trial_ids))
        if expected_recording_id is not None and image_trial_id != expected_recording_id:
            raise ValueError(
                f"STFT images belong to {image_trial_id}, but the MAT recording is "
                f"{expected_recording_id}"
            )
        parsed_window_ids = [value[1] for value in parsed]
        if len(set(parsed_window_ids)) != len(parsed_window_ids):
            raise ValueError("STFT image paths contain duplicate window IDs")
        ordered = sorted(zip(paths, parsed_window_ids), key=lambda pair: pair[1])
        paths = [pair[0] for pair in ordered]
        window_ids = [pair[1] for pair in ordered]
    else:
        window_ids = list(range(len(paths)))
    return paths, window_ids


class InferenceService:
    """Load and run the frozen adaptive-fusion inference pipeline.

    ``predict`` accepts either ``{"eeg": array}`` or ``{"mat_path": path}``.
    The numeric EEG shape is 62 channels by samples. The service generates all
    complete frozen STFT windows automatically. ``stft_image_paths`` remains
    an optional reference-data override for the frozen service regression.
    """

    def __init__(self, device=None, logger=None):
        self.logger = logger or LOGGER
        self.device = torch.device(device) if device is not None else _device_from_frozen_policy()
        self.models = None
        self.gcn_metadata = None
        self._load_error = None
        self._load_lock = threading.Lock()
        self.configuration_id = None
        try:
            config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
            if config.get("status") != "frozen":
                raise RuntimeError("Frozen ML configuration is not marked frozen")
            if config != FROZEN_CONFIG:
                raise RuntimeError(
                    "Frozen ML configuration changed after the inference modules were loaded"
                )
            if config.get("class_order") != CLASS_NAMES:
                raise RuntimeError("Frozen configuration class order differs from the inference engine")
            self.configuration_id = config["configuration_id"]
        except Exception as exc:
            self._load_error = exc

    def _ensure_models_loaded(self):
        if self._load_error is not None:
            raise ModelLoadingError(f"Frozen configuration/model loading failed: {self._load_error}")
        if self.models is None:
            with self._load_lock:
                if self.models is None:
                    try:
                        loaded, checkpoint_paths, _, metadata, _ = load_models(self.device)
                    except FileNotFoundError as exc:
                        if "checkpoint" in str(exc).lower():
                            raise MissingModelCheckpointError(str(exc)) from exc
                        raise ModelLoadingError(str(exc)) from exc
                    except Exception as exc:
                        raise ModelLoadingError(str(exc)) from exc
                    self.models = loaded
                    self.gcn_metadata = metadata
                    checkpoint_summary = ", ".join(
                        f"{name}={Path(path).name}" for name, path in checkpoint_paths.items()
                    )
                    self.logger.info(
                        "Frozen model checkpoints loaded on %s: %s",
                        self.device,
                        checkpoint_summary,
                    )
        return self.models

    def _predict_with_stft_paths(
        self, eeg, recording_id, input_name, source_path, stft_paths, stft_window_ids, stft_source
    ):
        models = self._ensure_models_loaded()
        self.logger.info(
            "Inference started for recording_id=%s input_type=%s shape=%s STFT windows=%d",
            recording_id,
            input_name,
            eeg.shape,
            len(stft_paths),
        )

        raw_outputs = {}
        for model_name in ("EEGNet + GCN", "Raw EEG Transformer"):
            try:
                output, _ = infer_raw_recording(
                    eeg,
                    model_name,
                    models[model_name],
                    self.device,
                    self.gcn_metadata,
                    recording_id=recording_id,
                )
            except Exception as exc:
                raise PreprocessingError(
                    f"{model_name} preprocessing/inference failed: {exc}"
                ) from exc
            raw_outputs[model_name] = {recording_id: output}
            self.logger.info(
                "Raw preprocessing/inference completed for recording_id=%s model=%s windows=%d",
                recording_id,
                model_name,
                output["window_count"],
            )

        stft_outputs = {}
        for model_name in ("STFT CRNN", "STFT Transformer"):
            try:
                window_outputs = infer_stft_image_paths(
                    stft_paths,
                    model_name,
                    models[model_name],
                    self.device,
                    window_ids=stft_window_ids,
                )
                item = {"windows": window_outputs, "window_count": len(window_outputs)}
                stft_outputs[model_name] = {
                    recording_id: aggregate_model_trial(recording_id, item, true_class=None)
                }
            except Exception as exc:
                raise InferenceExecutionError(f"{model_name} inference failed: {exc}") from exc

        model_outputs = {**raw_outputs, **stft_outputs}
        fused = fuse_adaptively(
            "inference", model_outputs, FROZEN_RELIABILITY_PRIOR
        )[recording_id]
        response = self._format_response(
            recording_id,
            eeg.shape[1],
            fused,
            input_name=input_name,
            source_path=source_path,
            stft_source=stft_source,
            stft_window_ids=stft_window_ids,
        )
        self.logger.info(
            "Inference completed for recording_id=%s prediction=%s",
            recording_id,
            response["prediction"],
        )
        return response

    def predict(self, request):
        """Run one recording and return a prediction or a structured error."""
        if request is None:
            return _error("missing_input", "Provide an input mapping with EEG data")
        if not isinstance(request, Mapping):
            return _error("invalid_input", "Inference request must be a mapping")

        supplied_id = request.get("recording_id")
        recording_id = str(supplied_id)[:160] if supplied_id is not None else None
        try:
            eeg, input_name, source_path = _load_eeg_input(request)
            if recording_id is None:
                recording_id = source_path.stem if source_path else "in_memory_recording"
            if not recording_id.strip():
                raise ValueError("'recording_id' must not be empty")

            supplied_stft_paths = request.get("stft_image_paths")
            if supplied_stft_paths is not None:
                stft_paths, stft_window_ids = _load_stft_paths(
                    supplied_stft_paths,
                    expected_recording_id=source_path.stem if source_path else None,
                )
                return self._predict_with_stft_paths(
                    eeg,
                    recording_id,
                    input_name,
                    source_path,
                    stft_paths,
                    stft_window_ids,
                    stft_source="provided_reference_images",
                )

            # Keep the generated PNGs alive through both image models, then
            # remove them regardless of whether inference succeeds.
            with tempfile.TemporaryDirectory(prefix="eeg_emotion_stft_") as output_directory:
                try:
                    generated_paths = render_stft_pngs(
                        eeg,
                        output_directory,
                        recording_id=recording_id,
                    )
                except Exception as exc:
                    raise STFTGenerationError(
                        f"Frozen STFT generation failed: {exc}"
                    ) from exc
                stft_paths, stft_window_ids = _load_stft_paths(generated_paths)
                return self._predict_with_stft_paths(
                    eeg,
                    recording_id,
                    input_name,
                    source_path,
                    stft_paths,
                    stft_window_ids,
                    stft_source="frozen_stft_renderer",
                )
        except MissingModelCheckpointError as exc:
            self.logger.error("Frozen checkpoint missing for recording_id=%s: %s", recording_id, exc)
            return _error("missing_checkpoint", exc, recording_id)
        except ModelLoadingError as exc:
            self.logger.exception("Frozen model loading failed for recording_id=%s", recording_id)
            return _error("model_loading_failed", exc, recording_id)
        except UnsupportedFormatError as exc:
            return _error("unsupported_format", exc, recording_id)
        except MissingInferenceInputError as exc:
            return _error("missing_input", exc, recording_id)
        except InvalidEEGError as exc:
            return _error("invalid_eeg", exc, recording_id)
        except MalformedMATFileError as exc:
            return _error("malformed_mat", exc, recording_id)
        except MissingEEGVariableError as exc:
            return _error("missing_eeg_variable", exc, recording_id)
        except InvalidSTFTImageError as exc:
            return _error("invalid_stft_image", exc, recording_id)
        except STFTGenerationError as exc:
            return _error("stft_generation_failed", exc, recording_id)
        except PreprocessingError as exc:
            self.logger.exception("Preprocessing failed for recording_id=%s", recording_id)
            return _error("preprocessing_failed", exc, recording_id)
        except InferenceExecutionError as exc:
            self.logger.exception("Inference failed for recording_id=%s", recording_id)
            return _error("inference_failed", exc, recording_id)
        except FileNotFoundError as exc:
            return _error("input_file_error", exc, recording_id)
        except (ValueError, KeyError, TypeError) as exc:
            return _error("invalid_input", exc, recording_id)
        except Exception as exc:
            self.logger.exception("Unexpected inference failure for recording_id=%s", recording_id)
            code = "model_loading_failed" if self.models is None else "inference_failed"
            return _error(code, exc, recording_id)

    @staticmethod
    def _format_response(
        recording_id,
        sample_count,
        fused,
        input_name,
        source_path,
        stft_source,
        stft_window_ids,
    ):
        model_results = {}
        for model_name, item in fused["individual_models"].items():
            model_results[model_name] = {
                "prediction": item["predicted_class"],
                "confidence": float(item["model_confidence_score_not_calibrated"]),
                "confidence_type": "not_calibrated",
                "confidence_description": (
                    "Maximum class value in this model's trial probability vector; not calibrated."
                ),
                "probabilities": _probability_map(item["probabilities"]),
                "window_count": int(item["window_count"]),
                "window_outputs": [
                    {
                        "window_id": int(window["window_id"]),
                        "logits": [float(value) for value in window["logits"]],
                        "probabilities": _probability_map(window["probabilities"]),
                    }
                    for window in item["window_outputs"]
                ],
            }

        adaptive_confidence = float(fused["model_confidence_score_not_calibrated"])
        return {
            "ok": True,
            "configuration_id": FROZEN_CONFIG["configuration_id"],
            "recording_id": recording_id,
            "class_order": list(CLASS_NAMES),
            "prediction": fused["final_emotion"],
            "confidence": adaptive_confidence,
            "adaptive_confidence_score": adaptive_confidence,
            "confidence_type": "frozen_adaptive_confidence_score",
            "confidence_description": (
                "Frozen adaptive confidence score (the winning fused probability value); "
                "this is not a calibrated probability."
            ),
            "probabilities": _probability_map(fused["fused_probabilities"]),
            "models": model_results,
            "adaptive_fusion": {
                "method": FROZEN_CONFIG["fusion"]["adaptive"]["method"],
                "reliability_prior_weights": dict(fused["reliability_prior_weights"]),
                "trial_weights": dict(fused["trial_adaptive_weights"]),
                "normalized_entropy_by_model": dict(fused["normalized_entropy_by_model"]),
                "certainty_gate_by_model": dict(fused["certainty_gate_by_model"]),
            },
            "input": {
                "input_type": input_name,
                "source_file": source_path.name if source_path else None,
                "eeg_channels": _EXPECTED_CHANNELS,
                "eeg_samples": int(sample_count),
                "sampling_rate_hz_assumption": _SAMPLING_RATE_HZ,
                "raw_window_count": int(
                    (sample_count - _WINDOW_SAMPLES) // _HOP_SAMPLES + 1
                ),
                "stft_window_count": len(stft_window_ids),
                "stft_window_ids": [int(value) for value in stft_window_ids],
                "stft_window_source": stft_source,
                "stft_image_dimensions_px": list(_STFT_IMAGE_DIMS),
                "stft_image_mode": _STFT_IMAGE_MODE,
            },
        }


def predict(request, device=None):
    """Convenience function for a one-off inference call."""
    return InferenceService(device=device).predict(request)
