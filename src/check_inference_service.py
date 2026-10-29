"""Regression check for the public inference service using one frozen sample."""

import argparse
import csv
import json
import logging
import math
import re
from pathlib import Path
from unittest.mock import patch

import inference_service as inference_module
from inference_service import InferenceService
from frozen_stft_renderer import render_stft_pngs
from fusion_engine import PROJECT_ROOT


TRIAL_ID = "1_20160518_trial17"
REFERENCE_CSV = (
    PROJECT_ROOT
    / "models/fusion/adaptive_run_20261027_132228/validation_predictions.csv"
)
RAW_SAMPLE = PROJECT_ROOT / "data/output/fear" / f"{TRIAL_ID}.mat"
STFT_SAMPLE_DIR = PROJECT_ROOT / "data/processed/crnn_dataset/validation/fear"
TOLERANCE = 1e-5


def _close(actual, expected, label, tolerance=TOLERANCE):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: {actual} != {expected} (tolerance {tolerance})")


def _compare_probability_maps(actual, expected, label):
    if set(actual) != set(expected):
        raise AssertionError(f"{label}: class keys differ")
    for class_name in expected:
        _close(actual[class_name], expected[class_name], f"{label}.{class_name}")


def _load_reference():
    with REFERENCE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["trial_id"] == TRIAL_ID:
                return row
    raise FileNotFoundError(f"Frozen validation result for {TRIAL_ID} was not found in {REFERENCE_CSV}")


def run_check():
    stft_paths = sorted(STFT_SAMPLE_DIR.glob(f"{TRIAL_ID}_window_*_STFT_RGB.png"))
    if not RAW_SAMPLE.is_file():
        raise FileNotFoundError(f"Frozen raw sample is missing: {RAW_SAMPLE}")
    if not stft_paths:
        raise FileNotFoundError(f"Frozen STFT windows are missing for {TRIAL_ID}")
    window_ids = [
        int(re.search(r"_window_(\d+)_", path.name).group(1))
        for path in stft_paths
    ]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    service = InferenceService()
    # Re-render the exact frozen validation window subset through the service's
    # automatic path. The stored filenames supply IDs only; callers still pass
    # EEG data without manually creating or supplying PNGs.
    def render_reference_windows(eeg, output_directory, recording_id):
        return render_stft_pngs(
            eeg,
            output_directory,
            recording_id=recording_id,
            window_ids=window_ids,
        )

    with patch.object(
        inference_module,
        "render_stft_pngs",
        side_effect=render_reference_windows,
    ):
        result = service.predict({
            "recording_id": TRIAL_ID,
            "mat_path": str(RAW_SAMPLE),
        })
    if not result.get("ok"):
        raise RuntimeError(f"Inference service returned an error: {json.dumps(result, sort_keys=True)}")
    if result["input"]["stft_window_source"] != "frozen_stft_renderer":
        raise AssertionError("Inference service did not use the frozen renderer automatically")
    if result["input"]["stft_window_count"] != len(stft_paths):
        raise AssertionError("Inference service rendered a different validation window count")

    reference = _load_reference()
    if result["prediction"] != reference["adaptive_prediction"]:
        raise AssertionError(
            f"Adaptive prediction differs: {result['prediction']} != {reference['adaptive_prediction']}"
        )
    _close(
        result["confidence"],
        reference["adaptive_confidence_score_not_calibrated"],
        "adaptive confidence",
    )
    _compare_probability_maps(
        result["probabilities"],
        json.loads(reference["adaptive_probabilities_json"]),
        "adaptive probabilities",
    )
    _compare_probability_maps(
        result["adaptive_fusion"]["trial_weights"],
        json.loads(reference["adaptive_trial_weights_json"]),
        "adaptive trial weights",
    )

    model_column_prefixes = {
        "EEGNet + GCN": "EEGNet_plus_GCN",
        "Raw EEG Transformer": "Raw_EEG_Transformer",
        "STFT CRNN": "STFT_CRNN",
        "STFT Transformer": "STFT_Transformer",
    }
    for model_name, prefix in model_column_prefixes.items():
        model_result = result["models"][model_name]
        if model_result["prediction"] != reference[f"{prefix}_prediction"]:
            raise AssertionError(f"{model_name} prediction differs from frozen validation output")
        if model_result["window_count"] != int(reference[f"{prefix}_window_count"]):
            raise AssertionError(f"{model_name} window count differs from frozen validation output")
        _compare_probability_maps(
            model_result["probabilities"],
            json.loads(reference[f"{prefix}_probabilities_json"]),
            f"{model_name} probabilities",
        )

    print(
        "PASS: inference service matches the frozen validation result for "
        f"{TRIAL_ID} (class exact; probabilities, confidence, and weights within {TOLERANCE:g})."
    )
    print(f"Prediction: {result['prediction']} ({result['confidence']:.10f}, not calibrated)")
    print(f"Reference: {REFERENCE_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_check()

