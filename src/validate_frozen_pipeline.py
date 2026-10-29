"""Compare frozen-config inference with the saved adaptive-fusion run.

This read-only sanity check does not write evaluation artifacts. The held-out
test split is used only for the requested prediction-equivalence check; its
labels and metrics are never used to derive or change the frozen weights.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from adaptive_fusion import (
    FROZEN_RELIABILITY_PRIOR,
    fuse_adaptively,
    equal_weight_fuse,
    validation_reliability_prior,
)
from evaluate_adaptive_fusion import (
    combine_trial_outputs,
    infer_split,
    individual_metrics,
    rows_for_csv,
)
from fusion_engine import (
    CLASS_NAMES,
    FROZEN_CONFIG,
    MODEL_NAMES,
    compute_metrics,
    load_models,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = PROJECT_ROOT / FROZEN_CONFIG["source_run"]
PROBABILITY_TOLERANCE = 1e-5


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["trial_id"]: row for row in csv.DictReader(stream)}


def _assert_close_mapping(actual_json, expected_json, label, keys=None):
    actual = json.loads(actual_json)
    expected = json.loads(expected_json)
    if set(actual) != set(expected):
        raise AssertionError(f"{label}: key mismatch {set(actual)} != {set(expected)}")
    ordered_keys = keys or CLASS_NAMES
    actual_vector = np.asarray([actual[name] for name in ordered_keys], dtype=np.float64)
    expected_vector = np.asarray([expected[name] for name in ordered_keys], dtype=np.float64)
    error = float(np.max(np.abs(actual_vector - expected_vector)))
    if error > PROBABILITY_TOLERANCE:
        raise AssertionError(f"{label}: max absolute difference {error:.3g} exceeds tolerance")
    return error


def _compare_split(split, predictions, metrics, stored_results):
    expected_csv_path = REFERENCE_RUN / f"{split}_predictions.csv"
    expected_rows = _read_csv(expected_csv_path)
    actual_rows = {row["trial_id"]: row for row in rows_for_csv(predictions)}
    if set(actual_rows) != set(expected_rows):
        raise AssertionError(f"{split}: trial IDs differ from the saved evaluation")

    class_fields = ["adaptive_prediction", "equal_prediction"]
    probability_fields = [
        "adaptive_probabilities_json",
        "equal_probabilities_json",
        *[
            f"{name.replace(' ', '_').replace('+', 'plus')}_probabilities_json"
            for name in MODEL_NAMES
        ],
    ]
    max_probability_error = 0.0
    for trial_id, actual in actual_rows.items():
        expected = expected_rows[trial_id]
        for field in class_fields:
            if actual[field] != expected[field]:
                raise AssertionError(f"{split} {trial_id}: {field} differs")
        for name in MODEL_NAMES:
            prefix = name.replace(" ", "_").replace("+", "plus")
            prediction_field = f"{prefix}_prediction"
            count_field = f"{prefix}_window_count"
            if actual[prediction_field] != expected[prediction_field]:
                raise AssertionError(f"{split} {trial_id}: {prediction_field} differs")
            if str(actual[count_field]) != str(expected[count_field]):
                raise AssertionError(f"{split} {trial_id}: {count_field} differs")
        for field in probability_fields:
            error = _assert_close_mapping(actual[field], expected[field], f"{split} {trial_id} {field}")
            max_probability_error = max(max_probability_error, error)
        error = _assert_close_mapping(
            actual["adaptive_trial_weights_json"],
            expected["adaptive_trial_weights_json"],
            f"{split} {trial_id} adaptive weights",
            keys=MODEL_NAMES,
        )
        max_probability_error = max(max_probability_error, error)

    for metric_key, metrics_key in (
        ("equal_weight_metrics", "equal_weight_metrics"),
        ("adaptive_metrics", "adaptive_metrics"),
    ):
        expected_metrics = stored_results[split][metric_key]
        actual_metrics = metrics[metrics_key]
        for name in ("accuracy", "macro_f1"):
            if not np.isclose(actual_metrics[name], expected_metrics[name], atol=1e-12, rtol=0.0):
                raise AssertionError(f"{split}: {metric_key}.{name} differs")
    return len(actual_rows), max_probability_error


def main():
    result_path = REFERENCE_RUN / "results.json"
    saved_results = json.loads(result_path.read_text(encoding="utf-8"))
    device = torch.device("cpu")
    models, checkpoints, _, graph_metadata, _ = load_models(device)
    for name in MODEL_NAMES:
        frozen_path = str((PROJECT_ROOT / FROZEN_CONFIG["models"][name]["checkpoint"]["path"]).resolve())
        if str(Path(checkpoints[name]).resolve()) != frozen_path:
            raise AssertionError(f"Loaded checkpoint path differs from frozen configuration for {name}")

    validation_outputs, _ = infer_split("validation", models, device, graph_metadata)
    reliability_prior, reliability_scores = validation_reliability_prior(validation_outputs)
    if reliability_scores != FROZEN_CONFIG["fusion"]["adaptive"]["validation_reliability_macro_f1"]:
        raise AssertionError("Validation reliability scores differ from the frozen configuration")
    if reliability_prior != FROZEN_RELIABILITY_PRIOR:
        raise AssertionError("Validation reliability weights differ from the frozen configuration")

    for split in ("validation", "test"):
        outputs = validation_outputs if split == "validation" else infer_split(
            split, models, device, graph_metadata
        )[0]
        equal_trials = equal_weight_fuse(split, outputs)
        adaptive_trials = fuse_adaptively(split, outputs, FROZEN_RELIABILITY_PRIOR)
        predictions = combine_trial_outputs(equal_trials, adaptive_trials)
        metrics = {
            "equal_weight_metrics": compute_metrics(equal_trials),
            "adaptive_metrics": compute_metrics(adaptive_trials),
        }
        count, max_error = _compare_split(split, predictions, metrics, saved_results)
        print(
            f"{split}: PASS — {count}/{count} trial labels, all four base-model labels, "
            f"window counts, and fused probabilities match; max probability/weight error "
            f"{max_error:.3g}; adaptive accuracy={metrics['adaptive_metrics']['accuracy']:.4f}, "
            f"macro-F1={metrics['adaptive_metrics']['macro_f1']:.4f}"
        )

    print("Frozen configuration sanity check passed. No files were written and no parameters were fit from test results.")


if __name__ == "__main__":
    main()
