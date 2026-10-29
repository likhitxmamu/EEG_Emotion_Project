"""Run validation-only four-model fusion, with test evaluation opt-in."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from fusion_engine import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    FUSION_WEIGHTS,
    MODEL_NAMES,
    PROJECT_ROOT,
    compute_metrics,
    expected_trials,
    fuse_trial_predictions,
    load_models,
    make_predictions_for_split,
    validate_static_test_contract,
)


FUSION_ROOT = PROJECT_ROOT / "models" / "fusion"


def device_for_inference():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    return value


def model_only_trial_records(model_outputs):
    records = {}
    for trial_key, item in model_outputs.items():
        records[trial_key] = {
            "true_class": item["true_class"],
            "final_emotion": item["predicted_class"],
        }
    return records


def split_metrics(model_outputs):
    return {
        model_name: compute_metrics(model_only_trial_records(model_outputs[model_name]))
        for model_name in MODEL_NAMES
    }


def safety_check(validation_outputs, test_manifest):
    if list(CLASS_NAMES) != ["fear", "happy", "neutral", "sad"]:
        raise RuntimeError(f"Unexpected fusion class order: {CLASS_NAMES}")
    expected_val = set(expected_trials("validation"))
    if expected_val != {"1_20160518_trial17", "1_20160518_trial20", "1_20160518_trial4", "1_20160518_trial12"}:
        raise RuntimeError("Validation trial assignment changed unexpectedly")
    if set(validation_outputs) != set(MODEL_NAMES):
        raise RuntimeError("One or more model branches did not produce validation outputs")
    for model_name, outputs in validation_outputs.items():
        if set(outputs) != expected_val:
            raise RuntimeError(f"Incomplete validation trial coverage for {model_name}")
        for trial_key, item in outputs.items():
            if item["true_class"] != expected_trials("validation")[trial_key]:
                raise RuntimeError(f"Validation label mismatch for {model_name}/{trial_key}")
            vector = np.asarray(item["probabilities"], dtype=np.float64)
            if vector.shape != (4,) or not np.isfinite(vector).all() or not np.isclose(vector.sum(), 1.0, atol=1e-5):
                raise RuntimeError(f"Invalid trial probability vector: {model_name}/{trial_key}")
    if abs(sum(FUSION_WEIGHTS.values()) - 1.0) > 1e-12:
        raise RuntimeError("Fusion weights must sum to one")
    if test_manifest.get("data_opened") or test_manifest.get("predictions_generated"):
        raise RuntimeError("Validation-only run unexpectedly opened or predicted test data")


def model_metadata(checkpoint_paths, configs):
    return {
        name: {
            "checkpoint_path": checkpoint_paths[name],
            **configs[name],
        }
        for name in MODEL_NAMES
    }


def markdown_report(results, report_path):
    lines = [
        "# Four-model EEG emotion fusion",
        "",
        f"Run: `{results['run_id']}`",
        f"Class order: `{results['class_order']}`",
        "Fusion: equal-weight arithmetic mean of each model's trial-averaged softmax probabilities.",
        "Final confidence is the maximum normalized fused score; it is **not calibrated probability or uncertainty**.",
        "",
        "## Models and checkpoints",
        "",
        "| Model | Checkpoint |",
        "|---|---|",
    ]
    for name in MODEL_NAMES:
        lines.append(f"| {name} | `{results['models'][name]['checkpoint_path']}` |")
    lines += ["", "## Equal fusion weights", ""]
    for name in MODEL_NAMES:
        lines.append(f"- {name}: {results['fusion_weights'][name]:.2f}")
    lines += ["", "## Validation fusion metrics", ""]
    lines.extend(metric_markdown_table(results["validation"]["metrics"]))
    lines += ["", "### Individual model validation metrics", ""]
    lines.append("| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in MODEL_NAMES:
        metrics = results["validation"]["individual_model_metrics"][name]
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['macro_precision']:.4f} | {metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} |"
        )
    lines += ["", "### Validation trial predictions", ""]
    lines.append("| Trial | Truth | EEGNet+GCN | Raw Transformer | STFT CRNN | STFT Transformer | Fusion | Confidence score* |")
    lines.append("|---|---|---|---|---|---|---|---:|")
    for trial_key, trial in results["validation"]["trials"].items():
        individual = trial["individual_models"]
        lines.append(
            f"| {trial_key} | {trial['true_class']} | "
            f"{individual['EEGNet + GCN']['predicted_class']} | "
            f"{individual['Raw EEG Transformer']['predicted_class']} | "
            f"{individual['STFT CRNN']['predicted_class']} | "
            f"{individual['STFT Transformer']['predicted_class']} | "
            f"{trial['final_emotion']} | {trial['model_confidence_score_not_calibrated']:.4f} |"
        )
    lines += ["", "*Confidence is the maximum fused score, not calibrated probability/uncertainty.", ""]
    if results.get("test", {}).get("status") == "not_evaluated":
        lines += [
            "## Held-out test",
            "",
            "Not evaluated in this validation-only run. Equal weights were fixed before test access.",
            "Static test assignment coverage was checked without opening test examples.",
            "",
        ]
    else:
        lines += ["## Held-out test metrics", ""]
        lines.extend(metric_markdown_table(results["test"]["metrics"]))
        lines += ["", "### Individual model test metrics", ""]
        lines.append("| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name in MODEL_NAMES:
            metrics = results["test"]["individual_model_metrics"][name]
            lines.append(
                f"| {name} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
                f"{metrics['macro_precision']:.4f} | {metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} |"
            )
        lines += ["", "### Test trial predictions", ""]
        lines.append("| Trial | Truth | EEGNet+GCN | Raw Transformer | STFT CRNN | STFT Transformer | Fusion | Confidence score* |")
        lines.append("|---|---|---|---|---|---|---|---:|")
        for trial_key, trial in results["test"]["trials"].items():
            individual = trial["individual_models"]
            lines.append(
                f"| {trial_key} | {trial['true_class']} | "
                f"{individual['EEGNet + GCN']['predicted_class']} | "
                f"{individual['Raw EEG Transformer']['predicted_class']} | "
                f"{individual['STFT CRNN']['predicted_class']} | "
                f"{individual['STFT Transformer']['predicted_class']} | "
                f"{trial['final_emotion']} | {trial['model_confidence_score_not_calibrated']:.4f} |"
            )
        lines += ["", "*Confidence is the maximum fused score, not calibrated probability/uncertainty.", ""]
    lines += [
        "## Leakage and compatibility checks",
        "",
    ]
    for check_name, check_value in results["checks"].items():
        lines.append(f"- {check_name}: `{check_value}`")
    lines += [
        "",
        "## Preprocessing configuration",
        "",
        "```json",
        json.dumps(results["preprocessing"], indent=2),
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def metric_markdown_table(metrics):
    lines = [
        f"- Accuracy: **{metrics['accuracy']:.4f}**",
        f"- Balanced accuracy: **{metrics['balanced_accuracy']:.4f}**",
        f"- Macro precision: **{metrics['macro_precision']:.4f}**",
        f"- Macro recall: **{metrics['macro_recall']:.4f}**",
        f"- Macro F1: **{metrics['macro_f1']:.4f}**",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for emotion in CLASS_NAMES:
        item = metrics["per_class"][emotion]
        lines.append(
            f"| {emotion} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']} |"
        )
    lines += ["", "Confusion matrix (rows=true, columns=predicted; class order fear, happy, neutral, sad):", "", "```text"]
    lines += [str(row) for row in metrics["confusion_matrix_rows_true_columns_predicted"]]
    lines += ["```", ""]
    return lines


def write_run_files(run_dir, results):
    result_path = run_dir / "results.json"
    report_path = run_dir / "fusion_report.md"
    result_path.write_text(json.dumps(to_json(results), indent=2), encoding="utf-8")
    markdown_report(results, report_path)
    return result_path, report_path


def run_validation(device):
    print("FUSION ENGINE — VALIDATION ONLY")
    print(f"Dataset (raw): {PROJECT_ROOT / 'data' / 'output'}")
    print(f"Dataset (STFT): {PROJECT_ROOT / 'data' / 'processed' / 'crnn_dataset'}")
    print(f"Class order: {CLASS_NAMES}")
    print(f"Device: {device}")
    print("Test files will not be opened in this run.")

    models, checkpoint_paths, configs, gcn_metadata, _ = load_models(device)
    test_manifest = validate_static_test_contract()
    model_outputs, runtime_preprocessing = make_predictions_for_split(
        "validation", models, device, gcn_metadata
    )
    safety_check(model_outputs, test_manifest)
    fused_trials = fuse_trial_predictions("validation", model_outputs)
    validation_metrics = compute_metrics(fused_trials)
    individual_metrics = split_metrics(model_outputs)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = FUSION_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results = {
        "schema_version": 1,
        "run_id": run_id,
        "class_order": CLASS_NAMES,
        "model_names": MODEL_NAMES,
        "models": model_metadata(checkpoint_paths, configs),
        "fusion_method": "arithmetic weighted average of each model's per-trial mean softmax probabilities",
        "fusion_weights": FUSION_WEIGHTS,
        "confidence_definition": "maximum component of normalized fused vector; not calibrated probability or uncertainty",
        "preprocessing": {
            "raw_models": {
                "common": "62-channel MAT EEG; 200 Hz project assumption; 2-second windows; 0.5-second hop",
                "EEGNet + GCN": {
                    "bandpass_hz": [1.0, 45.0],
                    "normalization": "saved per-channel mean/std fitted on training trials only",
                    "graph": "saved absolute channel-correlation graph computed from training trials only",
                },
                "Raw EEG Transformer": {
                    "bandpass_hz": [1.0, 45.0],
                    "normalization": "recomputed per-channel mean/std from training trials only",
                    **runtime_preprocessing["Raw EEG Transformer train-only normalization"],
                },
            },
            "STFT_models": {
                "dataset": str(PROJECT_ROOT / "data" / "processed" / "crnn_dataset"),
                "split": "existing trial-level split; no new split generated",
                "CRNN": "RGB, resize 224x224, ImageNet normalization",
                "Transformer": "grayscale, resize 224x224, mean/std 0.5 normalization; evaluation is deterministic",
                "excluded_datasets": [
                    "fixed-scale CRNN variants",
                    "global-scale CRNN variant",
                    "stft_transformer_dataset (random image-level split)",
                    "GASF-CNN",
                ],
            },
        },
        "validation": {
            "metrics": validation_metrics,
            "individual_model_metrics": individual_metrics,
            "trials": fused_trials,
        },
        "test": {
            "status": "not_evaluated",
            "reason": "Validation-only stage; equal weights finalized before any test inference.",
            "expected_trial_ids_static_check": test_manifest["expected_trial_ids"],
        },
        "checks": {
            "all_four_checkpoints_loaded_strictly": True,
            "common_class_order": CLASS_NAMES,
            "all_validation_trials_have_all_four_models": True,
            "probability_vectors_finite_and_normalized": True,
            "equal_weights_sum_to_one": True,
            "all_preprocessing_statistics_and_graph_use_training_only": True,
            "test_examples_opened_or_inferred": False,
            "test_used_for_fusion_configuration": False,
            "fixed_split_assignments_preserved": True,
        },
        "warnings": [
            "Raw MAT sampling rate is assumed to be 200 Hz; this is not stored in the MAT files.",
            "Each validation/test split contains one independent trial per class, so trial metrics are coarse.",
            "Model confidence is a normalized score, not calibrated probability or uncertainty.",
        ],
    }
    result_path, report_path = write_run_files(run_dir, results)
    print_metrics("VALIDATION FUSION", validation_metrics)
    print("\nIndividual validation trial accuracies:")
    for name in MODEL_NAMES:
        print(f"  {name:<24}: {individual_metrics[name]['accuracy']:.4f}")
    print(f"\nRun directory: {run_dir}")
    print(f"Results JSON: {result_path}")
    print(f"Human-readable report: {report_path}")
    print("Test data was not opened or inferred.")
    return run_dir


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(
        f"  accuracy={metrics['accuracy']:.4f}; balanced_accuracy={metrics['balanced_accuracy']:.4f}; "
        f"macro_precision={metrics['macro_precision']:.4f}; macro_recall={metrics['macro_recall']:.4f}; "
        f"macro_F1={metrics['macro_f1']:.4f}"
    )
    for emotion in CLASS_NAMES:
        item = metrics["per_class"][emotion]
        print(
            f"  {emotion:<8}: precision={item['precision']:.4f}, recall={item['recall']:.4f}, "
            f"F1={item['f1']:.4f}, support={item['support']}"
        )
    print("  confusion matrix:")
    for row in metrics["confusion_matrix_rows_true_columns_predicted"]:
        print(f"    {row}")


def run_test(device, run_dir):
    run_dir = Path(run_dir).resolve()
    result_path = run_dir / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Validation fusion results not found: {result_path}")
    results = json.loads(result_path.read_text(encoding="utf-8"))
    if results.get("test", {}).get("status") != "not_evaluated":
        raise RuntimeError("This run already has a test evaluation or is not a validation-only run; refusing another test pass")
    if results.get("fusion_weights") != FUSION_WEIGHTS:
        raise RuntimeError("Fusion weights changed since validation; refusing test evaluation")
    if results.get("class_order") != CLASS_NAMES:
        raise RuntimeError("Class order differs from validation configuration")

    guard_path = run_dir / "test_evaluation_guard.json"
    if guard_path.exists():
        raise RuntimeError(f"Test evaluation was already started for this run: {guard_path}")
    guard_path.write_text(
        json.dumps({"status": "started", "split": "test", "fusion_weights": FUSION_WEIGHTS}, indent=2),
        encoding="utf-8",
    )

    models, checkpoint_paths, configs, gcn_metadata, _ = load_models(device)
    if model_metadata(checkpoint_paths, configs) != results["models"]:
        raise RuntimeError("Checkpoint/configuration differs from the validation run")
    static_test_manifest = validate_static_test_contract()
    # The test split is opened only after the fixed validation configuration and guard are verified.
    model_outputs, runtime_preprocessing = make_predictions_for_split(
        "test", models, device, gcn_metadata
    )
    expected = set(expected_trials("test"))
    for model_name in MODEL_NAMES:
        if set(model_outputs[model_name]) != expected:
            raise RuntimeError(f"Test predictions incomplete for {model_name}")
    test_trials = fuse_trial_predictions("test", model_outputs)
    test_metrics = compute_metrics(test_trials)
    individual_metrics = split_metrics(model_outputs)

    results["test"] = {
        "status": "evaluated_once",
        "metrics": test_metrics,
        "individual_model_metrics": individual_metrics,
        "trials": test_trials,
        "coverage": static_test_manifest,
        "runtime_preprocessing": runtime_preprocessing,
    }
    results["checks"]["all_test_trials_have_all_four_models"] = True
    results["checks"]["test_predictions_finite_and_normalized"] = True
    results["checks"]["test_evaluated_after_fusion_configuration_finalized"] = True
    guard_path.write_text(
        json.dumps({"status": "completed", "split": "test", "fusion_weights": FUSION_WEIGHTS}, indent=2),
        encoding="utf-8",
    )
    result_path, report_path = write_run_files(run_dir, results)
    print_metrics("HELD-OUT TEST FUSION (evaluated once)", test_metrics)
    print("\nTest accuracy summary (trial-level; all models use the same four test trials):")
    for name in MODEL_NAMES:
        print(f"  {name:<24}: {individual_metrics[name]['accuracy']:.4f}")
    print(f"  {'Fusion':<24}: {test_metrics['accuracy']:.4f}")
    print(f"Updated results: {result_path}")
    print(f"Updated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validation-only", action="store_true", help="Run validation inference only (default).")
    mode.add_argument("--eval-test", action="store_true", help="Evaluate test exactly once for an existing validation run.")
    parser.add_argument("--run-dir", help="Existing validation run directory; required with --eval-test.")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = device_for_inference()
    else:
        device = torch.device(args.device)
        if args.device == "mps" and not torch.backends.mps.is_available():
            parser.error("MPS was requested but is not available")
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is not available")

    if args.eval_test:
        if not args.run_dir:
            parser.error("--run-dir is required with --eval-test")
        run_test(device, args.run_dir)
    else:
        if args.run_dir:
            parser.error("--run-dir is only used with --eval-test; validation creates a new timestamped run")
        run_validation(device)


if __name__ == "__main__":
    main()

