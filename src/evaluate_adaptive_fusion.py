"""Validation-first adaptive fusion evaluation; test requires explicit opt-in."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from adaptive_fusion import EPSILON, equal_weight_fuse, fuse_adaptively, validation_reliability_prior
from fusion_engine import (
    CLASS_NAMES,
    FROZEN_CONFIG,
    FROZEN_CONFIG_PATH,
    FUSION_WEIGHTS,
    MODEL_NAMES,
    PROJECT_ROOT,
    aggregate_model_outputs,
    compute_metrics,
    expected_trials,
    infer_raw_branch,
    infer_stft_branch,
    load_models,
    validate_static_test_contract,
)


OUTPUT_ROOT = PROJECT_ROOT / "models" / "fusion"


def device_for_inference():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_split(split, models, device, gcn_metadata):
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unsupported split: {split}")
    raw_gcn, _ = infer_raw_branch(split, "EEGNet + GCN", models["EEGNet + GCN"], device, gcn_metadata)
    raw_transformer, raw_norm = infer_raw_branch(
        split, "Raw EEG Transformer", models["Raw EEG Transformer"], device
    )
    stft_crnn = infer_stft_branch(split, "STFT CRNN", models["STFT CRNN"], device)
    stft_transformer = infer_stft_branch(split, "STFT Transformer", models["STFT Transformer"], device)
    unaggregated = {
        "EEGNet + GCN": raw_gcn,
        "Raw EEG Transformer": raw_transformer,
        "STFT CRNN": stft_crnn,
        "STFT Transformer": stft_transformer,
    }
    aggregated = {name: aggregate_model_outputs(split, unaggregated[name]) for name in MODEL_NAMES}
    expected = expected_trials(split)
    for name in MODEL_NAMES:
        if set(aggregated[name]) != set(expected):
            raise RuntimeError(f"{name} has incomplete {split} trial coverage")
        for trial_id, trial in aggregated[name].items():
            if trial["true_class"] != expected[trial_id]:
                raise RuntimeError(f"{name} label mismatch for {trial_id}")
            vector = np.asarray(trial["probabilities"], dtype=np.float64)
            if vector.shape != (4,) or not np.isfinite(vector).all() or not np.isclose(vector.sum(), 1.0, atol=1e-5):
                raise RuntimeError(f"{name} produced invalid trial probabilities for {trial_id}")
    return aggregated, {"raw_transformer_train_only_normalization": raw_norm}


def individual_metrics(outputs):
    result = {}
    for name in MODEL_NAMES:
        records = {
            trial: {"true_class": value["true_class"], "final_emotion": value["predicted_class"]}
            for trial, value in outputs[name].items()
        }
        result[name] = compute_metrics(records)
    return result


def rows_for_csv(predictions):
    for trial_id, result in predictions.items():
        row = {
            "trial_id": trial_id,
            "true_emotion": result["true_class"],
            "adaptive_prediction": result["final_emotion"],
            "adaptive_confidence_score_not_calibrated": result["model_confidence_score_not_calibrated"],
            "adaptive_probabilities_json": json.dumps(dict(zip(CLASS_NAMES, result["fused_probabilities"])), separators=(",", ":")),
            "adaptive_trial_weights_json": json.dumps(result["trial_adaptive_weights"], separators=(",", ":")),
            "equal_prediction": result["equal_weight_baseline"]["final_emotion"],
            "equal_confidence_score_not_calibrated": result["equal_weight_baseline"]["model_confidence_score_not_calibrated"],
            "equal_probabilities_json": json.dumps(dict(zip(CLASS_NAMES, result["equal_weight_baseline"]["fused_probabilities"])), separators=(",", ":")),
        }
        for name in MODEL_NAMES:
            prefix = name.replace(" ", "_").replace("+", "plus")
            branch = result["individual_models"][name]
            row[f"{prefix}_prediction"] = branch["predicted_class"]
            row[f"{prefix}_window_count"] = branch["window_count"]
            row[f"{prefix}_probabilities_json"] = json.dumps(dict(zip(CLASS_NAMES, branch["probabilities"])), separators=(",", ":"))
        yield row


def write_csv(split, predictions, path):
    rows = list(rows_for_csv(predictions))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{split.upper()} PREDICTIONS CSV: {path}")


def evaluate_pair(equal_trials, adaptive_trials):
    equal_metrics = compute_metrics(equal_trials)
    adaptive_metrics = compute_metrics(adaptive_trials)
    return equal_metrics, adaptive_metrics, {
        "accuracy_absolute_change": adaptive_metrics["accuracy"] - equal_metrics["accuracy"],
        "macro_f1_absolute_change": adaptive_metrics["macro_f1"] - equal_metrics["macro_f1"],
    }


def combine_trial_outputs(equal_trials, adaptive_trials):
    combined = {}
    for trial_id, adaptive in adaptive_trials.items():
        combined[trial_id] = dict(adaptive)
        combined[trial_id]["equal_weight_baseline"] = {
            key: equal_trials[trial_id][key]
            for key in ("fused_probabilities", "final_emotion", "model_confidence_score_not_calibrated", "fusion_weights")
        }
    return combined


def base_model_table(metrics, input_types):
    return {
        name: {
            "input_type": input_types[name],
            "validation_metrics": metrics[name],
        }
        for name in MODEL_NAMES
    }


def markdown_report(results, path):
    lines = [
        "# Adaptive EEG Emotion Fusion",
        "",
        f"Run: `{results['run_id']}`",
        f"Frozen configuration: `{results['frozen_configuration']}`",
        f"Class order: `{results['class_order']}`",
        "Adaptive rule: validation macro-F1 reliability prior multiplied by a trial-specific normalized-entropy certainty gate; weights are normalized per trial.",
        f"Smoothing epsilon: `{EPSILON}`. Test labels and test metrics were not used to configure this rule.",
        "Adaptive confidence is the maximum fused score, not calibrated probability or uncertainty.",
        "",
        "## Base-model validation results",
        "",
        "| Model | Input | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        m = results["validation"]["base_model_metrics"][name]
        lines.append(f"| {name} | {results['models'][name]['input_type']} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['macro_f1']:.4f} |")
    lines += ["", "## Equal-weight fusion validation", "", *metric_lines(results["validation"]["equal_weight_metrics"])]
    lines += ["", "## Adaptive fusion validation", "", *metric_lines(results["validation"]["adaptive_metrics"])]
    changes = results["validation"]["comparison_absolute_change"]
    lines += [
        "",
        "## Equal versus adaptive validation comparison",
        "",
        f"- Accuracy absolute change: **{changes['accuracy_absolute_change']:+.4f}**",
        f"- Macro-F1 absolute change: **{changes['macro_f1_absolute_change']:+.4f}**",
        "- Validation reliability macro-F1 by model: " + json.dumps(results["validation"]["reliability_macro_f1"], sort_keys=True),
        "- Normalized reliability prior weights: " + json.dumps(
            results["adaptive_configuration"]["reliability_prior_weights"], sort_keys=True
        ),
        "",
        "### Trial-specific gate weights and predictions",
        "",
        "| Trial | True | Equal | Adaptive | Adaptive confidence score | Trial weights (model order) | Adaptive probabilities (fear, happy, neutral, sad) |",
        "|---|---|---|---|---:|---|---|",
    ]
    for trial_id, trial in results["validation"]["trials"].items():
        weights = ", ".join(f"{name}={trial['trial_adaptive_weights'][name]:.3f}" for name in MODEL_NAMES)
        probs = ", ".join(f"{x:.4f}" for x in trial["fused_probabilities"])
        lines.append(f"| {trial_id} | {trial['true_class']} | {trial['equal_weight_baseline']['final_emotion']} | {trial['final_emotion']} | {trial['model_confidence_score_not_calibrated']:.4f} | {weights} | [{probs}] |")
    lines += [
        "",
        "## Held-out test",
        "",
        f"Status: **{results['test']['status']}**.",
        "",
    ]
    if results["test"]["status"] == "evaluated_once":
        lines += [
            "Test metrics use the fixed adaptive configuration selected before test access.",
            "",
            "### Equal-weight test metrics",
            "",
            *metric_lines(results["test"]["equal_weight_metrics"]),
            "### Adaptive test metrics",
            "",
            *metric_lines(results["test"]["adaptive_metrics"]),
            f"Absolute change versus equal weights: accuracy {results['test']['comparison_absolute_change']['accuracy_absolute_change']:+.4f}; macro F1 {results['test']['comparison_absolute_change']['macro_f1_absolute_change']:+.4f}.",
            "",
        ]
    else:
        lines.append("Test inputs were not opened in the validation-only stage.")
        lines.append("")
    lines += [
        "The validation set contains four trials total (one per emotion), so the reliability prior and validation comparison have substantial uncertainty. Do not interpret validation gains as evidence of generalization.",
        "",
        "## Preprocessing and leakage controls",
        "",
        "- Base checkpoints and preprocessing are reused from the existing inference engine; no base model was retrained.",
        "- Raw EEG: fixed trial assignments, 62 channels, 2 s windows/0.5 s hop, 200 Hz assumption; GCN graph/statistics are saved training-only values; raw Transformer normalization is recomputed from train trials only.",
        "- STFT models: existing `crnn_dataset`, deterministic inference transforms, fixed trial split; no alternate dataset or model is included.",
        "- Fixed train/validation/test trial assignments and `[fear, happy, neutral, sad]` class order are preserved.",
        "- Test data was not accessed to fit reliability scores, weights, or the gate.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def metric_lines(metrics):
    out = [
        f"- Trial count: {metrics['trial_count']}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- Macro precision: {metrics['macro_precision']:.4f}",
        f"- Macro recall: {metrics['macro_recall']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for emotion in CLASS_NAMES:
        item = metrics["per_class"][emotion]
        out.append(f"| {emotion} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |")
    out += ["", "Confusion matrix (rows=true, columns=predicted):", "", "```text"]
    out += [str(row) for row in metrics["confusion_matrix_rows_true_columns_predicted"]]
    out += ["```", ""]
    return out


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_results(run_dir, results):
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=json_default), encoding="utf-8")
    markdown_report(results, run_dir / "fusion_report.md")


def run_validation(device):
    print("ADAPTIVE FUSION — VALIDATION ONLY")
    print(f"Raw dataset: {PROJECT_ROOT / 'data/output'}")
    print(f"STFT dataset: {PROJECT_ROOT / 'data/processed/crnn_dataset'}")
    print(f"Class order: {CLASS_NAMES}")
    print(f"Device: {device}")
    print("Test files will not be opened.")
    models, checkpoints, configs, graph_metadata, _ = load_models(device)
    outputs, raw_norm = infer_split("validation", models, device, graph_metadata)
    model_metrics = individual_metrics(outputs)
    reliability_prior, reliability_scores = validation_reliability_prior(outputs)
    equal_trials = equal_weight_fuse("validation", outputs)
    adaptive_trials = fuse_adaptively("validation", outputs, reliability_prior)
    equal_metrics, adaptive_metrics, comparison = evaluate_pair(equal_trials, adaptive_trials)
    trials = combine_trial_outputs(equal_trials, adaptive_trials)
    static_test = validate_static_test_contract()

    run_id = datetime.now().strftime("adaptive_run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    input_types = {
        "EEGNet + GCN": "Raw EEG",
        "Raw EEG Transformer": "Raw EEG",
        "STFT CRNN": "STFT EEG",
        "STFT Transformer": "STFT EEG",
    }
    results = {
        "schema_version": 1,
        "run_id": run_id,
        "frozen_configuration": {
            "id": FROZEN_CONFIG["configuration_id"],
            "path": str(FROZEN_CONFIG_PATH.relative_to(PROJECT_ROOT)),
        },
        "class_order": CLASS_NAMES,
        "model_names": MODEL_NAMES,
        "models": {
            name: {"checkpoint_path": checkpoints[name], "input_type": input_types[name], **configs[name]}
            for name in MODEL_NAMES
        },
        "adaptive_configuration": {
            "method": "validation macro-F1 reliability prior multiplied by trial-specific normalized-entropy certainty gate",
            "smoothing_epsilon": EPSILON,
            "reliability_score": "per-model validation macro-F1, computed over the fixed four validation trials",
            "reliability_prior_weights": reliability_prior,
            "gate": "g_i,t = epsilon + (1 - H(p_i,t)/ln(4)); w_i,t = prior_i*g_i,t / sum_j(prior_j*g_j,t)",
            "nonnegative_weights_sum_to_one": True,
            "test_used_for_configuration": False,
        },
        "equal_weight_baseline_configuration": {
            "weights": FUSION_WEIGHTS,
            "fusion": "mean of the four within-trial mean softmax probability vectors",
        },
        "preprocessing": {
            "raw_common": "62 channels, 200 Hz assumption, 2-second windows, 0.5-second hop, 1-45 Hz bandpass",
            "EEGNet + GCN": "saved training-only z-score and training-only functional adjacency graph",
            "Raw EEG Transformer": "training-trial-only per-channel mean/std recomputed from raw training recordings",
            "STFT CRNN": "existing crnn_dataset; RGB 224x224; ImageNet normalization; deterministic inference",
            "STFT Transformer": "existing crnn_dataset; grayscale 224x224; 0.5 mean/std; deterministic inference",
            "fixed_trial_split": {
                split: expected_trials(split)
                for split in ("train", "validation", "test")
            },
            "raw_transformer_runtime_normalization": raw_norm,
        },
        "validation": {
            "trial_count": len(trials),
            "number_of_windows_by_model": {
                name: sum(item["window_count"] for item in outputs[name].values()) for name in MODEL_NAMES
            },
            "base_model_metrics": model_metrics,
            "reliability_macro_f1": reliability_scores,
            "equal_weight_metrics": equal_metrics,
            "adaptive_metrics": adaptive_metrics,
            "comparison_absolute_change": comparison,
            "trials": trials,
        },
        "test": {
            "status": "not_evaluated",
            "reason": "Validation results shown for review before the separate one-time test stage.",
            "static_assignment_check_without_opening_data": static_test,
        },
        "checks": {
            "all_four_checkpoints_loaded_strictly": True,
            "common_class_order": CLASS_NAMES,
            "validation_has_all_four_model_outputs_for_every_trial": True,
            "per_window_logits_and_softmax_probabilities_generated": True,
            "trial_aggregation_precedes_fusion": True,
            "adaptive_trial_weights_nonnegative_normalized": True,
            "equal_baseline_weights_exactly_one_quarter_each": True,
            "test_data_opened_or_inferred": False,
            "test_used_for_fit_or_configuration": False,
            "fixed_split_preserved": True,
        },
        "limitations": [
            "Only four validation trials are available (one per class); validation-derived reliability weights are highly uncertain and potentially optimistic.",
            "The 200 Hz raw sampling rate is a project assumption because it is not stored in MAT files.",
            "Model confidence scores are not calibrated probabilities or uncertainty estimates.",
        ],
    }
    save_results(run_dir, results)
    write_csv("validation", trials, run_dir / "validation_predictions.csv")
    print(f"\nValidation reliability scores: {reliability_scores}")
    print(f"Validation reliability prior weights: {reliability_prior}")
    print_metrics("BASE-MODEL VALIDATION", model_metrics)
    print_metrics("EQUAL-WEIGHT VALIDATION", equal_metrics)
    print_metrics("ADAPTIVE VALIDATION", adaptive_metrics)
    print(f"Accuracy absolute change: {comparison['accuracy_absolute_change']:+.4f}")
    print(f"Macro-F1 absolute change: {comparison['macro_f1_absolute_change']:+.4f}")
    print("Trial gate weights:")
    for trial_id, item in trials.items():
        print(f"  {trial_id}: {item['trial_adaptive_weights']}")
    print(f"\nRun directory: {run_dir}")
    print(f"Results: {run_dir / 'results.json'}")
    print(f"Report: {run_dir / 'fusion_report.md'}")
    print("Test data was not opened or inferred.")
    return run_dir


def print_metrics(title, model_metrics):
    if set(model_metrics) == set(MODEL_NAMES):
        print(f"\n{title}")
        for name in MODEL_NAMES:
            metric = model_metrics[name]
            print(f"  {name}: accuracy={metric['accuracy']:.4f}, balanced_accuracy={metric['balanced_accuracy']:.4f}, macro_precision={metric['macro_precision']:.4f}, macro_recall={metric['macro_recall']:.4f}, macro_F1={metric['macro_f1']:.4f}")
        return
    print(f"\n{title}")
    print(f"  trials={model_metrics['trial_count']}; accuracy={model_metrics['accuracy']:.4f}; balanced_accuracy={model_metrics['balanced_accuracy']:.4f}; macro_precision={model_metrics['macro_precision']:.4f}; macro_recall={model_metrics['macro_recall']:.4f}; macro_F1={model_metrics['macro_f1']:.4f}")
    for emotion in CLASS_NAMES:
        item = model_metrics["per_class"][emotion]
        print(f"  {emotion}: precision={item['precision']:.4f}, recall={item['recall']:.4f}, F1={item['f1']:.4f}")
    print(f"  confusion matrix={model_metrics['confusion_matrix_rows_true_columns_predicted']}")


def run_test(device, run_dir):
    run_dir = Path(run_dir).resolve()
    result_path = run_dir / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    results = json.loads(result_path.read_text(encoding="utf-8"))
    if results.get("test", {}).get("status") != "not_evaluated":
        raise RuntimeError("Adaptive test evaluation already completed or run is invalid; refusing another pass")
    if results["class_order"] != CLASS_NAMES or results["equal_weight_baseline_configuration"]["weights"] != FUSION_WEIGHTS:
        raise RuntimeError("Saved validation configuration does not match current class order/equal baseline")
    guard = run_dir / "adaptive_test_evaluation_guard.json"
    if guard.exists():
        raise RuntimeError(f"Test evaluation already started: {guard}")
    guard.write_text(json.dumps({"status": "started", "split": "test"}, indent=2), encoding="utf-8")
    models, checkpoints, configs, graph_metadata, _ = load_models(device)
    for name in MODEL_NAMES:
        if str(checkpoints[name]) != results["models"][name]["checkpoint_path"]:
            raise RuntimeError(f"Checkpoint changed for {name}")
    outputs, _ = infer_split("test", models, device, graph_metadata)
    test_trials_equal = equal_weight_fuse("test", outputs)
    test_trials_adaptive = fuse_adaptively("test", outputs, results["adaptive_configuration"]["reliability_prior_weights"])
    test_trials = combine_trial_outputs(test_trials_equal, test_trials_adaptive)
    equal_metrics, adaptive_metrics, comparison = evaluate_pair(test_trials_equal, test_trials_adaptive)
    results["test"] = {
        "status": "evaluated_once",
        "trial_count": len(test_trials),
        "number_of_windows_by_model": {
            name: sum(item["window_count"] for item in outputs[name].values()) for name in MODEL_NAMES
        },
        "base_model_metrics": individual_metrics(outputs),
        "equal_weight_metrics": equal_metrics,
        "adaptive_metrics": adaptive_metrics,
        "comparison_absolute_change": comparison,
        "trials": test_trials,
    }
    results["checks"].update({
        "test_predictions_have_all_four_models_and_trials": True,
        "test_per_window_probabilities_finite_and_normalized": True,
        "test_configuration_fixed_before_test_access": True,
    })
    save_results(run_dir, results)
    write_csv("test", test_trials, run_dir / "test_predictions.csv")
    guard.write_text(json.dumps({"status": "completed", "split": "test"}, indent=2), encoding="utf-8")
    print_metrics("BASE-MODEL TEST", results["test"]["base_model_metrics"])
    print_metrics("EQUAL-WEIGHT TEST", equal_metrics)
    print_metrics("ADAPTIVE TEST", adaptive_metrics)
    print(f"Test accuracy change: {comparison['accuracy_absolute_change']:+.4f}")
    print(f"Test macro-F1 change: {comparison['macro_f1_absolute_change']:+.4f}")
    print("Test evaluation completed once; no tuning was performed from test results.")
    print(f"Updated results: {result_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validation-only", action="store_true", help="Run validation inference only (default).")
    group.add_argument("--eval-test", action="store_true", help="Evaluate held-out test once for an existing adaptive run.")
    parser.add_argument("--run-dir", help="Existing adaptive run directory; required for --eval-test.")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        device = device_for_inference()
    else:
        device = torch.device(args.device)
        if args.device == "mps" and not torch.backends.mps.is_available():
            parser.error("MPS is not available")
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is not available")
    if args.eval_test:
        if not args.run_dir:
            parser.error("--run-dir is required with --eval-test")
        run_test(device, args.run_dir)
    else:
        if args.run_dir:
            parser.error("--run-dir is only valid with --eval-test")
        run_validation(device)


if __name__ == "__main__":
    main()
