"""Export CSV and project summary from an already-evaluated fusion run.

This is a report-only utility. It never loads a model or reads dataset files.
"""

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_INPUTS = {
    "EEGNet + GCN": "Raw EEG",
    "Raw EEG Transformer": "Raw EEG",
    "STFT CRNN": "STFT EEG",
    "STFT Transformer": "STFT EEG",
    "Fusion": "Raw EEG + STFT EEG",
}


def write_predictions_csv(split_name, split_data, class_order, output_path):
    fields = [
        "trial_id", "true_emotion", "predicted_emotion", "model_confidence_score_not_calibrated",
        "window_counts_per_model_json", "fused_probabilities_json",
    ]
    for model_name, branch in split_data["trials"][next(iter(split_data["trials"]))]["individual_models"].items():
        prefix = model_name.replace(" ", "_").replace("+", "plus")
        fields.extend([f"{prefix}_prediction", f"{prefix}_window_count", f"{prefix}_probabilities_json"])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial_id, trial in split_data["trials"].items():
            row = {
                "trial_id": trial_id,
                "true_emotion": trial["true_class"],
                "predicted_emotion": trial["final_emotion"],
                "model_confidence_score_not_calibrated": trial["model_confidence_score_not_calibrated"],
                "window_counts_per_model_json": json.dumps(
                    {name: item["window_count"] for name, item in trial["individual_models"].items()},
                    separators=(",", ":"),
                ),
                "fused_probabilities_json": json.dumps(dict(zip(class_order, trial["fused_probabilities"])), separators=(",", ":")),
            }
            for model_name, prediction in trial["individual_models"].items():
                prefix = model_name.replace(" ", "_").replace("+", "plus")
                row[f"{prefix}_prediction"] = prediction["predicted_class"]
                row[f"{prefix}_window_count"] = prediction["window_count"]
                row[f"{prefix}_probabilities_json"] = json.dumps(dict(zip(class_order, prediction["probabilities"])), separators=(",", ":"))
            writer.writerow(row)
    print(f"{split_name.upper()} CSV: {output_path}")


def per_branch_window_counts(split_data):
    return {
        name: sum(trial["individual_models"][name]["window_count"] for trial in split_data["trials"].values())
        for name in split_data["individual_model_metrics"]
    }


def performance_table(results):
    rows = []
    for model_name in results["model_names"]:
        val = results["validation"]["individual_model_metrics"][model_name]
        test = results["test"]["individual_model_metrics"][model_name]
        rows.append({
            "model": model_name,
            "input": MODEL_INPUTS[model_name],
            "validation_trial_accuracy": val["accuracy"],
            "validation_macro_f1": val["macro_f1"],
            "test_trial_accuracy": test["accuracy"],
            "test_balanced_accuracy": test["balanced_accuracy"],
            "test_macro_f1": test["macro_f1"],
        })
    val = results["validation"]["metrics"]
    test = results["test"]["metrics"]
    rows.append({
        "model": "Fusion",
        "input": MODEL_INPUTS["Fusion"],
        "validation_trial_accuracy": val["accuracy"],
        "validation_macro_f1": val["macro_f1"],
        "test_trial_accuracy": test["accuracy"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "test_macro_f1": test["macro_f1"],
    })
    return rows


def create_project_summary(results, run_dir, output_path):
    known_training = {
        "EEGNet + GCN": "Window accuracy 0.?? (read from model result JSON below)",
        "Raw EEG Transformer": "Trial accuracy 1.0000 (reported completed baseline)",
        "STFT CRNN": "Accuracy 0.6261 (reported completed baseline)",
        "STFT Transformer": "Not present in saved run artifacts",
    }
    # Replace the GCN placeholder using its recorded training-window metric.
    gcn_result = PROJECT_ROOT / "models/eegnet_gcn/run_20261027_122050/results.json"
    if gcn_result.is_file():
        data = json.loads(gcn_result.read_text(encoding="utf-8"))
        if "train_window_accuracy" in data:
            known_training["EEGNet + GCN"] = f"Window accuracy {data['train_window_accuracy']:.4f}"
        else:
            known_training["EEGNet + GCN"] = "Not present in saved model result JSON"
    rows = performance_table(results)
    lines = [
        "# EEG Emotion Project Pipeline Summary",
        "",
        f"Fusion run: `{run_dir}`",
        "",
        "All reported model comparisons use trial-level predictions on the same fixed validation/test trial assignments. The held-out test set has four trials (one per class); its scores are descriptive and do not establish reliable generalization.",
        "",
        "## Training performance",
        "",
        "| Branch | Saved training metric |",
        "|---|---|",
    ]
    for model in results["model_names"]:
        lines.append(f"| {model} | {known_training[model]} |")
    lines += [
        "",
        "Training metrics come from existing run metadata or the completed baseline values supplied for this project. Training window accuracy and trial accuracy are different measures and are labeled accordingly; unavailable values are not inferred.",
        "",
        "## Validation performance",
        "",
        "| Model | Input | Trial accuracy | Macro F1 |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['input']} | {row['validation_trial_accuracy']:.4f} | {row['validation_macro_f1']:.4f} |")
    lines += [
        "",
        "## Held-out test performance",
        "",
        "| Model | Input | Trial accuracy | Balanced accuracy | Macro F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['input']} | {row['test_trial_accuracy']:.4f} | {row['test_balanced_accuracy']:.4f} | {row['test_macro_f1']:.4f} |")
    lines += [
        "",
        "## Fusion configuration and status",
        "",
        f"- Status: `{results['test']['status']}`; test evaluation was already completed once and was not repeated by this report export.",
        f"- Class order: `{results['class_order']}`.",
        f"- Weights: `{results['fusion_weights']}`.",
        "- Fusion confidence is the maximum fused score, not calibrated probability or uncertainty.",
        "- Individual models and fusion are complete as baseline experiments; no optimization was performed.",
        "",
        "## Window counts by split and branch",
        "",
        "| Split | " + " | ".join(results["model_names"]) + " |",
        "|---|" + "---:|" * len(results["model_names"]),
    ]
    for split_name in ("validation", "test"):
        counts = per_branch_window_counts(results[split_name])
        lines.append("| " + split_name.capitalize() + " | " + " | ".join(str(counts[name]) for name in results["model_names"]) + " |")
    lines += [
        "",
        f"Detailed results: `{run_dir / 'results.json'}`",
        f"Human-readable fusion report: `{run_dir / 'fusion_report.md'}`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"PROJECT SUMMARY: {output_path}")


def append_prediction_details(results, report_path):
    marker = "## Saved trial probability details"
    current = report_path.read_text(encoding="utf-8") if report_path.is_file() else "# Four-model EEG emotion fusion\n"
    if marker in current:
        current = current.split(marker, 1)[0].rstrip() + "\n\n"
    lines = [marker, ""]
    for split_name in ("validation", "test"):
        lines += [f"### {split_name.capitalize()} trials", ""]
        for trial_id, trial in results[split_name]["trials"].items():
            lines += [
                f"#### {trial_id}",
                "",
                f"- True emotion: **{trial['true_class']}**",
                f"- Fused prediction: **{trial['final_emotion']}**",
                f"- Model confidence score (not calibrated): **{trial['model_confidence_score_not_calibrated']:.6f}**",
                f"- Fused probabilities in class order `{results['class_order']}`: `{json.dumps(trial['fused_probabilities'])}`",
                "- Individual trial predictions/probabilities and window counts:",
            ]
            for name, individual in trial["individual_models"].items():
                lines.append(
                    f"  - {name}: prediction={individual['predicted_class']}; "
                    f"windows={individual['window_count']}; probabilities={json.dumps(individual['probabilities'])}"
                )
            lines.append("")
    report_path.write_text(current + "\n".join(lines), encoding="utf-8")


def print_saved_results(results):
    for split_name in ("validation", "test"):
        print(f"\n{split_name.upper()} SAVED METRICS")
        metrics = results[split_name]["metrics"]
        for key in ("trial_count", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"):
            print(f"  {key}: {metrics[key]}")
        print("  per_class:", json.dumps(metrics["per_class"], sort_keys=True))
        print("  confusion_matrix (rows=true, columns=predicted):", metrics["confusion_matrix_rows_true_columns_predicted"])
        print("  total windows by model:", json.dumps(per_branch_window_counts(results[split_name]), sort_keys=False))
        print("  trial predictions:")
        for trial_id, trial in results[split_name]["trials"].items():
            print(
                f"    {trial_id}: true={trial['true_class']}, predicted={trial['final_emotion']}, "
                f"confidence_score={trial['model_confidence_score_not_calibrated']:.6f}, "
                f"fused_probabilities={json.dumps(trial['fused_probabilities'])}"
            )
    print("\nFINAL MODEL COMPARISON (trial-level)")
    print("Model | Input | Validation accuracy | Validation macro F1 | Test accuracy | Test balanced accuracy | Test macro F1")
    for row in performance_table(results):
        print(
            f"{row['model']} | {row['input']} | {row['validation_trial_accuracy']:.4f} | "
            f"{row['validation_macro_f1']:.4f} | {row['test_trial_accuracy']:.4f} | "
            f"{row['test_balanced_accuracy']:.4f} | {row['test_macro_f1']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing completed fusion run directory")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    result_path = run_dir / "results.json"
    if not result_path.is_file():
        parser.error(f"Missing saved results: {result_path}")
    results = json.loads(result_path.read_text(encoding="utf-8"))
    if results.get("test", {}).get("status") != "evaluated_once":
        parser.error("This report-only exporter requires an already completed one-time test evaluation")
    if results.get("class_order") != ["fear", "happy", "neutral", "sad"]:
        parser.error("Saved result class order is incompatible")
    if results.get("fusion_weights") != {name: 0.25 for name in results["model_names"]}:
        parser.error("Saved result does not contain the agreed equal-weight configuration")
    for split_name in ("validation", "test"):
        split = results[split_name]
        if set(split["trials"]) != set(split.get("coverage", {}).get("expected_trial_ids", split["trials"])):
            # Validation stores no coverage map; its actual saved trial records are authoritative.
            if split_name == "test":
                parser.error(f"Saved {split_name} predictions do not match its expected trial coverage")
        if len(split["trials"]) != 4:
            parser.error(f"Expected four saved {split_name} trial records")
    write_predictions_csv("validation", results["validation"], results["class_order"], run_dir / "validation_predictions.csv")
    write_predictions_csv("test", results["test"], results["class_order"], run_dir / "test_predictions.csv")
    append_prediction_details(results, run_dir / "fusion_report.md")
    summary_path = PROJECT_ROOT / "models" / "fusion" / "PROJECT_PIPELINE_SUMMARY.md"
    create_project_summary(results, run_dir, summary_path)
    print_saved_results(results)
    print("No model inference, training, dataset access, or metric recomputation was performed.")
    print("Held-out test evaluation was not repeated.")


if __name__ == "__main__":
    main()
