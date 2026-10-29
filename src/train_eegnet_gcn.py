"""Train and evaluate the new raw multichannel EEGNet + GCN experiment.

Run ``python src/train_eegnet_gcn.py --check-data`` for a data-only loader and
shape check. Running without that flag starts training. Existing project
datasets and model files are not read or modified.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from eegnet_gcn_data import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    HOP_SAMPLES,
    SAMPLING_RATE_HZ,
    SEED,
    WINDOW_SAMPLES,
    prepare_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models" / "eegnet_gcn"

BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 4
NUM_WORKERS = 0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EEGNetGCN(nn.Module):
    """EEGNet global branch fused with graph features for 62 electrode nodes."""

    def __init__(self, normalized_adjacency, num_classes=4):
        super().__init__()
        self.register_buffer("adjacency", normalized_adjacency.float())

        # Shared temporal EEGNet convolution, preserving the electrode axis.
        self.temporal_conv = nn.Conv2d(1, 8, kernel_size=(1, 64), bias=False)
        self.temporal_bn = nn.BatchNorm2d(8)

        # Canonical EEGNet depthwise spatial filter across all electrodes.
        # This forms the global EEGNet branch and intentionally collapses
        # electrode height; the graph branch taps features before this layer.
        self.spatial_depthwise = nn.Conv2d(
            8, 16, kernel_size=(62, 1), groups=8, bias=False
        )
        self.spatial_bn = nn.BatchNorm2d(16)
        self.eegnet_dropout = nn.Dropout(0.25)
        self.separable_depthwise = nn.Conv2d(
            16, 16, kernel_size=(1, 16), groups=16, bias=False
        )
        self.separable_pointwise = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        self.separable_bn = nn.BatchNorm2d(16)

        # Per-channel node representation from the temporal features before
        # EEGNet's electrode-collapsing depthwise spatial convolution.
        self.node_projection = nn.Sequential(
            nn.Linear(8 * 25, 32),
            nn.LayerNorm(32),
            nn.ELU(),
        )

        # Two graph-convolution layers, with a residual projection.
        self.gcn_1 = nn.Linear(32, 64, bias=False)
        self.gcn_1_bias = nn.Parameter(torch.zeros(64))
        self.gcn_2 = nn.Linear(64, 64, bias=False)
        self.gcn_2_bias = nn.Parameter(torch.zeros(64))
        self.gcn_residual = nn.Linear(64, 64, bias=False)
        self.gcn_dropout = nn.Dropout(0.30)

        # EEGNet global feature (16) + graph mean/max readout (128).
        self.classifier = nn.Sequential(
            nn.Linear(16 + 64 + 64, 64),
            nn.ELU(),
            nn.Dropout(0.50),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _same_time_pad(x, total_padding):
        left = total_padding // 2
        right = total_padding - left
        return F.pad(x, (left, right, 0, 0))

    def forward(self, x):
        # x: batch x 1 x 62 channels x 400 samples
        x = self._same_time_pad(x, 63)
        temporal = F.elu(self.temporal_bn(self.temporal_conv(x)))

        # EEGNet global branch: depthwise spatial then separable temporal.
        global_features = F.elu(self.spatial_bn(self.spatial_depthwise(temporal)))
        global_features = F.avg_pool2d(global_features, kernel_size=(1, 4))
        global_features = self.eegnet_dropout(global_features)
        global_features = self._same_time_pad(global_features, 15)
        global_features = self.separable_depthwise(global_features)
        global_features = self.separable_pointwise(global_features)
        global_features = F.elu(self.separable_bn(global_features))
        global_features = F.avg_pool2d(global_features, kernel_size=(1, 8))
        global_features = global_features.mean(dim=(-2, -1))  # batch x 16

        # Graph branch: retain 62 nodes, summarize each node's temporal map.
        batch, filters, channels, samples = temporal.shape
        node_time = temporal.permute(0, 2, 1, 3).reshape(batch * channels, filters, samples)
        node_time = F.adaptive_avg_pool1d(node_time, output_size=25)
        node_time = node_time.reshape(batch, channels, filters * 25)
        nodes = self.node_projection(node_time)  # batch x 62 x 32

        adjacency_nodes = torch.einsum("ij,bjf->bif", self.adjacency, nodes)
        hidden = F.elu(self.gcn_1(adjacency_nodes) + self.gcn_1_bias)
        hidden = self.gcn_dropout(hidden)
        adjacency_hidden = torch.einsum("ij,bjf->bif", self.adjacency, hidden)
        graph_features = F.elu(
            self.gcn_2(adjacency_hidden)
            + self.gcn_2_bias
            + self.gcn_residual(hidden)
        )

        graph_mean = graph_features.mean(dim=1)
        graph_max = graph_features.max(dim=1).values
        fused = torch.cat((global_features, graph_mean, graph_max), dim=1)
        return self.classifier(fused)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def print_data_summary(datasets, metadata):
    print("RAW EEG DATA CHECK")
    print(f"  Source root: {PROJECT_ROOT / 'data' / 'output'}")
    print(f"  Channels: {metadata['channels']}")
    print(f"  Sampling rate assumption: {SAMPLING_RATE_HZ} Hz")
    print(f"  Window: {WINDOW_SAMPLES} samples ({WINDOW_SAMPLES / SAMPLING_RATE_HZ:.1f}s)")
    print(f"  Hop: {HOP_SAMPLES} samples ({HOP_SAMPLES / SAMPLING_RATE_HZ:.1f}s)")
    print(f"  Normalization: {metadata['normalization_source']}")
    print(f"  Graph: {metadata['graph_source']}")
    print("  Trial/window counts:")
    for split in ("train", "validation", "test"):
        print(
            f"    {split:10s}: trials={metadata['split_trial_counts'][split]}, "
            f"windows={len(datasets[split])}, "
            f"classes={metadata['split_class_window_counts'][split]}"
        )
    print(f"  Trial overlap check: {metadata['trial_intersections']}")
    for split, dataset in datasets.items():
        x, y, trial_index = dataset[0]
        print(
            f"  {split} sample shapes: x={tuple(x.shape)}, y={tuple(y.shape)}, "
            f"trial_index={tuple(trial_index.shape)}"
        )


def make_trial_balanced_window_weights(train_dataset):
    """Each class and each training trial receives equal total loss weight."""
    window_counts = train_dataset.window_counts_by_trial
    trial_classes = {}
    for trial, _ in train_dataset.index:
        trial_classes[trial.key] = CLASS_TO_INDEX[trial.emotion]
    trial_counts_per_class = {
        class_index: sum(value == class_index for value in trial_classes.values())
        for class_index in range(len(CLASS_NAMES))
    }
    weights = []
    for trial, _ in train_dataset.index:
        class_index = CLASS_TO_INDEX[trial.emotion]
        weight = 1.0 / (
            trial_counts_per_class[class_index] * window_counts[trial.key]
        )
        weights.append(weight)
    weights = np.asarray(weights, dtype=np.float32)
    weights /= weights.mean()
    return torch.from_numpy(weights), trial_counts_per_class


def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    all_trial_indices = []
    with torch.no_grad():
        for x, labels, trial_indices in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_trial_indices.append(trial_indices.cpu())

    window_logits = torch.cat(all_logits)
    window_labels = torch.cat(all_labels)
    window_trial_indices = torch.cat(all_trial_indices)

    trial_logits = []
    trial_labels = []
    for trial_index in torch.unique(window_trial_indices, sorted=True):
        mask = window_trial_indices == trial_index
        trial_logits.append(window_logits[mask].mean(dim=0))
        labels_for_trial = torch.unique(window_labels[mask])
        if labels_for_trial.numel() != 1:
            raise RuntimeError(f"Trial {trial_index.item()} has inconsistent emotion labels")
        trial_labels.append(labels_for_trial[0])

    trial_logits = torch.stack(trial_logits)
    trial_labels = torch.stack(trial_labels)
    trial_loss = F.cross_entropy(trial_logits, trial_labels).item()

    window_predictions = window_logits.argmax(dim=1).numpy()
    window_targets = window_labels.numpy()
    trial_predictions = trial_logits.argmax(dim=1).numpy()
    trial_targets = trial_labels.numpy()

    return {
        "trial_loss": trial_loss,
        "trial_accuracy": accuracy_score(trial_targets, trial_predictions),
        "trial_balanced_accuracy": balanced_accuracy_score(trial_targets, trial_predictions),
        "trial_macro_f1": f1_score(
            trial_targets, trial_predictions, labels=range(4), average="macro", zero_division=0
        ),
        "trial_precision_recall_f1": precision_recall_fscore_support(
            trial_targets, trial_predictions, labels=range(4), zero_division=0
        ),
        "trial_confusion_matrix": confusion_matrix(trial_targets, trial_predictions, labels=range(4)),
        "trial_report": classification_report(
            trial_targets,
            trial_predictions,
            labels=range(4),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ),
        "window_accuracy": accuracy_score(window_targets, window_predictions),
        "window_macro_f1": f1_score(
            window_targets, window_predictions, labels=range(4), average="macro", zero_division=0
        ),
        "window_confusion_matrix": confusion_matrix(window_targets, window_predictions, labels=range(4)),
        "trial_targets": trial_targets,
        "trial_predictions": trial_predictions,
    }


def run_data_check():
    datasets, adjacency, raw_adjacency, metadata = prepare_data()
    print_data_summary(datasets, metadata)
    expected = (1, 62, WINDOW_SAMPLES)
    for split, dataset in datasets.items():
        x, label, _ = dataset[0]
        if tuple(x.shape) != expected:
            raise RuntimeError(f"Unexpected {split} tensor shape {tuple(x.shape)} != {expected}")
        if label.item() not in range(4):
            raise RuntimeError(f"Invalid class index in {split}: {label.item()}")
    if tuple(adjacency.shape) != (62, 62) or tuple(raw_adjacency.shape) != (62, 62):
        raise RuntimeError("Expected 62 x 62 functional graph matrices")
    model = EEGNetGCN(adjacency)
    with torch.no_grad():
        logits = model(torch.zeros(2, 1, 62, WINDOW_SAMPLES))
    if tuple(logits.shape) != (2, 4):
        raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
    print(f"  Normalized adjacency shape: {tuple(adjacency.shape)}")
    print("  Model smoke shape: input=(2, 1, 62, 400), output=(2, 4)")
    print("DATA CHECK PASSED; no training was started and no files were written.")


def train():
    seed_everything(SEED)
    device = select_device()
    datasets, adjacency, raw_adjacency, metadata = prepare_data()
    print_data_summary(datasets, metadata)
    print(f"Device: {device}")
    print(f"Classes: {CLASS_NAMES}")

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
    )
    train_eval_loader = DataLoader(
        datasets["train"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    window_weights, trial_counts_per_class = make_trial_balanced_window_weights(datasets["train"])
    trial_index_to_weight = {}
    for sample_index, (trial, _) in enumerate(datasets["train"].index):
        trial_index = int(datasets["train"][sample_index][2])
        trial_index_to_weight.setdefault(trial_index, float(window_weights[sample_index]))
    print(f"Train trial counts per class: {trial_counts_per_class}")
    print("Loss weighting: equal total contribution per class and per trial within class")

    model = EEGNetGCN(adjacency).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE
    )

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = MODEL_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    best_path = run_dir / "best_eegnet_gcn.pth"
    (run_dir / "normalization_and_graph.json").write_text(
        json.dumps(
            json_ready(
                {
                    **metadata,
                    "raw_functional_adjacency": raw_adjacency,
                    "normalized_adjacency": adjacency.numpy(),
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print("\nTRAINING (validation trial-aggregated loss selects the checkpoint)")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        correct = 0
        sample_count = 0

        for x, labels, trial_indices in train_loader:
            x = x.to(device)
            labels = labels.to(device)
            trial_indices_cpu = trial_indices.tolist()
            sample_weights = torch.tensor(
                [trial_index_to_weight[index] for index in trial_indices_cpu],
                dtype=torch.float32,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            weighted_loss_sum += float((per_sample_loss.detach() * sample_weights).sum())
            weight_sum += float(sample_weights.sum())
            correct += int((logits.argmax(dim=1) == labels).sum())
            sample_count += labels.numel()

        validation = evaluate(model, validation_loader, device)
        scheduler.step(validation["trial_loss"])
        train_loss = weighted_loss_sum / max(weight_sum, 1e-12)
        train_accuracy = correct / max(sample_count, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_window_accuracy": train_accuracy,
            "validation_trial_loss": validation["trial_loss"],
            "validation_trial_accuracy": validation["trial_accuracy"],
            "validation_trial_macro_f1": validation["trial_macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | train loss={train_loss:.4f} "
            f"window acc={train_accuracy:.4f} | val trial loss={validation['trial_loss']:.4f} "
            f"acc={validation['trial_accuracy']:.4f} macro-F1={validation['trial_macro_f1']:.4f}"
        )

        if validation["trial_loss"] < best_validation_loss - 1e-4:
            best_validation_loss = validation["trial_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "best_epoch": best_epoch,
                    "best_validation_trial_loss": best_validation_loss,
                    "seed": SEED,
                    "sampling_rate_hz_assumption": SAMPLING_RATE_HZ,
                    "window_samples": WINDOW_SAMPLES,
                    "hop_samples": HOP_SAMPLES,
                },
                best_path,
            )
            print(f"  Saved best validation checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch} epochs without validation-loss improvement.")
            break

    if best_epoch == 0:
        raise RuntimeError("Training ended without a best validation checkpoint")

    # The test loader is first iterated here, once, after model selection.
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate(model, test_loader, device)
    train_result = evaluate(model, train_eval_loader, device)

    trial_precision, trial_recall, trial_f1, trial_support = test_result["trial_precision_recall_f1"]
    print("\nFINAL HELD-OUT TEST RESULTS (trial-aggregated; one vote per trial)")
    print(f"Best epoch: {best_epoch}")
    print(f"Training window accuracy: {train_result['window_accuracy']:.4f}")
    print(f"Test trial accuracy: {test_result['trial_accuracy']:.4f}")
    print(f"Test trial balanced accuracy: {test_result['trial_balanced_accuracy']:.4f}")
    print(f"Test trial macro F1: {test_result['trial_macro_f1']:.4f}")
    print("Per-class trial precision / recall / F1 / support:")
    for index, emotion in enumerate(CLASS_NAMES):
        print(
            f"  {emotion:<8}: {trial_precision[index]:.4f} / {trial_recall[index]:.4f} / "
            f"{trial_f1[index]:.4f} / {int(trial_support[index])}"
        )
    print("Trial confusion matrix (rows=true, columns=predicted):")
    print(test_result["trial_confusion_matrix"])
    print("\nSecondary overlapping-window metrics (windows are not independent):")
    print(f"  Window accuracy: {test_result['window_accuracy']:.4f}")
    print(f"  Window macro F1: {test_result['window_macro_f1']:.4f}")
    print("  Window confusion matrix:")
    print(test_result["window_confusion_matrix"])
    print("\nTrial-level classification report:")
    print(test_result["trial_report"])

    result_path = run_dir / "results.json"
    result_path.write_text(
        json.dumps(
            json_ready(
                {
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_validation_trial_loss": best_validation_loss,
                    "train_window_accuracy": train_result["window_accuracy"],
                    "test_trial_accuracy": test_result["trial_accuracy"],
                    "test_trial_balanced_accuracy": test_result["trial_balanced_accuracy"],
                    "test_trial_macro_f1": test_result["trial_macro_f1"],
                    "test_trial_precision_recall_f1_support": {
                        emotion: {
                            "precision": trial_precision[index],
                            "recall": trial_recall[index],
                            "f1": trial_f1[index],
                            "support": trial_support[index],
                        }
                        for index, emotion in enumerate(CLASS_NAMES)
                    },
                    "test_trial_confusion_matrix": test_result["trial_confusion_matrix"],
                    "test_window_accuracy_secondary": test_result["window_accuracy"],
                    "test_window_macro_f1_secondary": test_result["window_macro_f1"],
                    "test_window_confusion_matrix_secondary": test_result["window_confusion_matrix"],
                    "history": history,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nBest model and experiment outputs: {run_dir}")
    print(f"Results: {result_path}")
    print("The test set was evaluated once after validation-based checkpoint selection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Load data, verify split/graph/model tensor shapes, and exit without training.",
    )
    args = parser.parse_args()
    seed_everything(SEED)
    if args.check_data:
        run_data_check()
    else:
        train()


if __name__ == "__main__":
    main()

