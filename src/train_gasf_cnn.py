"""Rough GASF-CNN baseline using raw EEG and fixed trial-level splits.

GASF images are prepared in memory from the first EEG channel, matching the
channel-1 direction of src/test_gasf.py. No existing data or model files are
read for training outputs or modified. Use --check-data for a no-training
preprocessing and shape check.
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "output"
MODEL_ROOT = PROJECT_ROOT / "models" / "gasf_cnn"

CLASS_NAMES = ("fear", "happy", "neutral", "sad")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
MAT_LABELS = {"fear": 2, "happy": 3, "neutral": 0, "sad": 1}
SPLITS = ("train", "validation", "test")

SEED = 42
CHANNELS_IN_RAW = 62
GASF_CHANNEL_INDEX = 0  # Channel 1 in the existing one-based GASF test script.
SAMPLING_RATE_HZ = 200  # Project assumption; the MAT files do not store this value.
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(SAMPLING_RATE_HZ * WINDOW_SECONDS)
HOP_SECONDS = 0.5
HOP_SAMPLES = int(SAMPLING_RATE_HZ * HOP_SECONDS)
LOWCUT_HZ = 1.0
HIGHCUT_HZ = 40.0
NOTCH_HZ = 50.0
GASF_SIGNAL_SAMPLES = 64
SIGNAL_SMOOTHING_SIGMA = 2.0
GASF_SMOOTHING_SIGMA = 0.6
SCALE_PERCENTILES = (1.0, 99.0)

# Basic CNN and training settings; deliberately a baseline, not a search.
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7
SCHEDULER_PATIENCE = 3
NUM_WORKERS = 0

TRIAL_ASSIGNMENTS = {
    "train": {
        "fear": ("1_20160518_trial11", "1_20160518_trial18", "1_20160518_trial2"),
        "happy": ("1_20160518_trial16", "1_20160518_trial19", "1_20160518_trial24"),
        "neutral": ("1_20160518_trial21", "1_20160518_trial23", "1_20160518_trial6", "1_20160518_trial7"),
        "sad": ("1_20160518_trial10", "1_20160518_trial14"),
    },
    "validation": {
        "fear": ("1_20160518_trial17",),
        "happy": ("1_20160518_trial20",),
        "neutral": ("1_20160518_trial4",),
        "sad": ("1_20160518_trial12",),
    },
    "test": {
        "fear": ("1_20160518_trial5",),
        "happy": ("1_20160518_trial22",),
        "neutral": ("1_20160518_trial9",),
        "sad": ("1_20160518_trial1",),
    },
}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def select_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_assignment_map():
    assignments = {}
    for split, classes in TRIAL_ASSIGNMENTS.items():
        for emotion, keys in classes.items():
            for key in keys:
                if key in assignments:
                    raise ValueError(f"Trial assigned to multiple splits: {key}")
                assignments[key] = (split, emotion)
    if len(assignments) != 20:
        raise ValueError(f"Expected 20 trial assignments, got {len(assignments)}")
    return assignments


def preprocess_channel(eeg, path):
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2 or eeg.shape[0] != CHANNELS_IN_RAW:
        raise ValueError(
            f"Expected raw EEG shape ({CHANNELS_IN_RAW}, time) in {path}; got {eeg.shape}"
        )
    x = np.nan_to_num(eeg[GASF_CHANNEL_INDEX], nan=0.0, posinf=0.0, neginf=0.0)
    x = signal.detrend(x, type="linear")
    nyquist = SAMPLING_RATE_HZ / 2.0
    sos = signal.butter(
        4,
        [LOWCUT_HZ, HIGHCUT_HZ],
        btype="bandpass",
        fs=SAMPLING_RATE_HZ,
        output="sos",
    )
    x = signal.sosfiltfilt(sos, x)
    if NOTCH_HZ < nyquist:
        b, a = signal.iirnotch(w0=NOTCH_HZ, Q=30, fs=SAMPLING_RATE_HZ)
        x = signal.filtfilt(b, a, x)
    x = gaussian_filter1d(x, sigma=SIGNAL_SMOOTHING_SIGMA)
    if x.size < WINDOW_SAMPLES:
        raise ValueError(f"Recording shorter than one 2-second window: {path}")
    return x.astype(np.float32)


def load_trials():
    assignments = get_assignment_map()
    trials = []
    seen = set()
    for emotion in CLASS_NAMES:
        class_dir = RAW_ROOT / emotion
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing raw EEG class directory: {class_dir}")
        for path in sorted(class_dir.glob("*.mat")):
            key = path.stem
            if key not in assignments:
                raise ValueError(f"Raw trial is not included in the fixed split: {path}")
            split, expected_emotion = assignments[key]
            if expected_emotion != emotion:
                raise ValueError(f"Fixed split/folder class mismatch: {path}")
            mat = loadmat(path)
            if "EEG" not in mat or "label" not in mat:
                raise KeyError(f"Expected EEG and label variables in {path}")
            label = np.asarray(mat["label"]).squeeze()
            if label.size != 1 or int(label) != MAT_LABELS[emotion]:
                raise ValueError(f"MAT label does not match {emotion} folder in {path}: {label}")
            trials.append({
                "key": key,
                "emotion": emotion,
                "split": split,
                "signal": preprocess_channel(mat["EEG"], path),
            })
            seen.add(key)
    if seen != set(assignments):
        raise ValueError(
            f"Raw file coverage differs from fixed assignments: "
            f"missing={sorted(set(assignments)-seen)}, unexpected={sorted(seen-set(assignments))}"
        )
    return trials


def train_only_scale(trials):
    train_values = np.concatenate(
        [trial["signal"] for trial in trials if trial["split"] == "train"]
    )
    low, high = np.percentile(train_values, SCALE_PERCENTILES)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid training-only signal scale: low={low}, high={high}")
    return float(low), float(high), int(train_values.size)


def make_gasf(window, scale_low, scale_high):
    # The fixed range is calculated from training recordings only. The same
    # bounds are applied to validation and test, with no per-test rescaling.
    scaled = 2.0 * (window - scale_low) / (scale_high - scale_low) - 1.0
    scaled = np.clip(scaled, -1.0, 1.0)
    sampled = signal.resample(scaled, GASF_SIGNAL_SAMPLES)
    sampled = np.clip(sampled, -1.0, 1.0)
    phi = np.arccos(sampled)
    gasf = np.cos(phi[:, None] + phi[None, :])
    gasf = gaussian_filter(gasf, sigma=GASF_SMOOTHING_SIGMA)
    return np.clip(gasf, -1.0, 1.0).astype(np.float32)


class GASFWindowDataset(Dataset):
    """In-memory GASF tensors generated from each split's assigned raw trials."""

    def __init__(self, trials, split, scale_low, scale_high, trial_index_map):
        self.samples = []
        self.trial_index_map = trial_index_map
        self.window_counts_by_trial = {}
        for trial in trials:
            if trial["split"] != split:
                continue
            last_start = trial["signal"].size - WINDOW_SAMPLES
            count = 0
            for start in range(0, last_start + 1, HOP_SAMPLES):
                window = trial["signal"][start:start + WINDOW_SAMPLES]
                image = make_gasf(window, scale_low, scale_high)
                self.samples.append((
                    image,
                    CLASS_TO_INDEX[trial["emotion"]],
                    trial_index_map[trial["key"]],
                    trial["key"],
                    trial["emotion"],
                ))
                count += 1
            self.window_counts_by_trial[trial["key"]] = count
        if not self.samples:
            raise ValueError(f"No GASF windows created for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, label, trial_id, _, _ = self.samples[index]
        return (
            torch.from_numpy(image[None, :, :]),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(trial_id, dtype=torch.long),
        )


def prepare_data():
    trials = load_trials()
    scale_low, scale_high, train_value_count = train_only_scale(trials)
    keys = sorted(trial["key"] for trial in trials)
    trial_index_map = {key: index for index, key in enumerate(keys)}
    datasets = {
        split: GASFWindowDataset(trials, split, scale_low, scale_high, trial_index_map)
        for split in SPLITS
    }
    split_trials = {
        split: {trial["key"] for trial in trials if trial["split"] == split}
        for split in SPLITS
    }
    overlap = {
        "train∩validation": split_trials["train"] & split_trials["validation"],
        "train∩test": split_trials["train"] & split_trials["test"],
        "validation∩test": split_trials["validation"] & split_trials["test"],
    }
    if any(overlap.values()):
        raise RuntimeError(f"Trial leakage detected: {overlap}")
    metadata = {
        "train_scale_low": scale_low,
        "train_scale_high": scale_high,
        "train_scale_percentiles": list(SCALE_PERCENTILES),
        "train_scale_value_count": train_value_count,
        "scale_source": "training trials only",
        "split_trial_counts": {split: len(value) for split, value in split_trials.items()},
        "split_window_counts": {split: len(datasets[split]) for split in SPLITS},
        "split_class_counts": {
            split: {
                emotion: sum(sample[1] == CLASS_TO_INDEX[emotion] for sample in datasets[split].samples)
                for emotion in CLASS_NAMES
            }
            for split in SPLITS
        },
        "window_counts_by_trial": {
            split: datasets[split].window_counts_by_trial for split in SPLITS
        },
        "trial_assignments": TRIAL_ASSIGNMENTS,
        "trial_overlap": {key: sorted(value) for key, value in overlap.items()},
    }
    return datasets, metadata


class GASFCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        blocks = []
        in_channels = 1
        for out_channels in (32, 64, 128, 256):
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def trial_weights(dataset):
    trial_window_counts = dataset.window_counts_by_trial
    trial_class = {}
    for sample in dataset.samples:
        trial_class[sample[3]] = sample[4]
    class_trial_counts = {
        emotion: sum(value == emotion for value in trial_class.values())
        for emotion in CLASS_NAMES
    }
    weights = {}
    for key, emotion in trial_class.items():
        weights[key] = 1.0 / (class_trial_counts[emotion] * trial_window_counts[key])
    mean_weight = np.mean([
        weights[sample[3]] for sample in dataset.samples
    ])
    return {key: value / mean_weight for key, value in weights.items()}, class_trial_counts


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels, all_trial_ids = [], [], []
    with torch.no_grad():
        for images, labels, trial_ids in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels.cpu())
            all_trial_ids.append(trial_ids.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    trial_ids = torch.cat(all_trial_ids)
    trial_logits, trial_labels = [], []
    for trial_id in torch.unique(trial_ids, sorted=True):
        mask = trial_ids == trial_id
        labels_for_trial = torch.unique(labels[mask])
        if labels_for_trial.numel() != 1:
            raise RuntimeError("Trial has inconsistent class labels")
        trial_logits.append(logits[mask].mean(dim=0))
        trial_labels.append(labels_for_trial[0])
    trial_logits = torch.stack(trial_logits)
    trial_labels = torch.stack(trial_labels)
    target, prediction = trial_labels.numpy(), trial_logits.argmax(1).numpy()
    window_target, window_prediction = labels.numpy(), logits.argmax(1).numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        target, prediction, labels=range(4), zero_division=0
    )
    return {
        "trial_loss": F.cross_entropy(trial_logits, trial_labels).item(),
        "trial_accuracy": accuracy_score(target, prediction),
        "trial_balanced_accuracy": balanced_accuracy_score(target, prediction),
        "trial_macro_f1": f1_score(target, prediction, labels=range(4), average="macro", zero_division=0),
        "trial_precision": precision,
        "trial_recall": recall,
        "trial_f1": f1,
        "trial_support": support,
        "trial_confusion_matrix": confusion_matrix(target, prediction, labels=range(4)),
        "trial_report": classification_report(
            target, prediction, labels=range(4), target_names=CLASS_NAMES, digits=4, zero_division=0
        ),
        "window_accuracy": accuracy_score(window_target, window_prediction),
        "window_macro_f1": f1_score(
            window_target, window_prediction, labels=range(4), average="macro", zero_division=0
        ),
        "window_confusion_matrix": confusion_matrix(window_target, window_prediction, labels=range(4)),
    }


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_results(results, path):
    serializable = dict(results)
    for key in ("training", "validation", "test"):
        metrics = results.get(key)
        if isinstance(metrics, dict):
            metrics = dict(metrics)
            report = metrics.pop("trial_report", None)
            if report is not None:
                metrics.setdefault("trial_report_text", report)
            serializable[key] = metrics
    Path(path).write_text(json.dumps(json_ready(serializable), indent=2), encoding="utf-8")


def print_summary(datasets, metadata):
    print(f"Dataset: {RAW_ROOT} (raw MAT; GASF channel index {GASF_CHANNEL_INDEX + 1})")
    print(f"Class mapping: {CLASS_TO_INDEX}")
    print(
        f"Window: {WINDOW_SAMPLES} samples ({WINDOW_SECONDS}s), hop {HOP_SAMPLES} samples ({HOP_SECONDS}s); "
        f"GASF shape={GASF_SIGNAL_SAMPLES}x{GASF_SIGNAL_SAMPLES}"
    )
    print(f"Sampling rate assumption: {SAMPLING_RATE_HZ} Hz")
    print(
        f"Train-only signal clipping range (p{SCALE_PERCENTILES[0]:g}-p{SCALE_PERCENTILES[1]:g}): "
        f"{metadata['train_scale_low']:.6g} to {metadata['train_scale_high']:.6g}"
    )
    for split in SPLITS:
        print(
            f"{split:10s}: trials={metadata['split_trial_counts'][split]}, "
            f"windows={len(datasets[split])}, classes={metadata['split_class_counts'][split]}"
        )
    print(f"Trial overlap check: {metadata['trial_overlap']}")


def check_data():
    datasets, metadata = prepare_data()
    print_summary(datasets, metadata)
    for split, dataset in datasets.items():
        image, label, _ = dataset[0]
        if tuple(image.shape) != (1, GASF_SIGNAL_SAMPLES, GASF_SIGNAL_SAMPLES):
            raise RuntimeError(f"Unexpected {split} GASF tensor shape: {tuple(image.shape)}")
        if not torch.isfinite(image).all() or label.item() not in range(4):
            raise RuntimeError(f"Invalid GASF tensor or class in {split}")
    output = GASFCNN()(torch.zeros(2, 1, GASF_SIGNAL_SAMPLES, GASF_SIGNAL_SAMPLES))
    if tuple(output.shape) != (2, 4):
        raise RuntimeError(f"Unexpected CNN output shape: {tuple(output.shape)}")
    print("Shape check passed: input=(batch, 1, 64, 64), output=(batch, 4). No training started.")


def train():
    seed_everything()
    device = select_device()
    datasets, metadata = prepare_data()
    print("GASF-CNN BASELINE")
    print_summary(datasets, metadata)
    print(f"Device: {device}")
    print("Warnings: channel names/order and electrode coordinates are not stored in the MAT files.")
    print("Warnings: validation and test each contain only one independent trial per class.")
    print("GASF uses raw channel 1, matching the existing single-channel GASF test; no test-derived scaling is used.")

    generator = torch.Generator().manual_seed(SEED)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=BATCH_SIZE, shuffle=True,
            generator=generator, num_workers=NUM_WORKERS,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        ),
        "test": DataLoader(
            datasets["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        ),
    }
    sample_weights, class_trial_counts = trial_weights(datasets["train"])
    trial_weight_by_index = {
        sample[2]: sample_weights[sample[3]]
        for sample in datasets["train"].samples
    }
    model = GASFCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE
    )

    run_dir = MODEL_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    best_path = run_dir / "best_gasf_cnn.pth"
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    start_time = time.time()
    print(f"Training trial counts per class: {class_trial_counts}")
    print("Training loss weights balance class and trial contributions; validation/test are not rebalanced.")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = weight_sum = 0.0
        correct = total = 0
        for images, labels, trial_ids in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            weights = torch.tensor(
                [trial_weight_by_index[int(index)] for index in trial_ids.tolist()],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            per_item_loss = F.cross_entropy(logits, labels, reduction="none")
            loss = (per_item_loss * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float((per_item_loss.detach() * weights).sum())
            weight_sum += float(weights.sum())
            correct += int((logits.argmax(1) == labels).sum())
            total += labels.numel()

        validation = evaluate(model, loaders["validation"], device)
        scheduler.step(validation["trial_loss"])
        train_loss = loss_sum / max(weight_sum, 1e-12)
        train_accuracy = correct / max(total, 1)
        history.append({
            "epoch": epoch,
            "train_window_loss": train_loss,
            "train_window_accuracy": train_accuracy,
            "validation_trial_loss": validation["trial_loss"],
            "validation_trial_accuracy": validation["trial_accuracy"],
            "validation_trial_macro_f1": validation["trial_macro_f1"],
        })
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | train loss={train_loss:.4f} "
            f"window acc={train_accuracy:.4f} | val trial loss={validation['trial_loss']:.4f} "
            f"acc={validation['trial_accuracy']:.4f} macro-F1={validation['trial_macro_f1']:.4f}"
        )
        if validation["trial_loss"] < best_val_loss - 1e-4:
            best_val_loss = validation["trial_loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "best_epoch": best_epoch,
                "best_validation_trial_loss": best_val_loss,
                "seed": SEED,
            }, best_path)
            print(f"  Saved best validation checkpoint: {best_path}")
        else:
            stale_epochs += 1
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch} epochs without validation improvement.")
            break

    if best_epoch == 0:
        raise RuntimeError("No best validation checkpoint was saved")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics = evaluate(model, loaders["train"], device)
    validation_metrics = evaluate(model, loaders["validation"], device)
    # Test is first iterated once after validation-based checkpoint selection.
    test_metrics = evaluate(model, loaders["test"], device)

    print("\nFINAL METRICS (trial-aggregated primary results)")
    print(f"Best epoch: {best_epoch}; best validation trial loss={best_val_loss:.4f}")
    for name, metrics in (("train", train_metrics), ("validation", validation_metrics), ("test", test_metrics)):
        print(
            f"{name:10s}: trial accuracy={metrics['trial_accuracy']:.4f}, "
            f"balanced accuracy={metrics['trial_balanced_accuracy']:.4f}, "
            f"macro-F1={metrics['trial_macro_f1']:.4f}"
        )
    print("Test per-class precision / recall / F1 / support:")
    for index, emotion in enumerate(CLASS_NAMES):
        print(
            f"  {emotion:<8}: {test_metrics['trial_precision'][index]:.4f} / "
            f"{test_metrics['trial_recall'][index]:.4f} / {test_metrics['trial_f1'][index]:.4f} / "
            f"{int(test_metrics['trial_support'][index])}"
        )
    print("Test trial confusion matrix (rows=true, columns=predicted):")
    print(test_metrics["trial_confusion_matrix"])
    print("Secondary test window accuracy / macro-F1 (overlapping windows are correlated): "
          f"{test_metrics['window_accuracy']:.4f} / {test_metrics['window_macro_f1']:.4f}")

    config = {
        "model": "GASF-CNN baseline",
        "input_shape_per_window": [1, GASF_SIGNAL_SAMPLES, GASF_SIGNAL_SAMPLES],
        "raw_channel_index_zero_based": GASF_CHANNEL_INDEX,
        "sampling_rate_hz_assumption": SAMPLING_RATE_HZ,
        "bandpass_hz": [LOWCUT_HZ, HIGHCUT_HZ],
        "notch_hz": NOTCH_HZ,
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "gasf_signal_samples": GASF_SIGNAL_SAMPLES,
        "gasf_smoothing_sigma": GASF_SMOOTHING_SIGMA,
        "train_scale_percentiles": SCALE_PERCENTILES,
        "train_scale_low": metadata["train_scale_low"],
        "train_scale_high": metadata["train_scale_high"],
        "scale_source": "training trials only",
        "cnn_channels": [32, 64, 128, 256],
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "trial_assignments": TRIAL_ASSIGNMENTS,
        "warning": "Validation and test each have one independent trial per class.",
    }
    results = {
        "configuration": config,
        "best_epoch": best_epoch,
        "best_validation_trial_loss": best_val_loss,
        "training_minutes": (time.time() - start_time) / 60.0,
        "training": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "split_counts": metadata,
    }
    results_path = run_dir / "results.json"
    write_results(results, results_path)
    print(f"\nBest checkpoint: {best_path}")
    print(f"Results/configuration: {results_path}")
    print(f"Output directory: {run_dir}")
    print("Test set was evaluated once after validation-based checkpoint selection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true", help="Check GASF data and model shapes; do not train.")
    args = parser.parse_args()
    seed_everything()
    if args.check_data:
        datasets, metadata = prepare_data()
        print_summary(datasets, metadata)
        for split, dataset in datasets.items():
            image, label, _ = dataset[0]
            if tuple(image.shape) != (1, GASF_SIGNAL_SAMPLES, GASF_SIGNAL_SAMPLES):
                raise RuntimeError(f"Unexpected {split} GASF shape: {tuple(image.shape)}")
            if not torch.isfinite(image).all() or label.item() not in range(4):
                raise RuntimeError(f"Invalid GASF sample in {split}")
        output = GASFCNN()(torch.zeros(2, 1, GASF_SIGNAL_SAMPLES, GASF_SIGNAL_SAMPLES))
        if tuple(output.shape) != (2, 4):
            raise RuntimeError(f"Unexpected CNN output shape: {tuple(output.shape)}")
        print("Shape check passed: input=(batch, 1, 64, 64), output=(batch, 4). No training started.")
    else:
        train()


if __name__ == "__main__":
    main()

