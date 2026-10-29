"""Rough raw-EEG temporal Transformer baseline with trial-level evaluation.

This standalone experiment reads original MAT recordings from data/output.
It does not modify or use any existing processed dataset. Use --check-data to
validate loading, fixed splits, and tensor shapes without starting training.
"""

import argparse
import json
import random
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
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
MODEL_ROOT = PROJECT_ROOT / "models" / "raw_eeg_transformer"

CLASS_NAMES = ("fear", "happy", "neutral", "sad")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
MAT_LABELS = {"fear": 2, "happy": 3, "neutral": 0, "sad": 1}
SPLITS = ("train", "validation", "test")

SEED = 42
SAMPLING_RATE_HZ = 200  # Assumed by the existing project; absent from MAT metadata.
CHANNELS = 62
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(SAMPLING_RATE_HZ * WINDOW_SECONDS)
HOP_SECONDS = 0.5
HOP_SAMPLES = int(SAMPLING_RATE_HZ * HOP_SECONDS)
BANDPASS_HZ = (1.0, 45.0)
FILTER_ORDER = 4

# Rough baseline settings, following the existing Transformer script where
# applicable. No augmentation or hyperparameter search is performed.
BATCH_SIZE = 32
MAX_EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 10
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


def seed_everything(seed=SEED):
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


def assignment_map():
    result = {}
    for split, class_trials in TRIAL_ASSIGNMENTS.items():
        for emotion, keys in class_trials.items():
            for key in keys:
                if key in result:
                    raise ValueError(f"Trial assigned to multiple splits: {key}")
                result[key] = (split, emotion)
    if len(result) != 20:
        raise ValueError(f"Expected exactly 20 trial assignments, found {len(result)}")
    return result


def preprocess(eeg, path):
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2 or eeg.shape[0] != CHANNELS:
        raise ValueError(f"Expected ({CHANNELS}, time) EEG in {path}; got {eeg.shape}")
    if eeg.shape[1] < WINDOW_SAMPLES:
        raise ValueError(f"Recording shorter than a 2-second window: {path}")

    eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
    eeg -= eeg.mean(axis=1, keepdims=True)
    nyquist = SAMPLING_RATE_HZ / 2.0
    b, a = signal.butter(
        FILTER_ORDER,
        (BANDPASS_HZ[0] / nyquist, BANDPASS_HZ[1] / nyquist),
        btype="bandpass",
    )
    eeg = signal.filtfilt(b, a, eeg, axis=-1)
    eeg = signal.detrend(eeg, axis=-1, type="linear")
    return eeg.astype(np.float32)


def load_trials():
    assignments = assignment_map()
    trials = []
    seen = set()
    for emotion in CLASS_NAMES:
        class_dir = RAW_ROOT / emotion
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing raw EEG directory: {class_dir}")
        for path in sorted(class_dir.glob("*.mat")):
            key = path.stem
            if key not in assignments:
                raise ValueError(f"Raw recording is absent from the fixed split: {path}")
            split, assigned_emotion = assignments[key]
            if emotion != assigned_emotion:
                raise ValueError(f"Folder/split class mismatch for {path}")
            mat = loadmat(path)
            if "EEG" not in mat or "label" not in mat:
                raise KeyError(f"Expected EEG and label variables in {path}")
            label = np.asarray(mat["label"]).squeeze()
            if label.size != 1 or int(label) != MAT_LABELS[emotion]:
                raise ValueError(f"MAT label mismatch for {path}: {label}")
            trials.append({
                "key": key,
                "emotion": emotion,
                "split": split,
                "signal": preprocess(mat["EEG"], path),
            })
            seen.add(key)

    if seen != set(assignments):
        raise ValueError(
            f"Raw trial coverage mismatch; missing={sorted(set(assignments)-seen)}, "
            f"unexpected={sorted(seen-set(assignments))}"
        )
    return trials


def training_normalization(trials):
    """Per-channel z-score values estimated from training recordings only."""
    train_trials = [trial for trial in trials if trial["split"] == "train"]
    sums = np.zeros(CHANNELS, dtype=np.float64)
    sums_sq = np.zeros(CHANNELS, dtype=np.float64)
    sample_count = 0
    for trial in train_trials:
        values = trial["signal"].astype(np.float64, copy=False)
        sums += values.sum(axis=1)
        sums_sq += np.square(values).sum(axis=1)
        sample_count += values.shape[1]
    mean = sums / sample_count
    variance = np.maximum(sums_sq / sample_count - mean ** 2, 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), sample_count


class WindowDataset(Dataset):
    """On-demand overlapping windows with a trial index for grouped scoring."""

    def __init__(self, trials, split, trial_index_map):
        self.trials = [trial for trial in trials if trial["split"] == split]
        self.trial_index_map = trial_index_map
        self.index = []
        for trial in self.trials:
            last_start = trial["signal"].shape[1] - WINDOW_SAMPLES
            for start in range(0, last_start + 1, HOP_SAMPLES):
                self.index.append((trial, start))
        if not self.index:
            raise ValueError(f"No windows in {split} split")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        trial, start = self.index[index]
        x = trial["signal"][:, start:start + WINDOW_SAMPLES]
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.tensor(CLASS_TO_INDEX[trial["emotion"]], dtype=torch.long),
            torch.tensor(self.trial_index_map[trial["key"]], dtype=torch.long),
        )

    @property
    def window_counts_by_trial(self):
        counts = {}
        for trial, _ in self.index:
            counts[trial["key"]] = counts.get(trial["key"], 0) + 1
        return counts


def prepare_data():
    trials = load_trials()
    train_trials = [trial for trial in trials if trial["split"] == "train"]
    mean, std, sample_count = training_normalization(train_trials)
    for trial in trials:
        trial["signal"] = ((trial["signal"] - mean[:, None]) / std[:, None]).astype(np.float32)

    trial_keys = sorted(trial["key"] for trial in trials)
    trial_index_map = {key: index for index, key in enumerate(trial_keys)}
    datasets = {split: WindowDataset(trials, split, trial_index_map) for split in SPLITS}
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
        "normalization_mean": mean,
        "normalization_std": std,
        "normalization_source": "training trials only",
        "normalization_sample_count": sample_count,
        "split_trial_counts": {split: len(keys) for split, keys in split_trials.items()},
        "split_window_counts": {split: len(ds) for split, ds in datasets.items()},
        "split_class_counts": {
            split: {
                emotion: sum(1 for trial, _ in ds.index if trial["emotion"] == emotion)
                for emotion in CLASS_NAMES
            }
            for split, ds in datasets.items()
        },
        "window_counts_by_trial": {
            split: datasets[split].window_counts_by_trial for split in SPLITS
        },
        "trial_assignments": TRIAL_ASSIGNMENTS,
        "trial_overlap": {key: sorted(value) for key, value in overlap.items()},
    }
    return datasets, metadata


class RawEEGTransformer(nn.Module):
    """Temporal Transformer over non-overlapping 125 ms multichannel patches."""

    def __init__(self, channels=CHANNELS, embedding_dim=64, heads=4, layers=2, num_classes=4):
        super().__init__()
        self.patch_size = 25  # 125 ms under the assumed 200 Hz sampling rate.
        self.patch_embedding = nn.Conv1d(
            channels, embedding_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.num_tokens = WINDOW_SAMPLES // self.patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        # x: batch x 62 channels x 400 samples
        tokens = self.patch_embedding(x).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = self.encoder(tokens + self.position_embedding)
        return self.classifier(self.norm(tokens[:, 0]))


def trial_balanced_weights(dataset):
    counts = dataset.window_counts_by_trial
    class_trial_counts = {
        emotion: sum(trial["emotion"] == emotion for trial in dataset.trials)
        for emotion in CLASS_NAMES
    }
    sample_weights = []
    trial_weight = {}
    for trial, _ in dataset.index:
        value = 1.0 / (class_trial_counts[trial["emotion"]] * counts[trial["key"]])
        trial_weight[trial["key"]] = value
        sample_weights.append(value)
    scale = float(np.mean(sample_weights))
    return {key: value / scale for key, value in trial_weight.items()}, class_trial_counts


def evaluate(model, loader, device):
    model.eval()
    logits_all, labels_all, trial_ids_all = [], [], []
    with torch.no_grad():
        for x, labels, trial_ids in loader:
            logits_all.append(model(x.to(device)).cpu())
            labels_all.append(labels.cpu())
            trial_ids_all.append(trial_ids.cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    trial_ids = torch.cat(trial_ids_all)

    trial_logits, trial_labels = [], []
    for trial_id in torch.unique(trial_ids, sorted=True):
        mask = trial_ids == trial_id
        unique_labels = torch.unique(labels[mask])
        if unique_labels.numel() != 1:
            raise RuntimeError("A trial contains inconsistent emotion labels")
        trial_logits.append(logits[mask].mean(dim=0))
        trial_labels.append(unique_labels[0])
    trial_logits = torch.stack(trial_logits)
    trial_labels = torch.stack(trial_labels)
    target = trial_labels.numpy()
    predicted = trial_logits.argmax(1).numpy()
    window_target = labels.numpy()
    window_predicted = logits.argmax(1).numpy()
    p, r, f1, support = precision_recall_fscore_support(
        target, predicted, labels=range(4), zero_division=0
    )
    return {
        "trial_loss": F.cross_entropy(trial_logits, trial_labels).item(),
        "trial_accuracy": accuracy_score(target, predicted),
        "trial_balanced_accuracy": balanced_accuracy_score(target, predicted),
        "trial_macro_f1": f1_score(target, predicted, labels=range(4), average="macro", zero_division=0),
        "trial_precision": p,
        "trial_recall": r,
        "trial_f1": f1,
        "trial_support": support,
        "trial_confusion_matrix": confusion_matrix(target, predicted, labels=range(4)),
        "trial_report": classification_report(
            target, predicted, labels=range(4), target_names=CLASS_NAMES, digits=4, zero_division=0
        ),
        "window_accuracy": accuracy_score(window_target, window_predicted),
        "window_macro_f1": f1_score(
            window_target, window_predicted, labels=range(4), average="macro", zero_division=0
        ),
        "window_confusion_matrix": confusion_matrix(window_target, window_predicted, labels=range(4)),
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


def write_results(results, result_path):
    """Serialize available split metrics without assuming optional report fields."""
    serializable = dict(results)
    for split_name in ("training", "validation", "test"):
        metrics = results.get(split_name)
        if not isinstance(metrics, dict):
            continue
        split_metrics = dict(metrics)
        report = split_metrics.pop("trial_report", None)
        if report is not None:
            split_metrics.setdefault("trial_report_text", report)
        serializable[split_name] = split_metrics
    Path(result_path).write_text(
        json.dumps(json_ready(serializable), indent=2), encoding="utf-8"
    )


def check_result_writer():
    """Test result serialization in a temporary directory, without training."""
    fixture = {
        "configuration": {"model": "writer-check fixture"},
        "training": {"trial_accuracy": 1.0, "trial_report": "training fixture report"},
        "validation": {"trial_accuracy": 0.5},
        "test": {"trial_accuracy": 0.5, "trial_report": "test fixture report"},
    }
    original = json.loads(json.dumps(fixture))
    with tempfile.TemporaryDirectory(prefix="raw-eeg-transformer-report-check-") as temp_dir:
        result_path = Path(temp_dir) / "results.json"
        write_results(fixture, result_path)
        saved = json.loads(result_path.read_text(encoding="utf-8"))

    if fixture != original:
        raise RuntimeError("Result writer unexpectedly mutated input metrics")
    if saved["training"].get("trial_accuracy") != 1.0:
        raise RuntimeError("Training metric was not preserved")
    if saved["training"].get("trial_report_text") != "training fixture report":
        raise RuntimeError("Available training report was not preserved")
    if saved["validation"].get("trial_accuracy") != 0.5:
        raise RuntimeError("Validation metric was not preserved")
    if "trial_report_text" in saved["validation"]:
        raise RuntimeError("A missing validation report was invented")
    if saved["test"].get("trial_report_text") != "test fixture report":
        raise RuntimeError("Available test report was not preserved")
    print("RESULT WRITER CHECK PASSED: actual split keys handled and available metrics preserved.")
    print("Missing reports remain absent; no training or test evaluation was run.")


def print_data_summary(datasets, metadata):
    print(f"Dataset: {RAW_ROOT} (raw MAT recordings)")
    print(f"Class mapping: {CLASS_TO_INDEX}")
    print(
        f"Input: {CHANNELS} channels x {WINDOW_SAMPLES} samples; "
        f"window={WINDOW_SECONDS}s, hop={HOP_SECONDS}s"
    )
    print(f"Sampling rate: {SAMPLING_RATE_HZ} Hz (assumption; not stored in MAT files)")
    print("Split counts:")
    for split in SPLITS:
        print(
            f"  {split:10s}: trials={metadata['split_trial_counts'][split]}, "
            f"windows={len(datasets[split])}, classes={metadata['split_class_counts'][split]}"
        )
    print(f"Trial overlap check: {metadata['trial_overlap']}")


def check_data():
    datasets, metadata = prepare_data()
    print_data_summary(datasets, metadata)
    for split, dataset in datasets.items():
        x, y, _ = dataset[0]
        if tuple(x.shape) != (CHANNELS, WINDOW_SAMPLES) or y.item() not in range(4):
            raise RuntimeError(f"Unexpected {split} sample shape/label: {tuple(x.shape)}, {y.item()}")
    model = RawEEGTransformer()
    with torch.no_grad():
        output = model(torch.zeros(2, CHANNELS, WINDOW_SAMPLES))
    if tuple(output.shape) != (2, 4):
        raise RuntimeError(f"Unexpected Transformer output shape: {tuple(output.shape)}")
    print("Shape check passed: input=(batch, 62, 400), output=(batch, 4). No training started.")


def train():
    seed_everything()
    device = select_device()
    datasets, metadata = prepare_data()
    print("RAW EEG TRANSFORMER BASELINE")
    print_data_summary(datasets, metadata)
    print(f"Device: {device}")
    print("Warnings: channel names/order and electrode coordinates are absent from the MAT metadata.")
    print("Warnings: validation and test each contain only one independent trial per class.")

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        datasets["train"], batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=NUM_WORKERS
    )
    eval_loaders = {
        split: DataLoader(datasets[split], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        for split in SPLITS
    }
    trial_weights, class_trial_counts = trial_balanced_weights(datasets["train"])
    trial_weight_by_index = {
        datasets["train"].trial_index_map[key]: value
        for key, value in trial_weights.items()
    }
    print(f"Training trial counts by class: {class_trial_counts}")
    print("Training loss weighting: equal total weight per class and per trial within class.")

    model = RawEEGTransformer().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE
    )

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = MODEL_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    best_path = run_dir / "best_raw_eeg_transformer.pth"
    start_time = time.time()
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    print("\nTraining: validation trial-aggregated loss selects the checkpoint.")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        correct = 0
        count = 0
        for x, labels, trial_ids in train_loader:
            x, labels = x.to(device), labels.to(device)
            weights = torch.tensor(
                [trial_weight_by_index[int(index)] for index in trial_ids.tolist()],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            losses = criterion(logits, labels)
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float((losses.detach() * weights).sum())
            total_weight += float(weights.sum())
            correct += int((logits.argmax(1) == labels).sum())
            count += labels.numel()

        validation = evaluate(model, eval_loaders["validation"], device)
        scheduler.step(validation["trial_loss"])
        train_loss = total_loss / max(total_weight, 1e-12)
        train_accuracy = correct / max(count, 1)
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
            f"acc={train_accuracy:.4f} | val trial loss={validation['trial_loss']:.4f} "
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
            print(f"  Saved best checkpoint: {best_path}")
        else:
            stale_epochs += 1
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch} epochs without validation improvement.")
            break

    if not best_epoch:
        raise RuntimeError("No best validation checkpoint was produced")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Report train and validation metrics from the selected checkpoint.
    train_metrics = evaluate(model, eval_loaders["train"], device)
    validation_metrics = evaluate(model, eval_loaders["validation"], device)
    # Held-out test loader is evaluated exactly once, after checkpoint selection.
    test_metrics = evaluate(model, eval_loaders["test"], device)

    print("\nFINAL RESULTS (primary metrics aggregate windows to one prediction per trial)")
    print(f"Best epoch: {best_epoch}; best validation trial loss: {best_val_loss:.4f}")
    for split, metrics in (
        ("training", train_metrics),
        ("validation", validation_metrics),
        ("test", test_metrics),
    ):
        print(
            f"{split.capitalize():10s}: trial accuracy={metrics['trial_accuracy']:.4f}, "
            f"balanced accuracy={metrics['trial_balanced_accuracy']:.4f}, "
            f"macro F1={metrics['trial_macro_f1']:.4f}"
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
    print("Secondary test window accuracy / macro F1 (overlapping windows are correlated): "
          f"{test_metrics['window_accuracy']:.4f} / {test_metrics['window_macro_f1']:.4f}")

    config = {
        "model": "Raw EEG temporal Transformer",
        "input_shape_per_window": [CHANNELS, WINDOW_SAMPLES],
        "patch_samples": 25,
        "embedding_dim": 64,
        "attention_heads": 4,
        "transformer_layers": 2,
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "sampling_rate_hz_assumption": SAMPLING_RATE_HZ,
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "bandpass_hz": BANDPASS_HZ,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "normalization_source": "training trials only",
        "trial_assignments": TRIAL_ASSIGNMENTS,
        "warning": "Only four independent trials are in validation and four in test.",
    }
    results = {
        "configuration": config,
        "best_epoch": best_epoch,
        "best_validation_trial_loss": best_val_loss,
        "training": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "training_minutes": (time.time() - start_time) / 60.0,
    }
    result_path = run_dir / "results.json"
    write_results(results, result_path)
    if not best_path.is_file():
        raise RuntimeError("Best model checkpoint missing")
    print(f"\nOutput directory: {run_dir}")
    print(f"Best checkpoint: {best_path}")
    print(f"Results/configuration: {result_path}")
    print(f"Training duration: {results['training_minutes']:.2f} minutes")
    print("Held-out test evaluation completed once after validation-based model selection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true", help="Validate data and shapes without training.")
    parser.add_argument(
        "--check-reporting",
        action="store_true",
        help="Test result writing with a temporary fixture; does not train or evaluate data.",
    )
    args = parser.parse_args()
    seed_everything()
    if args.check_reporting:
        check_result_writer()
    elif args.check_data:
        datasets, metadata = prepare_data()
        print_data_summary(datasets, metadata)
        for split, dataset in datasets.items():
            x, y, _ = dataset[0]
            if tuple(x.shape) != (CHANNELS, WINDOW_SAMPLES) or y.item() not in range(4):
                raise RuntimeError(f"Invalid {split} tensor or label: {tuple(x.shape)}, {y.item()}")
        with torch.no_grad():
            output = RawEEGTransformer()(torch.zeros(2, CHANNELS, WINDOW_SAMPLES))
        if tuple(output.shape) != (2, 4):
            raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
        print("Shape check passed: input=(batch, 62, 400), output=(batch, 4). No training started.")
    else:
        train()


if __name__ == "__main__":
    main()

