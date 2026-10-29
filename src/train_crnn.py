"""Train the baseline CRNN architecture on the fixed-scale trial-level data."""

import copy
import hashlib
import inspect
import json
import os
import random
import re
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "processed", "crnn_dataset_fixedscale_1024")
PARAMETERS_PATH = os.path.join(
    PROJECT_ROOT, "data", "diagnostics", "fixedscale_1024_experiment", "scaling_parameters.json"
)
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models", "fixedscale_1024_experiment")
DIAGNOSTIC_ROOT = os.path.join(PROJECT_ROOT, "data", "diagnostics", "fixedscale_1024_experiment")

# Baseline training settings from the run that produced 31.58% test accuracy.
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.001
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 7
SCHEDULER_PATIENCE = 2
GRADIENT_CLIP_NORM = 1.0
SEED = 42
CLASS_NAMES = ["fear", "happy", "neutral", "sad"]
SPLITS = ("train", "validation", "test")
TRIAL_PATTERN = re.compile(r"^(?P<trial>.+_trial\d+)_window_\d+_STFT_RGB$", re.IGNORECASE)
BASELINE_TEST_ACCURACY = 0.3158
GLOBAL_SCALE_TEST_ACCURACY = 0.2775


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def trial_key(path):
    match = TRIAL_PATTERN.fullmatch(os.path.splitext(os.path.basename(path))[0])
    if not match:
        raise ValueError(f"Cannot parse trial key from filename: {path}")
    return match.group("trial")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name, condition, details=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {details}" if details else ""))
    if not condition:
        raise RuntimeError(f"Dataset sanity check failed: {name}. {details}")


print("=" * 76)
print("FIXED-SCALE CRNN EXPERIMENT — BASELINE ARCHITECTURE/SETTINGS")
print("=" * 76)
print(f"Dataset path: {DATASET_ROOT}")
print(f"Scale parameters: {PARAMETERS_PATH}")
print(f"Seed: {SEED}")
print(f"Selected device: {DEVICE}")
print("Test set is excluded from optimization and model selection.")

if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(f"Missing fixed-scale dataset: {DATASET_ROOT}")
if not os.path.isfile(PARAMETERS_PATH):
    raise FileNotFoundError(f"Missing train-only scale parameters: {PARAMETERS_PATH}")
with open(PARAMETERS_PATH, encoding="utf-8") as parameter_file:
    scale_parameters = json.load(parameter_file)
check(
    "Scaling metadata confirms train-only statistics",
    "TRAIN trials only" in scale_parameters.get("scale_source", "")
    and scale_parameters.get("clip_percentiles") == [1.0, 99.0],
    f"range={scale_parameters.get('scale_min_db')} to {scale_parameters.get('scale_max_db')} dB",
)

datasets_by_split = {}
for split in SPLITS:
    split_dir = os.path.join(DATASET_ROOT, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    datasets_by_split[split] = datasets.ImageFolder(split_dir, transform=TRANSFORM)

class_mapping = datasets_by_split["train"].class_to_idx
check("Four expected classes in requested order", datasets_by_split["train"].classes == CLASS_NAMES, str(class_mapping))
check(
    "Class mapping identical across train/validation/test",
    all(dataset.class_to_idx == class_mapping for dataset in datasets_by_split.values()),
)
for split, dataset in datasets_by_split.items():
    labels_present = {label for _, label in dataset.samples}
    check(f"All classes and valid labels in {split}", labels_present == set(range(4)))

image_keys = {
    split: {(dataset.classes[label], os.path.basename(path)) for path, label in dataset.samples}
    for split, dataset in datasets_by_split.items()
}
image_overlaps = {
    (a, b): image_keys[a] & image_keys[b]
    for i, a in enumerate(SPLITS)
    for b in SPLITS[i + 1:]
}
check("No source image occurs in multiple splits", not any(image_overlaps.values()), str({str(k): len(v) for k, v in image_overlaps.items()}))

trials = {
    split: {(dataset.classes[label], trial_key(path)) for path, label in dataset.samples}
    for split, dataset in datasets_by_split.items()
}
trial_overlaps = {
    (a, b): trials[a] & trials[b]
    for i, a in enumerate(SPLITS)
    for b in SPLITS[i + 1:]
}
check("No trial appears in multiple splits", not any(trial_overlaps.values()), str({str(k): len(v) for k, v in trial_overlaps.items()}))

hashes = {
    split: {sha256(path) for path, _ in dataset.samples}
    for split, dataset in datasets_by_split.items()
}
hash_overlaps = {
    (a, b): hashes[a] & hashes[b]
    for i, a in enumerate(SPLITS)
    for b in SPLITS[i + 1:]
}
check("No exact duplicate image across splits", not any(hash_overlaps.values()), str({str(k): len(v) for k, v in hash_overlaps.items()}))

sizes, modes = set(), set()
for dataset in datasets_by_split.values():
    for path, _ in dataset.samples:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            sizes.add(image.size)
            modes.add(image.mode)
check("Consistent valid image dimensions/channels", len(sizes) == 1 and len(modes) == 1, f"sizes={sizes}, modes={modes}")
transform_is_deterministic = not any(
    type(transform).__name__.startswith("Random") for transform in TRANSFORM.transforms
)
check("Validation/test transform deterministic", transform_is_deterministic)
print("[PASS] Train/validation/test transformations are identical deterministic resize/tensor/normalize operations.")

print("\nCLASS MAPPING")
for index, name in enumerate(CLASS_NAMES):
    print(f"  {index} -> {name}")
print("\nDATASET COUNTS")
counts_by_split = {}
for split, dataset in datasets_by_split.items():
    counts = np.bincount([label for _, label in dataset.samples], minlength=4)
    counts_by_split[split] = counts.tolist()
    print(f"  {split:<10}: total={len(dataset)}; " + ", ".join(f"{name}={counts[i]}" for i, name in enumerate(CLASS_NAMES)))


class EEG_CRNN(nn.Module):
    """Architecture matching the saved 31.58% baseline checkpoint."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        self.sequence_projection = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        features = self.cnn(images)
        height = features.shape[2]
        bands = []
        for band_index in range(4):
            start = (band_index * height) // 4
            end = ((band_index + 1) * height + 3) // 4
            bands.append(features[:, :, start:end, :].mean(dim=2))
        sequence = torch.stack(bands, dim=2)
        sequence = sequence.permute(0, 3, 2, 1).flatten(start_dim=2)
        sequence = self.sequence_projection(sequence)
        _, (hidden, _) = self.lstm(sequence)
        final_state = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(final_state)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum = 0.0
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return loss_sum / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total


generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(datasets_by_split["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=generator)
validation_loader = DataLoader(datasets_by_split["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(datasets_by_split["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
train_eval_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=TRANSFORM)
train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

train_source = inspect.getsource(train_one_epoch)
check("Test loader is not referenced by the optimization function", "test_loader" not in train_source)
print("[PASS] Test loader is evaluated once after validation-based model selection.")

train_counts = np.asarray(counts_by_split["train"], dtype=np.float64)
class_weights = len(datasets_by_split["train"]) / (len(CLASS_NAMES) * train_counts)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE),
    label_smoothing=LABEL_SMOOTHING,
)
print("\nTrain-only class weights:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<8}: count={int(train_counts[i])}, weight={class_weights[i]:.4f}")

model = EEG_CRNN(num_classes=4).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6
)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_run_dir = os.path.join(MODEL_ROOT, f"run_{run_id}")
diagnostic_run_dir = os.path.join(DIAGNOSTIC_ROOT, f"run_{run_id}")
os.makedirs(model_run_dir, exist_ok=False)
os.makedirs(diagnostic_run_dir, exist_ok=False)
best_model_path = os.path.join(model_run_dir, "best_crnn_fixedscale.pth")
final_model_path = os.path.join(model_run_dir, "final_crnn_fixedscale.pth")

history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
best_validation_accuracy = -1.0
best_epoch = 0
no_improvement = 0
start_time = time.time()

print("\nBASELINE SETTINGS")
print(f"IMAGE_SIZE={IMAGE_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")
print(f"LEARNING_RATE={LEARNING_RATE}, WEIGHT_DECAY={WEIGHT_DECAY}, label_smoothing={LABEL_SMOOTHING}")
print(f"Early stopping patience={EARLY_STOPPING_PATIENCE}; scheduler patience={SCHEDULER_PATIENCE}")
print("\nTRAINING — validation selects the checkpoint; test remains untouched")

for epoch in range(EPOCHS):
    epoch_start = time.time()
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion)
    validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion)
    scheduler.step(validation_accuracy)
    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)
    history["train_accuracy"].append(train_accuracy)
    history["validation_accuracy"].append(validation_accuracy)
    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | train loss={train_loss:.4f} acc={train_accuracy:.4f} | "
        f"val loss={validation_loss:.4f} acc={validation_accuracy:.4f} | {time.time()-epoch_start:.1f}s"
    )
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_epoch = epoch + 1
        no_improvement = 0
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(
            {
                "model_state_dict": best_state,
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "best_epoch": best_epoch,
                "validation_accuracy": best_validation_accuracy,
                "seed": SEED,
            },
            best_model_path,
        )
        print(f"  Best checkpoint saved: {best_model_path}")
    else:
        no_improvement += 1
        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch + 1} epochs without validation improvement.")
            break

training_seconds = time.time() - start_time
best_checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(best_checkpoint["model_state_dict"])

# Report train accuracy for the selected checkpoint with the deterministic, unaugmented transform.
_, training_accuracy = evaluate(model, train_eval_loader, criterion)
validation_accuracy = best_validation_accuracy

# Single held-out test evaluation, after model selection is complete.
model.eval()
test_labels, test_predictions = [], []
with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images.to(DEVICE))
        test_predictions.extend(logits.argmax(1).cpu().numpy().tolist())
        test_labels.extend(labels.numpy().tolist())

test_labels = np.asarray(test_labels)
test_predictions = np.asarray(test_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)
macro_precision = precision_score(test_labels, test_predictions, labels=range(4), average="macro", zero_division=0)
macro_recall = recall_score(test_labels, test_predictions, labels=range(4), average="macro", zero_division=0)
macro_f1 = f1_score(test_labels, test_predictions, labels=range(4), average="macro", zero_division=0)
weighted_f1 = f1_score(test_labels, test_predictions, labels=range(4), average="weighted", zero_division=0)
report = classification_report(
    test_labels, test_predictions, labels=range(4), target_names=CLASS_NAMES, digits=4, zero_division=0
)
cm = confusion_matrix(test_labels, test_predictions, labels=range(4))

print("\nFINAL FIXED-SCALE RESULTS")
print(f"Training accuracy:   {training_accuracy:.4f}")
print(f"Validation accuracy: {validation_accuracy:.4f} (best epoch {best_epoch})")
print(f"Test accuracy:       {test_accuracy:.4f}")
print(f"Macro precision:     {macro_precision:.4f}")
print(f"Macro recall:        {macro_recall:.4f}")
print(f"Macro F1:            {macro_f1:.4f}")
print(f"Weighted F1:         {weighted_f1:.4f}")
print("\nPer-class precision / recall / F1:")
print(report)
print("Confusion matrix (rows=actual, columns=predicted):")
print(cm)
print("Per-emotion accuracy:")
for i, emotion in enumerate(CLASS_NAMES):
    class_total = cm[i].sum()
    print(f"  {emotion:<8}: {cm[i, i]/class_total:.4f}" if class_total else f"  {emotion:<8}: N/A")

comparison = {
    "original_test_accuracy": BASELINE_TEST_ACCURACY,
    "global_scale_test_accuracy": GLOBAL_SCALE_TEST_ACCURACY,
    "fixedscale_test_accuracy": float(test_accuracy),
    "delta_vs_original": float(test_accuracy - BASELINE_TEST_ACCURACY),
    "delta_vs_global": float(test_accuracy - GLOBAL_SCALE_TEST_ACCURACY),
    "interpretation": "improved over original" if test_accuracy > BASELINE_TEST_ACCURACY else "did not improve over original",
}
report_path = os.path.join(diagnostic_run_dir, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write("FIXED-SCALE CRNN HELD-OUT TEST REPORT\n")
    report_file.write(f"Dataset: {DATASET_ROOT}\nSeed: {SEED}\nDevice: {DEVICE}\nBest epoch: {best_epoch}\n")
    report_file.write(f"Training accuracy: {training_accuracy:.6f}\nValidation accuracy: {validation_accuracy:.6f}\nTest accuracy: {test_accuracy:.6f}\n")
    report_file.write(f"Macro precision: {macro_precision:.6f}\nMacro recall: {macro_recall:.6f}\nMacro F1: {macro_f1:.6f}\nWeighted F1: {weighted_f1:.6f}\n\n")
    report_file.write(report)
    report_file.write("\nConfusion matrix (rows=actual, columns=predicted):\n")
    report_file.write(np.array2string(cm))
    report_file.write("\n\nComparison with earlier results:\n")
    report_file.write(json.dumps(comparison, indent=2))

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4), CLASS_NAMES, rotation=30, ha="right")
ax.set_yticks(range(4), CLASS_NAMES)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Fixed-scale CRNN confusion matrix (test)")
for row in range(4):
    for col in range(4):
        ax.text(col, row, str(cm[row, col]), ha="center", va="center")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(diagnostic_run_dir, "confusion_matrix.png"), dpi=160)
plt.close(fig)

epochs_done = np.arange(1, len(history["train_loss"]) + 1)
for metric, ylabel, filename in (
    ("loss", "Loss", "training_validation_loss.png"),
    ("accuracy", "Accuracy", "training_validation_accuracy.png"),
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_done, history[f"train_{metric}"], marker="o", label="Train")
    ax.plot(epochs_done, history[f"validation_{metric}"], marker="o", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Training and validation {metric}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(diagnostic_run_dir, filename), dpi=160)
    plt.close(fig)

torch.save(
    {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "best_epoch": best_epoch,
        "training_accuracy": float(training_accuracy),
        "validation_accuracy": float(validation_accuracy),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "seed": SEED,
    },
    final_model_path,
)
with open(os.path.join(diagnostic_run_dir, "results.json"), "w", encoding="utf-8") as result_file:
    json.dump(
        {
            "seed": SEED,
            "best_epoch": best_epoch,
            "training_accuracy": float(training_accuracy),
            "validation_accuracy": float(validation_accuracy),
            "test_accuracy": float(test_accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "class_counts": counts_by_split,
            "comparison": comparison,
        },
        result_file,
        indent=2,
    )

print("\nEXPERIMENT COMPARISON")
print(f"Original baseline test accuracy: {BASELINE_TEST_ACCURACY:.2%}")
print(f"Global-scale test accuracy:      {GLOBAL_SCALE_TEST_ACCURACY:.2%}")
print(f"Fixed-scale test accuracy:       {test_accuracy:.2%}")
print(f"Difference vs original:         {test_accuracy - BASELINE_TEST_ACCURACY:+.2%}")
print(f"Interpretation: {comparison['interpretation']}")
print("\nOUTPUTS")
print(f"Best model: {best_model_path}")
print(f"Final model: {final_model_path}")
print(f"Classification report and plots: {diagnostic_run_dir}")
print(f"Training duration: {training_seconds / 60:.2f} minutes")
print("Test set evaluated once, after validation-based checkpoint selection.")
 