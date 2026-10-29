"""Read-only multichannel EEG loading for the EEGNet + GCN experiment.

The module reads the original MAT recordings directly. It does not read,
modify, or create any of the project's existing processed datasets.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from scipy.io import loadmat
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "output"

CLASS_NAMES = ("fear", "happy", "neutral", "sad")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
MAT_LABELS = {"fear": 2, "happy": 3, "neutral": 0, "sad": 1}

# The supplied MAT files do not store a sampling-rate field. This is the
# sampling rate assumed by the existing project preprocessing and must be
# verified against the upstream dataset metadata.
SAMPLING_RATE_HZ = 200
BANDPASS_HZ = (1.0, 45.0)
FILTER_ORDER = 4
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(SAMPLING_RATE_HZ * WINDOW_SECONDS)
HOP_SECONDS = 0.5
HOP_SAMPLES = int(SAMPLING_RATE_HZ * HOP_SECONDS)
EXPECTED_CHANNELS = 62
SEED = 42

# Full trial stems from the already audited trial-level CRNN split.
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


@dataclass
class Trial:
    key: str
    emotion: str
    split: str
    signal: np.ndarray  # channels x samples, float32 after normalization


def _assignment_map():
    mapping = {}
    for split, classes in TRIAL_ASSIGNMENTS.items():
        for emotion, trial_keys in classes.items():
            for key in trial_keys:
                if key in mapping:
                    raise ValueError(f"Trial assigned more than once: {key}")
                mapping[key] = (split, emotion)
    if len(mapping) != 20:
        raise ValueError(f"Expected 20 assigned trial groups, found {len(mapping)}")
    return mapping


def _preprocess_trial(eeg, path):
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2 or eeg.shape[0] != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected EEG shape ({EXPECTED_CHANNELS}, time) in {path}; got {eeg.shape}"
        )
    if eeg.shape[1] < WINDOW_SAMPLES:
        raise ValueError(f"Recording is shorter than one window: {path}, shape={eeg.shape}")

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


def _load_trials():
    assignment = _assignment_map()
    trials = []
    seen = set()

    for emotion in CLASS_NAMES:
        class_dir = RAW_ROOT / emotion
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing raw emotion directory: {class_dir}")
        for path in sorted(class_dir.glob("*.mat")):
            key = path.stem
            if key not in assignment:
                raise ValueError(f"Raw trial is not in the fixed split: {path}")
            split, assigned_emotion = assignment[key]
            if assigned_emotion != emotion:
                raise ValueError(f"Trial folder disagrees with fixed assignment: {path}")
            if key in seen:
                raise ValueError(f"Duplicate trial key found: {key}")

            mat = loadmat(path)
            if "EEG" not in mat or "label" not in mat:
                raise KeyError(f"Expected EEG and label variables in {path}")
            raw_label = np.asarray(mat["label"]).squeeze()
            if raw_label.size != 1:
                raise ValueError(f"Expected one scalar label in {path}; got {raw_label.shape}")
            label_value = int(raw_label)
            if label_value != MAT_LABELS[emotion]:
                raise ValueError(
                    f"MAT label/folder mismatch in {path}: {label_value} != {MAT_LABELS[emotion]}"
                )

            eeg = _preprocess_trial(mat["EEG"], path)
            trials.append(Trial(key=key, emotion=emotion, split=split, signal=eeg))
            seen.add(key)

    if seen != set(assignment):
        raise ValueError(
            f"Raw files do not exactly match fixed assignments. Missing={sorted(set(assignment)-seen)}, "
            f"unexpected={sorted(seen-set(assignment))}"
        )
    return trials


def _training_normalization(trials):
    train_trials = [trial for trial in trials if trial.split == "train"]
    if not train_trials:
        raise ValueError("No training trials available for normalization")

    sums = np.zeros(EXPECTED_CHANNELS, dtype=np.float64)
    sums_sq = np.zeros(EXPECTED_CHANNELS, dtype=np.float64)
    count = 0
    for trial in train_trials:
        values = trial.signal.astype(np.float64, copy=False)
        sums += values.sum(axis=1)
        sums_sq += np.square(values).sum(axis=1)
        count += values.shape[1]

    mean = sums / count
    variance = np.maximum(sums_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), count


def _training_functional_graph(trials):
    """Build equal-trial-weighted absolute correlations from TRAIN trials only."""
    train_trials = [trial for trial in trials if trial.split == "train"]
    if not train_trials:
        raise ValueError("No training trials available for graph construction")

    correlations = []
    for trial in train_trials:
        matrix = np.corrcoef(trial.signal.astype(np.float64, copy=False))
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        correlations.append(np.abs(matrix))

    adjacency = np.mean(np.stack(correlations, axis=0), axis=0)
    adjacency = (adjacency + adjacency.T) / 2.0
    np.fill_diagonal(adjacency, 0.0)
    adjacency += np.eye(EXPECTED_CHANNELS, dtype=np.float64)
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.power(np.maximum(degree, 1e-12), -0.5)
    normalized = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    return normalized.astype(np.float32), (adjacency - np.eye(EXPECTED_CHANNELS)).astype(np.float32)


class EEGWindowDataset(Dataset):
    """Window view over trial arrays; overlapping windows are not duplicated in memory."""

    def __init__(self, trials, split, trial_indices):
        self.trials = [trial for trial in trials if trial.split == split]
        self.trial_indices = trial_indices
        self.index = []
        for trial in self.trials:
            last_start = trial.signal.shape[1] - WINDOW_SAMPLES
            for start in range(0, last_start + 1, HOP_SAMPLES):
                self.index.append((trial, start))
        if not self.index:
            raise ValueError(f"No EEG windows found for split {split}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        trial, start = self.index[index]
        window = trial.signal[:, start:start + WINDOW_SAMPLES]
        x = torch.from_numpy(np.ascontiguousarray(window[None, :, :]))
        y = torch.tensor(CLASS_TO_INDEX[trial.emotion], dtype=torch.long)
        trial_index = torch.tensor(self.trial_indices[trial.key], dtype=torch.long)
        return x, y, trial_index

    @property
    def window_counts_by_trial(self):
        counts = {}
        for trial, _ in self.index:
            counts[trial.key] = counts.get(trial.key, 0) + 1
        return counts


def prepare_data():
    """Load trials, calculate TRAIN-only statistics/graph, then expose windows."""
    trials = _load_trials()
    train_trials = [trial for trial in trials if trial.split == "train"]
    mean, std, train_sample_count = _training_normalization(train_trials)

    # The functional graph is computed from filtered TRAIN recordings only,
    # before normalization. Validation and test signals are never passed here.
    normalized_adjacency, raw_functional_adjacency = _training_functional_graph(train_trials)

    for trial in trials:
        trial.signal = ((trial.signal - mean[:, None]) / std[:, None]).astype(np.float32)

    ordered_trial_keys = sorted(trial.key for trial in trials)
    trial_indices = {key: index for index, key in enumerate(ordered_trial_keys)}
    datasets = {
        split: EEGWindowDataset(trials, split, trial_indices)
        for split in ("train", "validation", "test")
    }

    split_trial_keys = {
        split: {trial.key for trial in trials if trial.split == split}
        for split in ("train", "validation", "test")
    }
    intersections = {
        "train∩validation": split_trial_keys["train"] & split_trial_keys["validation"],
        "train∩test": split_trial_keys["train"] & split_trial_keys["test"],
        "validation∩test": split_trial_keys["validation"] & split_trial_keys["test"],
    }
    if any(intersections.values()):
        raise RuntimeError(f"Trial leakage detected: {intersections}")

    metadata = {
        "sampling_rate_hz_assumption": SAMPLING_RATE_HZ,
        "channels": EXPECTED_CHANNELS,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "normalization_source": "TRAIN trials only",
        "normalization_mean": mean,
        "normalization_std": std,
        "normalization_training_sample_count": train_sample_count,
        "graph_source": "absolute channel correlation; TRAIN trials only; equal weight per trial",
        "graph_train_trials": sorted(trial.key for trial in train_trials),
        "trial_assignments": TRIAL_ASSIGNMENTS,
        "split_trial_counts": {split: len(keys) for split, keys in split_trial_keys.items()},
        "split_window_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "split_class_window_counts": {
            split: {
                emotion: sum(
                    1 for trial, _ in dataset.index if trial.emotion == emotion
                )
                for emotion in CLASS_NAMES
            }
            for split, dataset in datasets.items()
        },
        "window_counts_by_trial": {
            split: datasets[split].window_counts_by_trial for split in datasets
        },
        "trial_intersections": {key: sorted(value) for key, value in intersections.items()},
    }
    return datasets, torch.from_numpy(normalized_adjacency), raw_functional_adjacency, metadata

