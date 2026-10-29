"""Inference-only components for the frozen four-model EEG fusion pipeline.

The frozen configuration controls model inputs, checkpoints, class order,
trial assignments, preprocessing, and adaptive fusion. This module never
trains a base model and verifies only the requested split's input assets.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)
from torchvision import transforms

import eegnet_gcn_data as gcn_data
import train_raw_eeg_transformer as raw_transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG_PATH = PROJECT_ROOT / "models" / "fusion" / "frozen_ml_config.json"
FROZEN_CONFIG = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
MODEL_CONFIGS = FROZEN_CONFIG["models"]
CLASS_NAMES = list(FROZEN_CONFIG["class_order"])
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLITS = tuple(FROZEN_CONFIG["splits"])
MODEL_NAMES = tuple(FROZEN_CONFIG["model_order"])
TRIAL_ASSIGNMENTS = {
    split: {
        emotion: tuple(trials)
        for emotion, trials in assignments.items()
    }
    for split, assignments in FROZEN_CONFIG["splits"].items()
}
FUSION_WEIGHTS = dict(FROZEN_CONFIG["fusion"]["equal_weight_baseline"]["weights"])

RAW_ROOT = PROJECT_ROOT / FROZEN_CONFIG["input_assets"]["raw_eeg"]["root"]
STFT_DATASET_ROOT = PROJECT_ROOT / FROZEN_CONFIG["input_assets"]["stft_images"]["root"]
GCN_CHECKPOINT = PROJECT_ROOT / MODEL_CONFIGS["EEGNet + GCN"]["checkpoint"]["path"]
GCN_METADATA = PROJECT_ROOT / MODEL_CONFIGS["EEGNet + GCN"]["preprocessing"]["normalization_graph_metadata"]["path"]
RAW_TRANSFORMER_CHECKPOINT = PROJECT_ROOT / MODEL_CONFIGS["Raw EEG Transformer"]["checkpoint"]["path"]
CRNN_CHECKPOINT = PROJECT_ROOT / MODEL_CONFIGS["STFT CRNN"]["checkpoint"]["path"]
STFT_TRANSFORMER_CHECKPOINT = PROJECT_ROOT / MODEL_CONFIGS["STFT Transformer"]["checkpoint"]["path"]
IMAGE_SIZE = int(MODEL_CONFIGS["STFT CRNN"]["preprocessing"]["resize_px"][0])
BATCH_SIZE = int(FROZEN_CONFIG["prediction_aggregation"]["inference_batch_size"])
TRIAL_FILE_RE = re.compile(
    r"^(?P<trial>.+_trial\d+)_window_(?P<window>\d+)_STFT_RGB$",
    re.IGNORECASE,
)


def _canonical_assignments(assignments):
    return {
        split: {
            emotion: tuple(trials)
            for emotion, trials in class_trials.items()
        }
        for split, class_trials in assignments.items()
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file_hash(path, expected, description):
    if not path.is_file():
        raise FileNotFoundError(f"Frozen {description} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Frozen {description} checksum mismatch for {path}: "
            f"expected {expected}, found {actual}"
        )


def _verify_tree_manifest(root, pattern, expected, description):
    if not root.is_dir():
        raise FileNotFoundError(f"Frozen {description} directory is missing: {root}")
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    if len(files) != expected["file_count"] or digest.hexdigest() != expected["sha256"]:
        raise RuntimeError(
            f"Frozen {description} manifest mismatch: expected "
            f"{expected['file_count']} files / {expected['sha256']}, found "
            f"{len(files)} files / {digest.hexdigest()}"
        )


def _verify_path_manifest(paths, expected, description):
    paths = sorted(paths)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    if len(paths) != expected["file_count"] or digest.hexdigest() != expected["sha256"]:
        raise RuntimeError(
            f"Frozen {description} manifest mismatch: expected "
            f"{expected['file_count']} files / {expected['sha256']}, found "
            f"{len(paths)} files / {digest.hexdigest()}"
        )


def validate_frozen_input_assets(splits=None, asset_types=("raw_eeg", "stft_images")):
    """Reject changed data, model, or class-order inputs before inference."""
    if FROZEN_CONFIG.get("status") != "frozen":
        raise RuntimeError("The inference configuration is not marked frozen")
    if CLASS_NAMES != ["fear", "happy", "neutral", "sad"]:
        raise RuntimeError(f"Unexpected frozen class order: {CLASS_NAMES}")
    if _canonical_assignments(raw_transformer.TRIAL_ASSIGNMENTS) != TRIAL_ASSIGNMENTS:
        raise RuntimeError("Raw Transformer assignments disagree with the frozen configuration")
    if _canonical_assignments(gcn_data.TRIAL_ASSIGNMENTS) != TRIAL_ASSIGNMENTS:
        raise RuntimeError("EEGNet + GCN assignments disagree with the frozen configuration")
    for source_path, expected_hash in FROZEN_CONFIG["implementation_sources"].items():
        _verify_file_hash(PROJECT_ROOT / source_path, expected_hash, "frozen implementation source")

    raw = FROZEN_CONFIG["common_raw_eeg_preprocessing"]
    for module in (gcn_data, raw_transformer):
        actual = (
            module.SAMPLING_RATE_HZ,
            module.CHANNELS if hasattr(module, "CHANNELS") else module.EXPECTED_CHANNELS,
            module.WINDOW_SAMPLES,
            module.HOP_SAMPLES,
            module.BANDPASS_HZ,
            module.FILTER_ORDER,
        )
        expected = (
            raw["sampling_rate_hz"],
            raw["channels"],
            raw["windowing"]["window_samples"],
            raw["windowing"]["hop_samples"],
            tuple(raw["bandpass_hz"]),
            raw["filter_order"],
        )
        if actual != expected:
            raise RuntimeError(f"Raw EEG preprocessing code disagrees with frozen configuration: {actual} != {expected}")

    if splits is None:
        for key, pattern in (("raw_eeg", "*/*.mat"), ("stft_images", "*/*/*.png")):
            if key in asset_types:
                asset = FROZEN_CONFIG["input_assets"][key]
                _verify_tree_manifest(
                    PROJECT_ROOT / asset["root"], pattern, asset, key.replace("_", " ")
                )
    else:
        for key in asset_types:
            asset = FROZEN_CONFIG["input_assets"][key]
            for split in splits:
                if key == "raw_eeg":
                    paths = [
                        PROJECT_ROOT / asset["root"] / emotion / f"{trial}.mat"
                        for emotion, trials in TRIAL_ASSIGNMENTS[split].items()
                        for trial in trials
                    ]
                else:
                    paths = [
                        path
                        for emotion in CLASS_NAMES
                        for path in (PROJECT_ROOT / asset["root"] / split / emotion).glob("*.png")
                        if path.is_file()
                    ]
                if any(not path.is_file() for path in paths):
                    raise FileNotFoundError(f"Frozen {key} inputs are incomplete for split {split}")
                _verify_path_manifest(
                    paths,
                    asset["split_manifests"][split],
                    f"{key.replace('_', ' ')} {split}",
                )
    graph_metadata = MODEL_CONFIGS["EEGNet + GCN"]["preprocessing"]["normalization_graph_metadata"]
    _verify_file_hash(PROJECT_ROOT / graph_metadata["path"], graph_metadata["sha256"], "GCN metadata")


class EEGNetGCN(nn.Module):
    """Inference-compatible copy of the saved EEGNet + GCN architecture."""

    def __init__(self, adjacency, num_classes=4):
        super().__init__()
        self.register_buffer("adjacency", adjacency.float())
        self.temporal_conv = nn.Conv2d(1, 8, kernel_size=(1, 64), bias=False)
        self.temporal_bn = nn.BatchNorm2d(8)
        self.spatial_depthwise = nn.Conv2d(8, 16, kernel_size=(62, 1), groups=8, bias=False)
        self.spatial_bn = nn.BatchNorm2d(16)
        self.eegnet_dropout = nn.Dropout(0.25)
        self.separable_depthwise = nn.Conv2d(16, 16, kernel_size=(1, 16), groups=16, bias=False)
        self.separable_pointwise = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        self.separable_bn = nn.BatchNorm2d(16)
        self.node_projection = nn.Sequential(
            nn.Linear(8 * 25, 32), nn.LayerNorm(32), nn.ELU()
        )
        self.gcn_1 = nn.Linear(32, 64, bias=False)
        self.gcn_1_bias = nn.Parameter(torch.zeros(64))
        self.gcn_2 = nn.Linear(64, 64, bias=False)
        self.gcn_2_bias = nn.Parameter(torch.zeros(64))
        self.gcn_residual = nn.Linear(64, 64, bias=False)
        self.gcn_dropout = nn.Dropout(0.30)
        self.classifier = nn.Sequential(
            nn.Linear(16 + 64 + 64, 64), nn.ELU(), nn.Dropout(0.50), nn.Linear(64, num_classes)
        )

    @staticmethod
    def _same_time_pad(x, total_padding):
        left = total_padding // 2
        return F.pad(x, (left, total_padding - left, 0, 0))

    def forward(self, x):
        x = self._same_time_pad(x, 63)
        temporal = F.elu(self.temporal_bn(self.temporal_conv(x)))
        global_features = F.elu(self.spatial_bn(self.spatial_depthwise(temporal)))
        global_features = F.avg_pool2d(global_features, kernel_size=(1, 4))
        global_features = self.eegnet_dropout(global_features)
        global_features = self._same_time_pad(global_features, 15)
        global_features = self.separable_depthwise(global_features)
        global_features = self.separable_pointwise(global_features)
        global_features = F.elu(self.separable_bn(global_features))
        global_features = F.avg_pool2d(global_features, kernel_size=(1, 8))
        global_features = global_features.mean(dim=(-2, -1))

        batch, filters, channels, samples = temporal.shape
        node_time = temporal.permute(0, 2, 1, 3).reshape(batch * channels, filters, samples)
        node_time = F.adaptive_avg_pool1d(node_time, output_size=25)
        node_time = node_time.reshape(batch, channels, filters * 25)
        nodes = self.node_projection(node_time)
        hidden = F.elu(torch.einsum("ij,bjf->bif", self.adjacency, nodes) @ self.gcn_1.weight.T + self.gcn_1_bias)
        hidden = self.gcn_dropout(hidden)
        adjacency_hidden = torch.einsum("ij,bjf->bif", self.adjacency, hidden)
        graph_features = F.elu(
            adjacency_hidden @ self.gcn_2.weight.T
            + self.gcn_2_bias
            + hidden @ self.gcn_residual.weight.T
        )
        fused = torch.cat((global_features, graph_features.mean(dim=1), graph_features.max(dim=1).values), dim=1)
        return self.classifier(fused)


class EEGCRNN(nn.Module):
    """Inference-compatible copy of the original 31.58% STFT CRNN."""

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
            nn.Linear(256 * 4, 256), nn.LayerNorm(256), nn.GELU()
        )
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, images):
        features = self.cnn(images)
        height = features.shape[2]
        bands = []
        for band_index in range(4):
            start = (band_index * height) // 4
            end = ((band_index + 1) * height + 3) // 4
            bands.append(features[:, :, start:end, :].mean(dim=2))
        sequence = torch.stack(bands, dim=2).permute(0, 3, 2, 1).flatten(start_dim=2)
        sequence = self.sequence_projection(sequence)
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=1, embedding_dim=64):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.projection(x).flatten(2).transpose(1, 2)


class STFTTransformer(nn.Module):
    """Inference-compatible copy of the saved STFT image Transformer."""

    def __init__(self, image_size=224, patch_size=16, embedding_dim=64,
                 num_heads=4, num_layers=2, num_classes=4, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, 1, embedding_dim)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        x = self.patch_embedding(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1) + self.position_embedding
        x = self.transformer_encoder(x)
        return self.classifier(self.norm(x[:, 0]))


def _load_checkpoint(path, device, expected_sha256):
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    _verify_file_hash(path, expected_sha256, "model checkpoint")
    return torch.load(path, map_location=device, weights_only=True)


def load_models(device):
    """Load only the four selected best-validation checkpoints, strictly."""
    # Model loading checks configuration and checkpoint artifacts only. Input
    # manifests are verified per split so validation-only runs never read test data.
    validate_frozen_input_assets(splits=(), asset_types=())
    gcn_metadata = json.loads(GCN_METADATA.read_text(encoding="utf-8"))
    if gcn_metadata.get("normalization_source") != "TRAIN trials only":
        raise RuntimeError("GCN normalization metadata does not confirm train-only statistics")
    expected_train_trials = sorted(
        trial for names in TRIAL_ASSIGNMENTS["train"].values() for trial in names
    )
    if sorted(gcn_metadata.get("graph_train_trials", [])) != expected_train_trials:
        raise RuntimeError("Saved GCN graph was not built from the expected training trial set")
    adjacency = torch.tensor(gcn_metadata["normalized_adjacency"], dtype=torch.float32)
    gcn_model = EEGNetGCN(adjacency).to(device)
    gcn_checkpoint = _load_checkpoint(
        GCN_CHECKPOINT, device, MODEL_CONFIGS["EEGNet + GCN"]["checkpoint"]["sha256"]
    )
    if list(gcn_checkpoint.get("class_names", [])) != CLASS_NAMES:
        raise RuntimeError("EEGNet + GCN checkpoint class order mismatch")
    gcn_model.load_state_dict(gcn_checkpoint["model_state_dict"], strict=True)
    if not torch.allclose(gcn_model.adjacency.cpu(), adjacency, atol=1e-6, rtol=1e-6):
        raise RuntimeError("GCN checkpoint adjacency differs from its saved training graph metadata")

    raw_model = raw_transformer.RawEEGTransformer().to(device)
    raw_checkpoint = _load_checkpoint(
        RAW_TRANSFORMER_CHECKPOINT, device, MODEL_CONFIGS["Raw EEG Transformer"]["checkpoint"]["sha256"]
    )
    if list(raw_checkpoint.get("class_names", [])) != CLASS_NAMES:
        raise RuntimeError("Raw EEG Transformer checkpoint class order mismatch")
    raw_model.load_state_dict(raw_checkpoint["model_state_dict"], strict=True)

    crnn_model = EEGCRNN().to(device)
    crnn_checkpoint = _load_checkpoint(
        CRNN_CHECKPOINT, device, MODEL_CONFIGS["STFT CRNN"]["checkpoint"]["sha256"]
    )
    if list(crnn_checkpoint.get("class_names", [])) != CLASS_NAMES:
        raise RuntimeError("Original STFT CRNN checkpoint class order mismatch")
    if int(crnn_checkpoint.get("image_size", -1)) != IMAGE_SIZE:
        raise RuntimeError("Original STFT CRNN checkpoint image-size metadata mismatch")
    crnn_model.load_state_dict(crnn_checkpoint["model_state_dict"], strict=True)

    stft_transformer = STFTTransformer().to(device)
    transformer_state = _load_checkpoint(
        STFT_TRANSFORMER_CHECKPOINT, device, MODEL_CONFIGS["STFT Transformer"]["checkpoint"]["sha256"]
    )
    # The historical STFT Transformer checkpoint is a bare state_dict.
    if not isinstance(transformer_state, dict) or "patch_embedding.projection.weight" not in transformer_state:
        raise RuntimeError("STFT Transformer checkpoint has an unexpected format")
    stft_transformer.load_state_dict(transformer_state, strict=True)

    for model in (gcn_model, raw_model, crnn_model, stft_transformer):
        model.eval()

    checkpoint_paths = {
        "EEGNet + GCN": str(GCN_CHECKPOINT),
        "Raw EEG Transformer": str(RAW_TRANSFORMER_CHECKPOINT),
        "STFT CRNN": str(CRNN_CHECKPOINT),
        "STFT Transformer": str(STFT_TRANSFORMER_CHECKPOINT),
    }
    model_configs = {
        "EEGNet + GCN": {
            "class_order": CLASS_NAMES,
            "selected_checkpoint": "best validation checkpoint",
            "best_epoch": gcn_checkpoint["best_epoch"],
            "input": "62 channels x 400 samples; 200 Hz assumption; 2 s/0.5 s windows",
            "preprocessing": "1-45 Hz, order-4 zero-phase bandpass; train-only channel z-score and saved train-only functional graph",
            "normalization_metadata_path": str(GCN_METADATA),
        },
        "Raw EEG Transformer": {
            "class_order": CLASS_NAMES,
            "selected_checkpoint": "best validation checkpoint",
            "best_epoch": raw_checkpoint["best_epoch"],
            "input": "62 channels x 400 samples; 200 Hz assumption; 2 s/0.5 s windows",
            "preprocessing": "1-45 Hz, order-4 zero-phase bandpass; train-only per-channel z-score recomputed from train trials",
            "normalization_metadata": "not stored; recomputed from training trials only using the baseline preprocessing",
        },
        "STFT CRNN": {
            "class_order": CLASS_NAMES,
            "selected_checkpoint": "original best validation checkpoint; fixed-scale variants excluded",
            "best_epoch": crnn_checkpoint.get("best_epoch"),
            "input_dataset": str(STFT_DATASET_ROOT),
            "preprocessing": "224x224 RGB; ToTensor; ImageNet mean/std normalization; deterministic evaluation",
        },
        "STFT Transformer": {
            "class_order": CLASS_NAMES,
            "selected_checkpoint": "best validation checkpoint",
            "input_dataset": str(STFT_DATASET_ROOT),
            "preprocessing": "224x224 grayscale; ToTensor; mean/std 0.5; deterministic evaluation",
            "class_order_source": "training script and ImageFolder class directories (checkpoint is bare state_dict)",
        },
    }
    models = {
        "EEGNet + GCN": gcn_model,
        "Raw EEG Transformer": raw_model,
        "STFT CRNN": crnn_model,
        "STFT Transformer": stft_transformer,
    }
    return models, checkpoint_paths, model_configs, gcn_metadata, crnn_checkpoint


def _load_raw_records(split, include_train_for_stats):
    assignments = TRIAL_ASSIGNMENTS
    wanted_splits = {split}
    if include_train_for_stats:
        wanted_splits.add("train")
    records = {}
    for source_split in ("train", split):
        if source_split not in wanted_splits:
            continue
        for emotion in CLASS_NAMES:
            for trial_key in assignments[source_split][emotion]:
                path = RAW_ROOT / emotion / f"{trial_key}.mat"
                if not path.is_file():
                    raise FileNotFoundError(f"Expected raw trial file missing: {path}")
                mat = loadmat(path)
                if "EEG" not in mat or "label" not in mat:
                    raise KeyError(f"Missing EEG/label in {path}")
                label = np.asarray(mat["label"]).squeeze()
                mat_labels = FROZEN_CONFIG["input_assets"]["raw_eeg"]["mat_label_values_by_class"]
                if label.size != 1 or int(label) != mat_labels[emotion]:
                    raise ValueError(f"Raw label/folder mismatch: {path}")
                processed = raw_transformer.preprocess(mat["EEG"], path)
                records[trial_key] = {
                    "key": trial_key,
                    "emotion": emotion,
                    "split": source_split,
                    "signal": processed,
                }
    return records


def _raw_windows(
    signal_array,
    window_samples=FROZEN_CONFIG["common_raw_eeg_preprocessing"]["windowing"]["window_samples"],
    hop_samples=FROZEN_CONFIG["common_raw_eeg_preprocessing"]["windowing"]["hop_samples"],
):
    starts = range(0, signal_array.shape[-1] - window_samples + 1, hop_samples)
    return [(start, signal_array[..., start:start + window_samples]) for start in starts]


def _infer_array_windows(model, windows, device, add_image_axis=False):
    logits_output = []
    for start in range(0, len(windows), BATCH_SIZE):
        batch = np.stack([item[1] for item in windows[start:start + BATCH_SIZE]]).astype(np.float32)
        tensor = torch.from_numpy(batch)
        if add_image_axis:
            tensor = tensor.unsqueeze(1)
        with torch.no_grad():
            logits = model(tensor.to(device)).detach().cpu()
            probabilities = torch.softmax(logits, dim=1)
        logits_np = logits.numpy()
        probabilities_np = probabilities.numpy()
        _validate_probability_batch(logits_np, probabilities_np)
        for local_index, item in enumerate(windows[start:start + BATCH_SIZE]):
            logits_output.append({
                "window_id": int(item[0]),
                "logits": logits_np[local_index].astype(float).tolist(),
                "probabilities": probabilities_np[local_index].astype(float).tolist(),
            })
    return logits_output


def _validate_probability_batch(logits, probabilities):
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise RuntimeError(f"Model produced unexpected logits shape {logits.shape}")
    if probabilities.shape != logits.shape:
        raise RuntimeError("Softmax probabilities do not match logits shape")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise RuntimeError("Model produced NaN or Inf logits/probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("At least one per-window probability vector does not sum to one")


def raw_model_normalization(model_name, gcn_metadata=None, train_records=None):
    """Return the frozen channel normalization used by a raw EEG branch."""
    if model_name == "Raw EEG Transformer":
        frozen_norm = MODEL_CONFIGS[model_name]["preprocessing"]
        mean = np.asarray(frozen_norm["mean"], dtype=np.float32)
        std = np.asarray(frozen_norm["std"], dtype=np.float32)
        if train_records is not None:
            derived_mean, derived_std, sample_count = raw_transformer.training_normalization(train_records)
            if (
                sample_count != frozen_norm["normalization_sample_count"]
                or not np.allclose(derived_mean, mean, atol=1e-12, rtol=0.0)
                or not np.allclose(derived_std, std, atol=1e-6, rtol=0.0)
            ):
                raise RuntimeError("Raw EEG Transformer training normalization no longer matches the frozen values")
        else:
            sample_count = frozen_norm["normalization_sample_count"]
        return mean, std, {
            "source": "training trials only",
            "sample_count": int(sample_count),
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
        }

    if model_name == "EEGNet + GCN":
        if not gcn_metadata or gcn_metadata.get("normalization_source") != "TRAIN trials only":
            raise RuntimeError("Missing verified train-only EEGNet + GCN normalization metadata")
        mean = np.asarray(gcn_metadata["normalization_mean"], dtype=np.float32)
        std = np.asarray(gcn_metadata["normalization_std"], dtype=np.float32)
        return mean, std, {
            "source": gcn_metadata["normalization_source"],
            "metadata_path": str(GCN_METADATA),
        }

    raise ValueError(f"Unknown raw EEG model branch: {model_name}")


def infer_raw_recording(eeg, model_name, model, device, gcn_metadata, recording_id="input"):
    """Infer one arbitrary EEG recording through a frozen raw branch."""
    processed = raw_transformer.preprocess(eeg, recording_id)
    mean, std, normalization_info = raw_model_normalization(model_name, gcn_metadata)
    normalized = (processed - mean[:, None]) / std[:, None]
    windows = _raw_windows(normalized)
    hop_samples = FROZEN_CONFIG["common_raw_eeg_preprocessing"]["windowing"]["hop_samples"]
    model_inputs = [(start // hop_samples, window) for start, window in windows]
    window_outputs = _infer_array_windows(
        model,
        model_inputs,
        device,
        add_image_axis=(model_name == "EEGNet + GCN"),
    )
    trial_item = {
        "true_class": None,
        "windows": window_outputs,
        "window_count": len(window_outputs),
    }
    return aggregate_model_trial(recording_id, trial_item, true_class=None), normalization_info


def infer_raw_branch(split, model_name, model, device, gcn_metadata=None):
    """Run one raw checkpoint on train-derived preprocessing plus one split only."""
    include_train_for_stats = model_name == "Raw EEG Transformer"
    raw_input_splits = tuple(dict.fromkeys(("train", split) if include_train_for_stats else (split,)))
    validate_frozen_input_assets(splits=raw_input_splits, asset_types=("raw_eeg",))
    records = _load_raw_records(split, include_train_for_stats)
    if model_name == "Raw EEG Transformer":
        train_records = [record for record in records.values() if record["split"] == "train"]
        mean, std, norm_info = raw_model_normalization(model_name, train_records=train_records)
    else:
        mean, std, norm_info = raw_model_normalization(model_name, gcn_metadata=gcn_metadata)

    output = {}
    expected_keys = {
        trial_key
        for emotion in CLASS_NAMES
        for trial_key in TRIAL_ASSIGNMENTS[split][emotion]
    }
    split_records = [record for record in records.values() if record["split"] == split]
    if {record["key"] for record in split_records} != expected_keys:
        raise RuntimeError(f"Raw {split} trial coverage differs from fixed assignments")

    for record in split_records:
        normalized_signal = (record["signal"] - mean[:, None]) / std[:, None]
        windows = _raw_windows(normalized_signal)
        hop_samples = FROZEN_CONFIG["common_raw_eeg_preprocessing"]["windowing"]["hop_samples"]
        model_inputs = [(start // hop_samples, window) for start, window in windows]
        if model_name == "EEGNet + GCN":
            model_outputs = _infer_array_windows(model, model_inputs, device, add_image_axis=True)
        else:
            model_outputs = _infer_array_windows(model, model_inputs, device)
        output[record["key"]] = {
            "true_class": record["emotion"],
            "windows": model_outputs,
            "window_count": len(model_outputs),
        }
    return output, norm_info


def _stft_eval_transform(model_name):
    preprocessing = MODEL_CONFIGS[model_name]["preprocessing"]
    resize = tuple(preprocessing["resize_px"])
    if model_name == "STFT CRNN":
        return transforms.Compose([
            transforms.Resize(
                resize,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=preprocessing["antialias"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocessing["normalize_mean"],
                std=preprocessing["normalize_std"],
            ),
        ])
    return transforms.Compose([
        transforms.Resize(
            resize,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=preprocessing["antialias"],
        ),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preprocessing["normalize_mean"],
            std=preprocessing["normalize_std"],
        ),
    ])


def infer_stft_image_paths(image_paths, model_name, model, device, window_ids=None):
    """Infer a caller-supplied ordered sequence of frozen STFT PNG windows."""
    if model_name not in ("STFT CRNN", "STFT Transformer"):
        raise ValueError(f"Unknown STFT model branch: {model_name}")
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("At least one pre-rendered STFT PNG is required")
    if window_ids is None:
        window_ids = list(range(len(paths)))
    if len(paths) != len(window_ids):
        raise ValueError("STFT image paths and window IDs must have the same length")

    transform = _stft_eval_transform(model_name)
    output = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        tensors = []
        for path in batch_paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch).detach().cpu()
            probabilities = torch.softmax(logits, dim=1)
        logits_np, probabilities_np = logits.numpy(), probabilities.numpy()
        _validate_probability_batch(logits_np, probabilities_np)
        for index, local_index in enumerate(range(start, start + len(batch_paths))):
            output.append({
                "window_id": int(window_ids[local_index]),
                "logits": logits_np[index].astype(float).tolist(),
                "probabilities": probabilities_np[index].astype(float).tolist(),
            })
    return output


def infer_stft_branch(split, model_name, model, device):
    validate_frozen_input_assets(splits=(split,), asset_types=("stft_images",))
    split_dir = STFT_DATASET_ROOT / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing requested STFT split directory: {split_dir}")
    folders = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    if folders != CLASS_NAMES:
        raise RuntimeError(f"STFT ImageFolder class order mismatch in {split}: {folders}")
    transform = _stft_eval_transform(model_name)
    files = []
    for emotion in CLASS_NAMES:
        for path in sorted((split_dir / emotion).glob("*.png")):
            match = TRIAL_FILE_RE.fullmatch(path.stem)
            if match is None:
                raise ValueError(f"Cannot parse trial/window from STFT filename: {path}")
            trial_key = match.group("trial")
            expected_emotion = next(
                (cls for cls in CLASS_NAMES for key in TRIAL_ASSIGNMENTS[split][cls] if key == trial_key),
                None,
            )
            if expected_emotion != emotion:
                raise RuntimeError(f"STFT image does not belong to the fixed {split} assignment: {path}")
            files.append((path, trial_key, emotion, int(match.group("window"))))

    output = {}
    for start in range(0, len(files), BATCH_SIZE):
        batch_files = files[start:start + BATCH_SIZE]
        tensors = []
        for path, _, _, _ in batch_files:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch).detach().cpu()
            probabilities = torch.softmax(logits, dim=1)
        logits_np, probabilities_np = logits.numpy(), probabilities.numpy()
        _validate_probability_batch(logits_np, probabilities_np)
        for index, (_, trial_key, emotion, window_id) in enumerate(batch_files):
            trial_output = output.setdefault(
                trial_key, {"true_class": emotion, "windows": [], "window_count": 0}
            )
            if trial_output["true_class"] != emotion:
                raise RuntimeError(f"Inconsistent emotion for STFT trial {trial_key}")
            trial_output["windows"].append({
                "window_id": window_id,
                "logits": logits_np[index].astype(float).tolist(),
                "probabilities": probabilities_np[index].astype(float).tolist(),
            })
            trial_output["window_count"] += 1
    return output


def aggregate_model_outputs(split, model_outputs):
    expected = {
        trial_key: emotion
        for emotion in CLASS_NAMES
        for trial_key in TRIAL_ASSIGNMENTS[split][emotion]
    }
    result = {}
    if set(model_outputs) != set(expected):
        raise RuntimeError(
            f"Model trial coverage mismatch for {split}: missing={sorted(set(expected)-set(model_outputs))}, "
            f"extra={sorted(set(model_outputs)-set(expected))}"
        )
    for trial_key, emotion in expected.items():
        item = model_outputs[trial_key]
        if item["true_class"] != emotion or not item["windows"]:
            raise RuntimeError(f"Missing or mislabeled model predictions for {trial_key}")
        result[trial_key] = aggregate_model_trial(trial_key, item, true_class=emotion)
    return result


def aggregate_model_trial(trial_key, item, true_class=None):
    """Apply the frozen window-to-trial aggregation to one model and recording."""
    windows = item.get("windows", [])
    if not windows:
        raise RuntimeError(f"Missing model window predictions for {trial_key}")
    probs = np.asarray([window["probabilities"] for window in windows], dtype=np.float64)
    logits = np.asarray([window["logits"] for window in windows], dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(CLASS_NAMES) or not np.isfinite(probs).all():
        raise RuntimeError(f"Invalid per-window probabilities for {trial_key}")
    if logits.shape != probs.shape or not np.isfinite(logits).all():
        raise RuntimeError(f"Invalid per-window logits for {trial_key}")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError(f"Per-window probabilities are not normalized for {trial_key}")
    mean_probability = probs.mean(axis=0)
    mean_probability /= mean_probability.sum()
    if not np.isfinite(mean_probability).all() or not np.isclose(mean_probability.sum(), 1.0, atol=1e-5):
        raise RuntimeError(f"Invalid trial probability vector for {trial_key}")
    prediction = int(np.argmax(mean_probability))
    return {
        "true_class": true_class,
        "window_count": len(windows),
        "mean_trial_logits_diagnostic_only": logits.mean(axis=0).astype(float).tolist(),
        "probabilities": mean_probability.astype(float).tolist(),
        "predicted_class": CLASS_NAMES[prediction],
        "model_confidence_score_not_calibrated": float(mean_probability[prediction]),
        "window_outputs": windows,
    }


def fuse_trial_predictions(split, model_trial_outputs):
    expected_trials = {
        key: emotion
        for emotion in CLASS_NAMES
        for key in TRIAL_ASSIGNMENTS[split][emotion]
    }
    if set(model_trial_outputs) != set(MODEL_NAMES):
        raise RuntimeError(f"Expected exactly four model outputs; got {list(model_trial_outputs)}")
    for model_name in MODEL_NAMES:
        if set(model_trial_outputs[model_name]) != set(expected_trials):
            raise RuntimeError(f"Not all {split} trials have outputs from {model_name}")

    fused_trials = {}
    for trial_key, true_class in expected_trials.items():
        model_items = {}
        fused = np.zeros(4, dtype=np.float64)
        for model_name in MODEL_NAMES:
            item = model_trial_outputs[model_name][trial_key]
            if item["true_class"] != true_class:
                raise RuntimeError(f"Class mismatch for {trial_key} from {model_name}")
            probabilities = np.asarray(item["probabilities"], dtype=np.float64)
            if probabilities.shape != (4,) or not np.isfinite(probabilities).all():
                raise RuntimeError(f"Invalid class probabilities for {model_name}, {trial_key}")
            if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
                raise RuntimeError(f"Probability sum is not one for {model_name}, {trial_key}")
            weight = FUSION_WEIGHTS[model_name]
            fused += weight * probabilities
            predicted_index = int(np.argmax(probabilities))
            model_items[model_name] = {
                "window_count": item["window_count"],
                "probabilities": probabilities.astype(float).tolist(),
                "predicted_class": CLASS_NAMES[predicted_index],
                "model_confidence_score_not_calibrated": float(probabilities[predicted_index]),
                "window_outputs": item["window_outputs"],
            }
        fused /= fused.sum()
        if not np.isfinite(fused).all() or not np.isclose(fused.sum(), 1.0, atol=1e-5):
            raise RuntimeError(f"Invalid fused probability vector for {trial_key}")
        final_index = int(np.argmax(fused))
        fused_trials[trial_key] = {
            "true_class": true_class,
            "individual_models": model_items,
            "fusion_weights": FUSION_WEIGHTS,
            "fused_probabilities": fused.astype(float).tolist(),
            "final_emotion": CLASS_NAMES[final_index],
            "model_confidence_score_not_calibrated": float(fused[final_index]),
        }
    return fused_trials


def compute_metrics(trial_records):
    ordered = list(trial_records.items())
    targets = np.asarray([CLASS_TO_INDEX[item["true_class"]] for _, item in ordered], dtype=np.int64)
    predictions = np.asarray([CLASS_TO_INDEX[item["final_emotion"]] for _, item in ordered], dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=range(4), zero_division=0
    )
    return {
        "trial_count": int(len(ordered)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(targets, predictions, labels=range(4), average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            targets, predictions, labels=range(4)
        ).tolist(),
    }


def expected_trials(split):
    return {
        key: emotion
        for emotion in CLASS_NAMES
        for key in TRIAL_ASSIGNMENTS[split][emotion]
    }


def make_predictions_for_split(split, models, device, gcn_metadata):
    if split not in ("validation", "test"):
        raise ValueError("Fusion inference is restricted to validation or test splits")
    outputs = {}
    runtime_preprocessing = {}
    outputs["EEGNet + GCN"], _ = infer_raw_branch(
        split, "EEGNet + GCN", models["EEGNet + GCN"], device, gcn_metadata
    )
    outputs["Raw EEG Transformer"], raw_norm = infer_raw_branch(
        split, "Raw EEG Transformer", models["Raw EEG Transformer"], device
    )
    outputs["STFT CRNN"] = infer_stft_branch(split, "STFT CRNN", models["STFT CRNN"], device)
    outputs["STFT Transformer"] = infer_stft_branch(
        split, "STFT Transformer", models["STFT Transformer"], device
    )
    runtime_preprocessing["Raw EEG Transformer train-only normalization"] = raw_norm
    return {
        model_name: aggregate_model_outputs(split, outputs[model_name])
        for model_name in MODEL_NAMES
    }, runtime_preprocessing


def validate_static_test_contract():
    """Check expected held-out IDs from the fixed split without opening test files."""
    expected = expected_trials("test")
    if len(expected) != 4 or set(expected.values()) != set(CLASS_NAMES):
        raise RuntimeError("Fixed test assignment must contain exactly one trial per class")
    return {"expected_trial_ids": expected, "data_opened": False, "predictions_generated": False}

