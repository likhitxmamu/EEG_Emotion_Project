"""Generate a full-window, train-scaled STFT experiment without touching old data.

Outputs:
  data/processed/stft_spectrograms_fixedscale/
  data/processed/crnn_dataset_fixedscale/{train,validation,test}/
  data/diagnostics/fixedscale_experiment/

The fixed train/validation/test trial assignment is validated against the
existing CRNN dataset. Every valid window in every source recording is used.
Existing output directories cause a safe failure; nothing is deleted.
"""

import hashlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import convert_to_spectrogram as eeg_source  # existing read/clean/window helpers only


DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "output"
ORIGINAL_STFT_ROOT = DATA_ROOT / "processed" / "stft_spectrograms"
REFERENCE_DATASET_ROOT = DATA_ROOT / "processed" / "crnn_dataset"
STFT_OUTPUT_ROOT = DATA_ROOT / "processed" / "stft_spectrograms_fixedscale_1024"
DATASET_OUTPUT_ROOT = DATA_ROOT / "processed" / "crnn_dataset_fixedscale_1024"
DIAGNOSTIC_ROOT = DATA_ROOT / "diagnostics" / "fixedscale_1024_experiment"
PARAMETERS_PATH = DIAGNOSTIC_ROOT / "scaling_parameters.json"

CLASSES = ("fear", "happy", "neutral", "sad")
SPLITS = ("train", "validation", "test")
SEED = 42
CLIP_PERCENTILES = (1.0, 99.0)
OUTPUT_SIZE = (1024, 1024)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
TRIAL_IMAGE_RE = re.compile(
    r"^(?P<trial>.+_trial\d+)_window_(?P<window>\d+)_STFT_RGB$",
    re.IGNORECASE,
)
RAW_TRIAL_RE = re.compile(r"^(?P<trial>.+_trial\d+)$", re.IGNORECASE)

np.random.seed(SEED)

# Full recording/trial keys from the user-provided, already audited split.
FIXED_TRIAL_ASSIGNMENTS = {
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


def image_trial_key(path):
    match = TRIAL_IMAGE_RE.fullmatch(Path(path).stem)
    if not match:
        raise ValueError(f"Cannot parse existing image trial naming: {path}")
    return match.group("trial")


def raw_trial_key(path):
    match = RAW_TRIAL_RE.fullmatch(Path(path).stem)
    if not match:
        raise ValueError(f"Cannot parse raw recording trial ID: {path}")
    return match.group("trial")


def fixed_assignments():
    mapping = {}
    for split, class_trials in FIXED_TRIAL_ASSIGNMENTS.items():
        for emotion, trials in class_trials.items():
            for trial in trials:
                key = (emotion, trial)
                if key in mapping:
                    raise ValueError(f"Trial assigned more than once: {key}")
                mapping[key] = split
    if len(mapping) != 20:
        raise ValueError(f"Expected 20 fixed trial groups, found {len(mapping)}")
    return mapping


def validate_reference_split(assignments):
    reference = {}
    for split in SPLITS:
        for emotion in CLASSES:
            folder = REFERENCE_DATASET_ROOT / split / emotion
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing existing split reference: {folder}")
            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    key = (emotion, image_trial_key(path))
                    previous = reference.setdefault(key, split)
                    if previous != split:
                        raise RuntimeError(f"Existing dataset leaks trial {key}: {previous}/{split}")
    if reference != assignments:
        missing = sorted(set(assignments) - set(reference))
        unexpected = sorted(set(reference) - set(assignments))
        changed = sorted(key for key in set(reference) & set(assignments) if reference[key] != assignments[key])
        raise RuntimeError(
            "Existing trial assignments do not exactly match the requested fixed split. "
            f"Missing={missing}, unexpected={unexpected}, changed={changed}"
        )


def power_db_for_window(eeg_window):
    frequencies, times, zxx = signal.stft(
        eeg_window,
        fs=eeg_source.FS,
        window="hann",
        nperseg=eeg_source.N_PER_SEG,
        noverlap=eeg_source.N_OVERLAP,
        nfft=eeg_source.N_FFT,
        detrend="constant",
        boundary=None,
        padded=False,
    )
    power_db = 10.0 * np.log10(np.abs(zxx) ** 2 + 1e-12)
    frequency_mask = (frequencies >= eeg_source.FREQ_MIN) & (frequencies <= eeg_source.FREQ_MAX)
    return frequencies[frequency_mask], times, power_db[frequency_mask]


def make_spectrogram(power_db, frequencies, times, output_path, scale_min, scale_max):
    normalized = np.clip((power_db - scale_min) / (scale_max - scale_min), 0.0, 1.0)
    fig = plt.figure(figsize=(OUTPUT_SIZE[0] / 100, OUTPUT_SIZE[1] / 100), dpi=100, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(
        times,
        frequencies,
        normalized,
        shading="gouraud",
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        antialiased=True,
    )
    ax.set_axis_off()
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(eeg_source.FREQ_MIN, eeg_source.FREQ_MAX)
    ax.margins(x=0, y=0)
    fig.savefig(output_path, dpi=100, bbox_inches=None, pad_inches=0, facecolor="black", edgecolor="none")
    plt.close(fig)
    return normalized


def update_pixel_moments(moment, pixels):
    values = pixels.astype(np.float64).ravel()
    moment["count"] += values.size
    moment["sum"] += float(values.sum())
    moment["sum_squares"] += float(np.dot(values, values))


def finish_pixel_moments(moment):
    mean = moment["sum"] / moment["count"] if moment["count"] else float("nan")
    variance = max(moment["sum_squares"] / moment["count"] - mean * mean, 0.0) if moment["count"] else float("nan")
    return {"mean": mean, "std": float(np.sqrt(variance)), "pixels": moment["count"]}


def aggregate_image_moments(means, stds):
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    if means.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    mean = float(means.mean())
    variance = float(np.mean(stds ** 2 + means ** 2) - mean ** 2)
    return {"mean": mean, "std": float(np.sqrt(max(variance, 0.0)))}


def grayscale_pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize(OUTPUT_SIZE, Image.Resampling.LANCZOS), dtype=np.uint8)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    print("FIXED-SCALE ALL-WINDOW STFT EXPERIMENT")
    print(f"Seed: {SEED}")
    print(f"Original STFT (read-only): {ORIGINAL_STFT_ROOT}")
    print(f"New STFT output: {STFT_OUTPUT_ROOT}")
    print(f"New CRNN dataset: {DATASET_OUTPUT_ROOT}")

    for target in (STFT_OUTPUT_ROOT, DATASET_OUTPUT_ROOT, DIAGNOSTIC_ROOT):
        if os.path.lexists(target):
            raise FileExistsError(f"Refusing to overwrite existing data: {target}")
    if not ORIGINAL_STFT_ROOT.is_dir():
        raise FileNotFoundError(f"Original source STFT directory missing: {ORIGINAL_STFT_ROOT}")

    assignment = fixed_assignments()
    validate_reference_split(assignment)
    raw_files = eeg_source.find_eeg_files()
    records = []
    seen_trials = set()

    for raw_path in raw_files:
        emotion = eeg_source.get_emotion_from_path(raw_path)
        trial = raw_trial_key(raw_path)
        key = (emotion, trial)
        if emotion not in CLASSES or key not in assignment:
            raise ValueError(f"Raw recording not covered by fixed split mapping: {raw_path}, {key}")
        if key in seen_trials:
            raise ValueError(f"More than one raw recording for trial group {key}")
        seen_trials.add(key)

        eeg, variable_name = eeg_source.load_one_eeg_file(raw_path)
        if eeg is None:
            raise RuntimeError(f"Could not load EEG recording: {raw_path}")
        eeg = eeg_source.clean_eeg_signal(eeg)
        windows = eeg_source.create_eeg_windows(eeg)
        print(f"{emotion:<8} {trial:<28} valid_windows={len(windows)} variable={variable_name}")
        if not windows:
            raise RuntimeError(f"No valid EEG windows in {raw_path}")

        split = assignment[key]
        for window_number, (start, end, window) in enumerate(windows, start=1):
            frequencies, times, power_db = power_db_for_window(window)
            records.append({
                "emotion": emotion,
                "trial": trial,
                "split": split,
                "window_number": window_number,
                "start": start,
                "end": end,
                "raw_path": raw_path,
                "power_db": power_db,
                "frequencies": frequencies,
                "times": times,
            })

    if seen_trials != set(assignment):
        raise RuntimeError(
            f"Raw trial coverage mismatch. Missing={sorted(set(assignment)-seen_trials)}, "
            f"unexpected={sorted(seen_trials-set(assignment))}"
        )

    # The only source of clipping bounds is numeric STFT values from train trials.
    train_values_by_class = {emotion: [] for emotion in CLASSES}
    for record in records:
        if record["split"] == "train":
            train_values_by_class[record["emotion"]].append(record["power_db"].ravel())
    all_train_values = np.concatenate([
        values for class_values in train_values_by_class.values() for values in class_values
    ])
    scale_min, scale_max = np.percentile(all_train_values, CLIP_PERCENTILES)
    if not np.isfinite(scale_min) or not np.isfinite(scale_max) or scale_min >= scale_max:
        raise RuntimeError(f"Invalid training-only scale: {scale_min}, {scale_max}")

    records_by_trial = defaultdict(list)
    records_by_split = {split: [] for split in SPLITS}
    records_by_class_split = {split: {emotion: [] for emotion in CLASSES} for split in SPLITS}
    for record in records:
        records_by_trial[(record["emotion"], record["trial"])].append(record)
        records_by_split[record["split"]].append(record)
        records_by_class_split[record["split"]][record["emotion"]].append(record)

    print("\nTRAIN-ONLY LOG-POWER SCALING")
    print(f"  Training numerical values: {len(all_train_values):,}")
    print(f"  Percentiles: {CLIP_PERCENTILES[0]:g}th to {CLIP_PERCENTILES[1]:g}th")
    print(f"  Shared log-power range: {scale_min:.8f} to {scale_max:.8f} dB")
    print("  Validation/test values did not contribute to scaling.")
    print("\nTRIAL WINDOW COUNTS")
    for split in SPLITS:
        print(f"{split.upper()}:")
        for emotion in CLASSES:
            trials = FIXED_TRIAL_ASSIGNMENTS[split][emotion]
            details = [f"{trial}={len(records_by_trial[(emotion, trial)])}" for trial in trials]
            print(f"  {emotion:<8}: " + ", ".join(details))
        print(f"  total windows: {len(records_by_split[split])}")

    # Prepare all new output folders only after source/split/scale validation.
    for emotion in CLASSES:
        (STFT_OUTPUT_ROOT / emotion).mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        for emotion in CLASSES:
            (DATASET_OUTPUT_ROOT / split / emotion).mkdir(parents=True, exist_ok=False)
    DIAGNOSTIC_ROOT.mkdir(parents=True, exist_ok=False)

    parameters = {
        "seed": SEED,
        "representation": "STFT power -> 10*log10(power + 1e-12)",
        "sampling_rate_hz": eeg_source.FS,
        "frequency_range_hz": [eeg_source.FREQ_MIN, eeg_source.FREQ_MAX],
        "window_seconds": eeg_source.WINDOW_SECONDS,
        "step_seconds": eeg_source.WINDOW_SECONDS * (1 - eeg_source.OVERLAP_RATIO),
        "nperseg": eeg_source.N_PER_SEG,
        "noverlap": eeg_source.N_OVERLAP,
        "nfft": eeg_source.N_FFT,
        "scale_source": "all numerical STFT log-power values from fixed TRAIN trials only",
        "clip_percentiles": list(CLIP_PERCENTILES),
        "scale_min_db": float(scale_min),
        "scale_max_db": float(scale_max),
        "normalization": "clip to train-only dB bounds, then (x-min)/(max-min) to [0,1]; same bounds for every split",
        "output_image_size": list(OUTPUT_SIZE),
        "trial_assignments": FIXED_TRIAL_ASSIGNMENTS,
    }
    PARAMETERS_PATH.write_text(json.dumps(parameters, indent=2), encoding="utf-8")

    clip_stats = {split: {emotion: {"below": 0, "above": 0, "total": 0} for emotion in CLASSES} for split in SPLITS}
    numeric_stats = {emotion: [] for emotion in CLASSES}
    pixel_moments = {split: {emotion: {"count": 0, "sum": 0.0, "sum_squares": 0.0} for emotion in CLASSES} for split in SPLITS}
    split_hashes = {split: set() for split in SPLITS}
    representative_records = {}

    for index, record in enumerate(records, start=1):
        emotion, trial = record["emotion"], record["trial"]
        filename = f"{trial}_window_{record['window_number']:04d}_STFT_RGB.png"
        source_image = STFT_OUTPUT_ROOT / emotion / filename
        normalized = make_spectrogram(
            record["power_db"], record["frequencies"], record["times"],
            source_image, scale_min, scale_max,
        )
        dataset_image = DATASET_OUTPUT_ROOT / record["split"] / emotion / filename
        shutil.copy2(source_image, dataset_image)
        stats = clip_stats[record["split"]][emotion]
        stats["below"] += int(np.count_nonzero(record["power_db"] < scale_min))
        stats["above"] += int(np.count_nonzero(record["power_db"] > scale_max))
        stats["total"] += int(record["power_db"].size)
        if record["split"] == "train":
            numeric_stats[emotion].append(record["power_db"].ravel())
        pixels = grayscale_pixels(source_image)
        update_pixel_moments(pixel_moments[record["split"]][emotion], pixels)
        split_hashes[record["split"]].add(file_hash(dataset_image))
        representative_records.setdefault((emotion, trial), (source_image, filename))
        if index % 500 == 0 or index == len(records):
            print(f"  Rendered {index}/{len(records)} all-window images")

    # Current per-window images are used only for same-file, train-only comparison.
    old_image_stats = {emotion: {"means": [], "stds": []} for emotion in CLASSES}
    matched_examples = {}
    for emotion in CLASSES:
        train_trials = FIXED_TRIAL_ASSIGNMENTS["train"][emotion]
        candidates = []
        for trial in train_trials:
            for record in records_by_trial[(emotion, trial)]:
                filename = f"{trial}_window_{record['window_number']:04d}_STFT_RGB.png"
                original = ORIGINAL_STFT_ROOT / emotion / filename
                if original.is_file():
                    pixels = grayscale_pixels(original)
                    old_image_stats[emotion]["means"].append(float(pixels.mean()))
                    old_image_stats[emotion]["stds"].append(float(pixels.std()))
                    candidates.append((record["window_number"], original, filename))
        if not candidates:
            raise RuntimeError(f"No matched original autoscaled training images for {emotion}")
        candidates.sort(key=lambda item: (item[0], item[2]))
        selected = candidates[len(candidates) // 2]
        new_image = STFT_OUTPUT_ROOT / emotion / selected[2]
        matched_examples[emotion] = (selected[1], new_image, selected[2])

    # Create representative side-by-side comparison, selected from train trials only.
    fig, axes = plt.subplots(4, 2, figsize=(8, 14))
    for row, emotion in enumerate(CLASSES):
        old_path, new_path, filename = matched_examples[emotion]
        for col, (path, title) in enumerate(((old_path, "Current per-window scale"), (new_path, "Fixed train-only scale"))):
            with Image.open(path) as image:
                axes[row, col].imshow(image.convert("RGB"))
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{title}\n{emotion} | {filename}", fontsize=8)
    fig.suptitle("Matched training-trial spectrogram examples")
    fig.tight_layout()
    comparison_path = DIAGNOSTIC_ROOT / "autoscaled_vs_fixedscale.png"
    fig.savefig(comparison_path, dpi=150)
    plt.close(fig)

    # Per-image intensity distribution summaries for matched train images.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for emotion in CLASSES:
        axes[0].hist(old_image_stats[emotion]["means"], bins=20, alpha=0.35, label=f"{emotion} autoscaled")
        new_means = []
        for trial in FIXED_TRIAL_ASSIGNMENTS["train"][emotion]:
            for record in records_by_trial[(emotion, trial)]:
                path = STFT_OUTPUT_ROOT / emotion / f"{trial}_window_{record['window_number']:04d}_STFT_RGB.png"
                new_means.append(float(grayscale_pixels(path).mean()))
        axes[0].hist(new_means, bins=20, histtype="step", linewidth=1.5, label=f"{emotion} fixed")
        old_moments = aggregate_image_moments(
            old_image_stats[emotion]["means"], old_image_stats[emotion]["stds"]
        )
        axes[1].scatter(
            [old_moments["mean"]],
            [old_moments["std"]],
            label=f"{emotion} autoscaled",
            marker="o",
        )
        new_mean = finish_pixel_moments(pixel_moments["train"][emotion])["mean"]
        new_std = finish_pixel_moments(pixel_moments["train"][emotion])["std"]
        axes[1].scatter([new_mean], [new_std], label=f"{emotion} fixed", marker="x", s=70)
    axes[0].set_title("Training image mean intensity distributions")
    axes[0].set_xlabel("Grayscale intensity (0–255)")
    axes[0].set_ylabel("Images")
    axes[1].set_title("Training class pixel mean/std")
    axes[1].set_xlabel("Class pixel mean")
    axes[1].set_ylabel("Class pixel standard deviation")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    distribution_path = DIAGNOSTIC_ROOT / "pixel_distribution_comparison.png"
    fig.savefig(distribution_path, dpi=150)
    plt.close(fig)

    trial_overlaps = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            left_set = {(emotion, trial) for emotion in CLASSES for trial in FIXED_TRIAL_ASSIGNMENTS[left][emotion]}
            right_set = {(emotion, trial) for emotion in CLASSES for trial in FIXED_TRIAL_ASSIGNMENTS[right][emotion]}
            trial_overlaps[(left, right)] = left_set & right_set
    image_counts = {split: {emotion: len(records_by_class_split[split][emotion]) for emotion in CLASSES} for split in SPLITS}
    hash_overlaps = {
        (left, right): split_hashes[left] & split_hashes[right]
        for i, left in enumerate(SPLITS)
        for right in SPLITS[i + 1:]
    }
    if any(trial_overlaps.values()) or any(hash_overlaps.values()):
        raise RuntimeError(f"Fixed-scale leakage check failed: trials={trial_overlaps}, image hashes={hash_overlaps}")

    clip_rates = {}
    for split in SPLITS:
        clip_rates[split] = {}
        for emotion in CLASSES:
            stats = clip_stats[split][emotion]
            clip_rates[split][emotion] = (stats["below"] + stats["above"]) / max(stats["total"], 1)

    report_lines = [
        "# Fixed-scale, all-window STFT preprocessing diagnostic",
        "",
        "This is a new experiment. The original STFT images, current CRNN dataset, and global-scale experiment were not modified.",
        "",
        "## Source window counts",
        "",
        "| Split | Emotion | Trial groups | Windows per trial | Total |",
        "|---|---|---:|---|---:|",
    ]
    for split in SPLITS:
        for emotion in CLASSES:
            trials = FIXED_TRIAL_ASSIGNMENTS[split][emotion]
            per_trial = [f"{trial}={len(records_by_trial[(emotion, trial)])}" for trial in trials]
            report_lines.append(f"| {split} | {emotion} | {len(trials)} | {', '.join(per_trial)} | {image_counts[split][emotion]} |")
        report_lines.append(f"| **{split} total** |  |  |  | **{len(records_by_split[split])}** |")
    report_lines += [
        "",
        "## Fixed preprocessing",
        "",
        f"- STFT power is converted to log-power: `10*log10(|Z|^2 + 1e-12)`, using {eeg_source.FS} Hz sampling, {eeg_source.FREQ_MIN}–{eeg_source.FREQ_MAX} Hz, {eeg_source.WINDOW_SECONDS}s windows, and {eeg_source.OVERLAP_RATIO:.0%} overlap.",
        f"- Shared clipping interval is the training-only {CLIP_PERCENTILES[0]:g}th–{CLIP_PERCENTILES[1]:g}th percentile: **{scale_min:.8f} to {scale_max:.8f} dB**.",
        "- Pixels are clipped to this interval and normalized linearly to [0, 1]. Identical dB values receive identical normalized values and colormap colors in train, validation, and test.",
        "- All {0:,} valid windows were used; no window was duplicated or sampled away.".format(len(records)),
        "- Scaling statistics were computed from TRAIN trials only. Validation and test STFT values were excluded from percentile calculation.",
        "",
        "### Train-only numerical log-power summary",
        "",
        "| Emotion | Training windows | Mean dB | SD dB |",
        "|---|---:|---:|---:|",
    ]
    for emotion in CLASSES:
        values = np.concatenate(numeric_stats[emotion])
        report_lines.append(f"| {emotion} | {image_counts['train'][emotion]} | {values.mean():.6f} | {values.std():.6f} |")
    report_lines += ["", "### Clipping fractions after applying the train scale", "", "| Split | fear | happy | neutral | sad |", "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        report_lines.append("| {} | {} |".format(split, " | ".join(f"{100*clip_rates[split][emotion]:.3f}%" for emotion in CLASSES)))
    report_lines += ["", "### Rendered-pixel mean/std comparison (TRAIN only)", "", "| Emotion | Existing autoscaled pixel mean ± SD | Fixed-scale all-window pixel mean ± SD |", "|---|---:|---:|"]
    for emotion in CLASSES:
        old_moments = aggregate_image_moments(old_image_stats[emotion]["means"], old_image_stats[emotion]["stds"])
        fixed_moments = finish_pixel_moments(pixel_moments["train"][emotion])
        report_lines.append(f"| {emotion} | {old_moments['mean']:.3f} ± {old_moments['std']:.3f} | {fixed_moments['mean']:.3f} ± {fixed_moments['std']:.3f} |")
    report_lines += [
        "",
        "Fixed scaling preserves relative log-power amplitude within the shared clipping range: it does not independently stretch each window. Values outside the global training percentile interval are clipped; the table above quantifies this by split. This is physically more comparable, but retaining amplitude information does not itself prove that emotion is more separable.",
        "",
        "## Leakage and reproducibility checks",
        "",
        f"- Seed: {SEED}.",
        f"- Train/validation/test trial intersections: { {f'{a}∩{b}': len(v) for (a,b),v in trial_overlaps.items()} }.",
        f"- Exact image-hash intersections across splits: { {f'{a}∩{b}': len(v) for (a,b),v in hash_overlaps.items()} }.",
        "- The fixed trial assignment was compared with `data/processed/crnn_dataset/` before generation.",
        "- No validation/test values were used for scale selection; no validation/test balancing or augmentation was performed.",
        "",
        "## Artifacts",
        "",
        f"- Scaling parameters: `{PARAMETERS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Matched autoscaled/fixed-scale examples: `{comparison_path.relative_to(PROJECT_ROOT)}`",
        f"- Pixel distribution comparison: `{distribution_path.relative_to(PROJECT_ROOT)}`",
    ]
    report_path = DIAGNOSTIC_ROOT / "preprocessing_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\nDATASET COUNTS")
    for split in SPLITS:
        print(f"{split.upper()}: {len(records_by_split[split])}")
        for emotion in CLASSES:
            print(f"  {emotion:<8}: {image_counts[split][emotion]}")
    print("\nLEAKAGE CHECK")
    print(f"  Trial overlaps: { {f'{a}∩{b}': len(v) for (a,b),v in trial_overlaps.items()} }")
    print(f"  Exact image hash overlaps: { {f'{a}∩{b}': len(v) for (a,b),v in hash_overlaps.items()} }")
    print("  Train-only scaling: PASS")
    print("  All available windows used: PASS")
    print(f"\nPreprocessing report: {report_path}")
    print(f"Scaling parameters: {PARAMETERS_PATH}")
    print(f"Diagnostic figures: {comparison_path}, {distribution_path}")
    print(f"New CRNN dataset: {DATASET_OUTPUT_ROOT}")
    print("Original STFT and CRNN datasets were not modified.")


if __name__ == "__main__":
    main()
