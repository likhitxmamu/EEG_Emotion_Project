"""Build a new CRNN dataset from globally scaled STFTs.

This script preserves the existing trial assignments by reading them from
data/processed/crnn_dataset/. It writes only to
data/processed/crnn_dataset_global/ and refuses to overwrite that output.
"""

import hashlib
import os
import re
import shutil
from collections import defaultdict

from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SOURCE_ROOT = os.path.join(DATA_ROOT, "processed", "stft_spectrograms_global")
REFERENCE_ROOT = os.path.join(DATA_ROOT, "processed", "crnn_dataset")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "processed", "crnn_dataset_global")

EMOTIONS = ("fear", "happy", "neutral", "sad")
SPLITS = ("train", "validation", "test")
SEED = 42
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FILENAME_PATTERN = re.compile(
    r"^(?P<trial>.+_trial\d+)_window_(?P<window>\d+)_STFT_RGB$",
    re.IGNORECASE,
)


def list_images(folder):
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS
    )


def trial_key(path):
    match = FILENAME_PATTERN.fullmatch(os.path.splitext(os.path.basename(path))[0])
    if not match:
        raise ValueError(f"Unexpected STFT filename; cannot identify trial: {path}")
    return match.group("trial")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    print("GLOBAL-SCALE CRNN DATASET PREPARATION")
    print(f"Seed: {SEED}")
    print(f"New STFT source: {SOURCE_ROOT}")
    print(f"Existing split reference (read-only): {REFERENCE_ROOT}")
    print(f"New CRNN output: {OUTPUT_ROOT}")

    if not os.path.isdir(SOURCE_ROOT):
        raise FileNotFoundError(f"New global-scale STFT source not found: {SOURCE_ROOT}")
    if not os.path.isdir(REFERENCE_ROOT):
        raise FileNotFoundError(f"Existing CRNN assignment reference not found: {REFERENCE_ROOT}")
    if os.path.lexists(OUTPUT_ROOT):
        raise FileExistsError(
            "Refusing to overwrite an existing dataset. Move to a new output path "
            f"only after preserving it: {OUTPUT_ROOT}"
        )

    assignments = {}
    expected_images = set()
    for split in SPLITS:
        for emotion in EMOTIONS:
            ref_dir = os.path.join(REFERENCE_ROOT, split, emotion)
            if not os.path.isdir(ref_dir):
                raise FileNotFoundError(f"Missing reference class directory: {ref_dir}")
            for path in list_images(ref_dir):
                key = (emotion, os.path.basename(path))
                expected_images.add(key)
                trial = trial_key(path)
                assignment_key = (emotion, trial)
                previous = assignments.setdefault(assignment_key, split)
                if previous != split:
                    raise RuntimeError(
                        f"Existing reference leaks trial {trial} across {previous} and {split}."
                    )

    source_files = {}
    for emotion in EMOTIONS:
        source_dir = os.path.join(SOURCE_ROOT, emotion)
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Missing global STFT class directory: {source_dir}")
        for path in list_images(source_dir):
            key = (emotion, os.path.basename(path))
            if key in source_files:
                raise RuntimeError(f"Duplicate source basename within class: {key}")
            source_files[key] = path

    actual_images = set(source_files)
    if actual_images != expected_images:
        raise RuntimeError(
            "The regenerated global STFT filenames do not exactly match the "
            "existing split reference; refusing to alter trial assignments. "
            f"Missing={len(expected_images - actual_images)}, "
            f"unexpected={len(actual_images - expected_images)}"
        )

    plan = {split: {emotion: [] for emotion in EMOTIONS} for split in SPLITS}
    for key, source_path in source_files.items():
        emotion, basename = key
        split = assignments[(emotion, trial_key(source_path))]
        plan[split][emotion].append(source_path)
    for split in SPLITS:
        for emotion in EMOTIONS:
            plan[split][emotion].sort()
            if not plan[split][emotion]:
                raise RuntimeError(f"No {emotion} images assigned to {split}.")

    # Validate image readability, dimensions, and modes before creating output.
    dimensions = set()
    modes = set()
    for path in source_files.values():
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            dimensions.add(image.size)
            modes.add(image.mode)
    if len(dimensions) != 1 or len(modes) != 1:
        raise RuntimeError(f"Inconsistent source image format: sizes={dimensions}, modes={modes}")

    os.makedirs(OUTPUT_ROOT)
    copied_hashes = {split: set() for split in SPLITS}
    copied_trials = {split: set() for split in SPLITS}
    counts = {split: {emotion: 0 for emotion in EMOTIONS} for split in SPLITS}

    for split in SPLITS:
        for emotion in EMOTIONS:
            destination_dir = os.path.join(OUTPUT_ROOT, split, emotion)
            os.makedirs(destination_dir)
            for source_path in plan[split][emotion]:
                output_path = os.path.join(destination_dir, os.path.basename(source_path))
                shutil.copy2(source_path, output_path)
                copied_hashes[split].add(sha256(output_path))
                copied_trials[split].add((emotion, trial_key(source_path)))
                counts[split][emotion] += 1

    print("\nDATASET SPLIT SUMMARY")
    for split in SPLITS:
        print(f"\n{split.upper()}")
        for emotion in EMOTIONS:
            print(f"  {emotion:<8}: {counts[split][emotion]}")
        print(f"  TOTAL    : {sum(counts[split].values())}")

    print("\nTRIAL SPLIT CHECK")
    for split in SPLITS:
        print(f"{split.title()} trials:")
        for emotion in EMOTIONS:
            trials = sorted(trial for cls, trial in copied_trials[split] if cls == emotion)
            print(f"  {emotion:<8}: {', '.join(trials)}")

    trial_overlaps = {
        (a, b): copied_trials[a] & copied_trials[b]
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1:]
    }
    hash_overlaps = {
        (a, b): copied_hashes[a] & copied_hashes[b]
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1:]
    }
    print("\nLEAKAGE / INTEGRITY CHECK")
    for pair in trial_overlaps:
        print(f"  {pair[0]} ∩ {pair[1]} trial groups: {len(trial_overlaps[pair])}")
        print(f"  {pair[0]} ∩ {pair[1]} exact image hashes: {len(hash_overlaps[pair])}")
    print(f"  Image size: {next(iter(dimensions))}; mode: {next(iter(modes))}")
    if any(trial_overlaps.values()) or any(hash_overlaps.values()):
        raise RuntimeError("Leakage/integrity check failed.")
    print("  Verification: PASS")
    print(f"\nNew dataset created: {OUTPUT_ROOT}")
    print("Existing STFT and CRNN datasets were not modified.")


if __name__ == "__main__":
    main()
