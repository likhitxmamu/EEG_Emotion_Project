"""Create a reproducible train/validation/test split for STFT images.

The source spectrogram folders are left untouched. Images are copied into a
separate dataset directory so the Transformer can use the expected split
layout without changing the original STFT output.
"""

import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "stft_spectrograms"
DEST_DIR = PROJECT_ROOT / "data" / "processed" / "stft_transformer_dataset"

CLASSES = ("fear", "happy", "neutral", "sad")
SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"STFT source directory not found: {SOURCE_DIR}")

    rng = random.Random(RANDOM_SEED)
    split_files = {split: {class_name: [] for class_name in CLASSES} for split in SPLITS}

    for class_name in CLASSES:
        source_class_dir = SOURCE_DIR / class_name
        if not source_class_dir.is_dir():
            raise FileNotFoundError(f"Missing source class directory: {source_class_dir}")

        images = sorted(
            path for path in source_class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No spectrogram images found in {source_class_dir}")

        rng.shuffle(images)
        train_end = int(len(images) * TRAIN_RATIO)
        validation_end = train_end + int(len(images) * VALIDATION_RATIO)
        split_files["train"][class_name] = images[:train_end]
        split_files["validation"][class_name] = images[train_end:validation_end]
        split_files["test"][class_name] = images[validation_end:]

    # Reject source basenames that would make the requested cross-split filename
    # check ambiguous. The current source set has unique basenames.
    source_names = [path.name for classes in split_files.values() for paths in classes.values() for path in paths]
    if len(source_names) != len(set(source_names)):
        raise ValueError("Source spectrogram filenames must be unique across classes.")

    expected_paths = {
        Path(split) / class_name / source_path.name
        for split, classes in split_files.items()
        for class_name, paths in classes.items()
        for source_path in paths
    }
    if DEST_DIR.exists():
        existing_paths = {
            path.relative_to(DEST_DIR)
            for path in DEST_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        stale_paths = existing_paths - expected_paths
        if stale_paths:
            raise RuntimeError(
                "The split destination contains image files outside this split. "
                "Move or remove the stale files before rebuilding: "
                f"{sorted(str(path) for path in stale_paths)[:10]}"
            )

    for split in SPLITS:
        for class_name in CLASSES:
            (DEST_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for class_name in CLASSES:
            destination_class_dir = DEST_DIR / split / class_name
            for source_path in split_files[split][class_name]:
                shutil.copy2(source_path, destination_class_dir / source_path.name)

    split_names = {
        split: {
            path.name
            for class_name in CLASSES
            for path in (DEST_DIR / split / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        for split in SPLITS
    }

    for split in SPLITS:
        found_classes = {
            path.name for path in (DEST_DIR / split).iterdir() if path.is_dir()
        }
        if found_classes != set(CLASSES):
            raise RuntimeError(f"Unexpected class folders in {DEST_DIR / split}: {found_classes}")

    for index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[index + 1:]:
            overlap = split_names[first_split] & split_names[second_split]
            if overlap:
                raise RuntimeError(
                    f"Filenames appear in both {first_split} and {second_split}: "
                    f"{sorted(overlap)[:10]}"
                )

    print(f"Dataset location: {DEST_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Random seed: {RANDOM_SEED}")
    print("Image counts by class:")
    for split in SPLITS:
        print(f"\n{split}:")
        for class_name in CLASSES:
            count = sum(
                1 for path in (DEST_DIR / split / class_name).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            print(f"  {class_name}: {count}")
    print("\nFilename overlap across train/validation/test: none")


if __name__ == "__main__":
    main()
