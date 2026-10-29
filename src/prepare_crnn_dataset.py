"""
Prepare the CRNN dataset by assigning complete EEG trials to splits.

Source images are copied from:
    data/processed/stft_spectrograms/<emotion>/

Output:
    data/processed/crnn_dataset/{train,validation,test}/<emotion>/

Trial group IDs are read from the existing filenames, for example:
    1_20160518_trial11_window_0001_STFT_RGB.png
The full trial group key is the filename prefix before "_window_":
    1_20160518_trial11

The script uses all source images. It never modifies the STFT source folders.
Only the generated CRNN output directory is cleared and recreated.
"""

import itertools
import math
import os
import random
import re
import shutil


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

SOURCE_ROOT = os.path.join(
    DATA_ROOT,
    "processed",
    "stft_spectrograms",
)

OUTPUT_ROOT = os.path.join(
    DATA_ROOT,
    "processed",
    "crnn_dataset",
)


# ============================================================
# SETTINGS
# ============================================================

EMOTIONS = ("fear", "happy", "neutral", "sad")
SPLITS = ("train", "validation", "test")

# These are approximate targets only. Complete trials are never divided.
TARGET_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

RANDOM_SEED = 42

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Existing audited filename format:
# 1_20160518_trial11_window_0001_STFT_RGB.png
TRIAL_FILENAME_PATTERN = re.compile(
    r"^(?P<trial_group>.+_trial(?P<trial_id>\d+))"
    r"_window_(?P<window_id>\d+)_STFT_RGB$",
    re.IGNORECASE,
)


# ============================================================
# SOURCE DISCOVERY AND TRIAL GROUPING
# ============================================================

def find_images(emotion):
    """Return the sorted source image paths for one emotion."""
    emotion_folder = os.path.join(SOURCE_ROOT, emotion)

    if not os.path.isdir(emotion_folder):
        raise FileNotFoundError(
            f"Missing source class folder: {emotion_folder}"
        )

    images = []

    for root, dirs, files in os.walk(emotion_folder):
        dirs.sort()

        for filename in sorted(files):
            extension = os.path.splitext(filename)[1].lower()

            if extension in SUPPORTED_EXTENSIONS:
                images.append(os.path.join(root, filename))

    images = sorted(set(os.path.realpath(path) for path in images))

    if not images:
        raise ValueError(
            f"No supported spectrogram images found in: {emotion_folder}"
        )

    return images


def parse_trial_group(image_path):
    """
    Parse the existing recording/trial group and numeric trial ID.

    The full prefix is retained so that a repeated trial number from a
    different recording/session would remain a separate group.
    """
    filename = os.path.basename(image_path)
    stem, extension = os.path.splitext(filename)

    if extension.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path}")

    match = TRIAL_FILENAME_PATTERN.fullmatch(stem)

    if match is None:
        raise ValueError(
            "Could not parse the audited filename pattern "
            "'<recording>_trial<ID>_window_<ID>_STFT_RGB': "
            f"{image_path}"
        )

    return match.group("trial_group"), match.group("trial_id")


def group_source_images():
    """Group all input images by emotion and full recording/trial key."""
    groups_by_emotion = {emotion: {} for emotion in EMOTIONS}
    group_owner = {}
    basename_owner = {}

    for emotion in EMOTIONS:
        for image_path in find_images(emotion):
            trial_group, _ = parse_trial_group(image_path)

            previous_owner = group_owner.get(trial_group)
            if previous_owner is not None and previous_owner != emotion:
                raise ValueError(
                    f"Trial group {trial_group!r} appears in both "
                    f"{previous_owner!r} and {emotion!r} folders."
                )
            group_owner[trial_group] = emotion

            # The output follows ImageFolder's class-directory layout and
            # therefore stores each image by basename within its class.
            basename = os.path.basename(image_path)
            basename_key = (emotion, basename)
            previous_path = basename_owner.get(basename_key)
            if previous_path is not None and previous_path != image_path:
                raise ValueError(
                    "Two source images in the same class have the same "
                    f"basename and would collide in the output: {basename}"
                )
            basename_owner[basename_key] = image_path

            groups_by_emotion[emotion].setdefault(trial_group, []).append(
                image_path
            )

    return groups_by_emotion


# ============================================================
# TRIAL-LEVEL SPLIT PLANNING
# ============================================================

def choose_trial_splits(groups_by_emotion):
    """
    Assign each whole trial to one split.

    For each class, all possible whole-trial assignments are compared to
    approximate the target image ratios. Each split must receive at least
    one trial for every class. The fixed-seed RNG resolves equal-scoring
    assignments reproducibly.
    """
    rng = random.Random(RANDOM_SEED)
    best_candidates_by_emotion = {}

    for emotion in EMOTIONS:
        trial_groups = sorted(groups_by_emotion[emotion])

        if len(trial_groups) < len(SPLITS):
            raise ValueError(
                f"{emotion} has {len(trial_groups)} trial groups; "
                f"at least {len(SPLITS)} are needed to put this class "
                "in train, validation, and test without splitting trials."
            )

        # Shuffle only the candidate enumeration order. The score selects
        # the closest whole-trial split; the seed makes ties deterministic.
        rng.shuffle(trial_groups)

        group_sizes = {
            group: len(groups_by_emotion[emotion][group])
            for group in trial_groups
        }

        total_images = sum(group_sizes.values())
        target_counts = {
            split: total_images * TARGET_SPLIT_RATIOS[split]
            for split in SPLITS
        }

        best_score = math.inf
        best_assignments = []

        for codes in itertools.product(range(len(SPLITS)), repeat=len(trial_groups)):
            if set(codes) != set(range(len(SPLITS))):
                continue

            candidate = {
                group: SPLITS[code]
                for group, code in zip(trial_groups, codes)
            }

            actual_counts = {split: 0 for split in SPLITS}
            for group, split in candidate.items():
                actual_counts[split] += group_sizes[group]

            # Ratios are approximate targets, not constraints. The score
            # compares each class's split proportions to those targets.
            score = sum(
                (
                    (actual_counts[split] - target_counts[split])
                    / max(target_counts[split], 1.0)
                ) ** 2
                for split in SPLITS
            )

            if score < best_score and not math.isclose(
                score, best_score, rel_tol=1e-12, abs_tol=1e-12
            ):
                best_score = score
                best_assignments = [candidate]
            elif math.isclose(
                score, best_score, rel_tol=1e-12, abs_tol=1e-12
            ):
                best_assignments.append(candidate)

        best_candidates_by_emotion[emotion] = best_assignments

    # Among assignments with the best per-class ratio score, choose a
    # combination that also balances the number of images per class within
    # each split. This resolves ties without splitting a trial or dropping
    # images. If still tied, prefer the larger independent test set.
    best_joint_score = None
    best_joint_assignments = []

    candidate_lists = [
        best_candidates_by_emotion[emotion]
        for emotion in EMOTIONS
    ]

    for candidate_combination in itertools.product(*candidate_lists):
        counts_by_split_and_class = {
            split: {emotion: 0 for emotion in EMOTIONS}
            for split in SPLITS
        }

        for emotion, candidate in zip(EMOTIONS, candidate_combination):
            for trial_group, split in candidate.items():
                counts_by_split_and_class[split][emotion] += len(
                    groups_by_emotion[emotion][trial_group]
                )

        split_totals = {
            split: sum(counts_by_split_and_class[split].values())
            for split in SPLITS
        }

        class_balance_score = 0.0
        for split in SPLITS:
            mean_class_count = split_totals[split] / len(EMOTIONS)
            class_balance_score += sum(
                (
                    (counts_by_split_and_class[split][emotion] - mean_class_count)
                    / max(mean_class_count, 1.0)
                ) ** 2
                for emotion in EMOTIONS
            )

        total_images = sum(split_totals.values())
        overall_ratio_score = sum(
            (
                (
                    split_totals[split]
                    - total_images * TARGET_SPLIT_RATIOS[split]
                )
                / max(total_images * TARGET_SPLIT_RATIOS[split], 1.0)
            ) ** 2
            for split in SPLITS
        )

        joint_score = (
            class_balance_score,
            overall_ratio_score,
            -split_totals["test"],
        )

        if best_joint_score is None or joint_score < best_joint_score:
            best_joint_score = joint_score
            best_joint_assignments = [candidate_combination]
        elif joint_score == best_joint_score:
            best_joint_assignments.append(candidate_combination)

    selected_by_emotion = rng.choice(best_joint_assignments)
    trial_to_split = {}

    for candidate in selected_by_emotion:
        for trial_group, split in candidate.items():
            if trial_group in trial_to_split:
                raise ValueError(
                    f"Trial group assigned more than once: {trial_group}"
                )
            trial_to_split[trial_group] = split

    return trial_to_split


def build_split_plan(groups_by_emotion, trial_to_split):
    """Create split/class source-image lists from the whole-trial assignment."""
    plan = {
        split: {emotion: [] for emotion in EMOTIONS}
        for split in SPLITS
    }

    for emotion in EMOTIONS:
        for trial_group, image_paths in groups_by_emotion[emotion].items():
            split = trial_to_split[trial_group]
            plan[split][emotion].extend(image_paths)

    for split in SPLITS:
        for emotion in EMOTIONS:
            plan[split][emotion].sort()

    return plan


def print_trial_assignments(groups_by_emotion, trial_to_split):
    print("\nTRIAL ASSIGNMENTS (full recording/trial keys)")

    for split in SPLITS:
        print(f"\n{split.upper()} trials:")

        for emotion in EMOTIONS:
            assigned = sorted(
                group
                for group in groups_by_emotion[emotion]
                if trial_to_split[group] == split
            )

            details = []
            for group in assigned:
                _, trial_id = parse_trial_group(
                    groups_by_emotion[emotion][group][0]
                )
                details.append(f"{group} (trial {trial_id})")

            print(f"  {emotion:<8}: {', '.join(details) or 'NONE'}")


# ============================================================
# SAFE OUTPUT RESET AND COPYING
# ============================================================

def validate_output_path():
    """Refuse to clear anything except the intended CRNN output directory."""
    expected_output = os.path.abspath(
        os.path.join(DATA_ROOT, "processed", "crnn_dataset")
    )
    output_path = os.path.abspath(OUTPUT_ROOT)
    source_path = os.path.abspath(SOURCE_ROOT)

    if output_path != expected_output:
        raise RuntimeError(
            "Refusing to clear an unexpected output path: "
            f"{output_path}"
        )

    if (
        output_path == source_path
        or output_path.startswith(source_path + os.sep)
        or source_path.startswith(output_path + os.sep)
    ):
        raise RuntimeError(
            "Refusing to clear output because it overlaps the STFT source: "
            f"{source_path}"
        )

    if os.path.islink(output_path):
        raise RuntimeError(
            f"Refusing to clear a symlink at the CRNN output path: {output_path}"
        )


def clear_old_dataset():
    """Clear and recreate only data/processed/crnn_dataset/."""
    validate_output_path()

    print("\nCRNN OUTPUT PATH TO CLEAR (generated dataset only):")
    print(f"  {OUTPUT_ROOT}")
    print("STFT source path to preserve:")
    print(f"  {SOURCE_ROOT}")

    if os.path.lexists(OUTPUT_ROOT):
        if not os.path.isdir(OUTPUT_ROOT):
            raise RuntimeError(
                f"CRNN output path exists but is not a directory: {OUTPUT_ROOT}"
            )

        print(
            f"\nClearing exactly this CRNN output directory:\n  {OUTPUT_ROOT}"
        )
        shutil.rmtree(OUTPUT_ROOT)

    for split in SPLITS:
        for emotion in EMOTIONS:
            os.makedirs(
                os.path.join(OUTPUT_ROOT, split, emotion),
                exist_ok=True,
            )


def copy_split_plan(plan):
    """Copy the planned files without moving or modifying source images."""
    for split in SPLITS:
        for emotion in EMOTIONS:
            destination = os.path.join(OUTPUT_ROOT, split, emotion)

            for source_path in plan[split][emotion]:
                destination_path = os.path.join(
                    destination,
                    os.path.basename(source_path),
                )
                shutil.copy2(source_path, destination_path)


# ============================================================
# VERIFICATION AND SUMMARY
# ============================================================

def verify_dataset(plan, groups_by_emotion, trial_to_split):
    """Verify class presence, source-image uniqueness, and trial isolation."""
    counts = {
        split: {emotion: 0 for emotion in EMOTIONS}
        for split in SPLITS
    }
    source_images_by_split = {split: set() for split in SPLITS}
    observed_trials_by_split = {split: set() for split in SPLITS}

    for split in SPLITS:
        for emotion in EMOTIONS:
            output_dir = os.path.join(OUTPUT_ROOT, split, emotion)

            if not os.path.isdir(output_dir):
                raise RuntimeError(f"Missing output class folder: {output_dir}")

            actual_names = {
                filename
                for filename in os.listdir(output_dir)
                if os.path.splitext(filename)[1].lower()
                in SUPPORTED_EXTENSIONS
            }
            expected_names = {
                os.path.basename(path)
                for path in plan[split][emotion]
            }

            if actual_names != expected_names:
                missing = sorted(expected_names - actual_names)
                unexpected = sorted(actual_names - expected_names)
                raise RuntimeError(
                    f"Output mismatch in {split}/{emotion}. "
                    f"Missing={missing}; unexpected={unexpected}"
                )

            counts[split][emotion] = len(actual_names)

            for source_path in plan[split][emotion]:
                source_identity = os.path.realpath(source_path)

                if source_identity in source_images_by_split[split]:
                    raise RuntimeError(
                        "A source image was scheduled more than once in "
                        f"{split}: {source_path}"
                    )
                source_images_by_split[split].add(source_identity)

                trial_group, _ = parse_trial_group(source_path)
                observed_trials_by_split[split].add(trial_group)

                if trial_to_split[trial_group] != split:
                    raise RuntimeError(
                        f"Trial leakage found: {trial_group} is assigned to "
                        f"{trial_to_split[trial_group]} but copied to {split}."
                    )

    for split in SPLITS:
        missing_classes = [
            emotion for emotion in EMOTIONS
            if counts[split][emotion] == 0
        ]
        if missing_classes:
            raise RuntimeError(
                f"Required class folders are empty in {split}: "
                f"{missing_classes}"
            )

    trial_overlaps = {
        ("train", "validation"):
            observed_trials_by_split["train"]
            & observed_trials_by_split["validation"],
        ("train", "test"):
            observed_trials_by_split["train"]
            & observed_trials_by_split["test"],
        ("validation", "test"):
            observed_trials_by_split["validation"]
            & observed_trials_by_split["test"],
    }

    for split_pair, overlap in trial_overlaps.items():
        if overlap:
            raise RuntimeError(
                f"Trial leakage between {split_pair[0]} and {split_pair[1]}: "
                f"{sorted(overlap)}"
            )

    source_overlaps = {
        ("train", "validation"):
            source_images_by_split["train"]
            & source_images_by_split["validation"],
        ("train", "test"):
            source_images_by_split["train"]
            & source_images_by_split["test"],
        ("validation", "test"):
            source_images_by_split["validation"]
            & source_images_by_split["test"],
    }

    for split_pair, overlap in source_overlaps.items():
        if overlap:
            raise RuntimeError(
                f"Source-image overlap between {split_pair[0]} and "
                f"{split_pair[1]}: {sorted(overlap)}"
            )

    # Ensure every source group was assigned and copied exactly once.
    expected_trial_groups = {
        trial_group
        for emotion_groups in groups_by_emotion.values()
        for trial_group in emotion_groups
    }
    observed_trial_groups = set().union(
        *observed_trials_by_split.values()
    )

    if expected_trial_groups != observed_trial_groups:
        raise RuntimeError(
            "Trial-group coverage mismatch. "
            f"Missing={sorted(expected_trial_groups - observed_trial_groups)}; "
            f"unexpected={sorted(observed_trial_groups - expected_trial_groups)}"
        )

    print("\nDATASET SPLIT SUMMARY")
    for split in SPLITS:
        print(f"\n{split.upper()}")
        split_total = 0

        for emotion in EMOTIONS:
            count = counts[split][emotion]
            print(f"  {emotion:<8}: {count}")
            split_total += count

        print(f"  TOTAL    : {split_total}")

    print("\nTRIAL SPLIT CHECK")
    for split in SPLITS:
        print(f"\n{split.title()} trials:")
        for emotion in EMOTIONS:
            groups = sorted(
                group
                for group in observed_trials_by_split[split]
                if group in groups_by_emotion[emotion]
            )
            print(f"  {emotion:<8}: {', '.join(groups) or 'NONE'}")

    print("\nLEAKAGE CHECK:")
    print(
        "  Train ∩ Validation = "
        f"{len(trial_overlaps[('train', 'validation')])}"
    )
    print(
        "  Train ∩ Test       = "
        f"{len(trial_overlaps[('train', 'test')])}"
    )
    print(
        "  Validation ∩ Test  = "
        f"{len(trial_overlaps[('validation', 'test')])}"
    )
    print(
        "  Source-image overlaps across split pairs = "
        f"{sum(len(overlap) for overlap in source_overlaps.values())}"
    )
    print("  All four classes exist in train, validation, and test.")
    print("  All source images were copied exactly once.")
    print("  Verification: PASS")

    return counts


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("EEG EMOTION PROJECT - TRIAL-LEVEL CRNN DATASET PREPARATION")
    print("=" * 72)
    print(f"Source STFT images: {SOURCE_ROOT}")
    print(f"CRNN output path:   {OUTPUT_ROOT}")
    print(f"Random seed:        {RANDOM_SEED}")
    print("Target ratios (approximate only): train 70%, validation 15%, test 15%")
    print("All images from one recording/trial group will remain in one split.")

    if not os.path.isdir(SOURCE_ROOT):
        raise FileNotFoundError(
            f"STFT source folder not found: {SOURCE_ROOT}"
        )

    groups_by_emotion = group_source_images()

    print("\nAVAILABLE SOURCE TRIAL GROUPS")
    for emotion in EMOTIONS:
        trial_groups = sorted(groups_by_emotion[emotion])
        image_count = sum(
            len(group_images)
            for group_images in groups_by_emotion[emotion].values()
        )
        print(
            f"  {emotion:<8}: {len(trial_groups)} trials, "
            f"{image_count} images"
        )
        print(f"           {', '.join(trial_groups)}")

    trial_to_split = choose_trial_splits(groups_by_emotion)
    plan = build_split_plan(groups_by_emotion, trial_to_split)

    print_trial_assignments(groups_by_emotion, trial_to_split)

    # Plan and source filenames are fully validated before the old generated
    # CRNN output is cleared. The original STFT source is never touched.
    clear_old_dataset()
    copy_split_plan(plan)
    verify_dataset(plan, groups_by_emotion, trial_to_split)

    print("\nCRNN DATASET PREPARATION COMPLETE")
    print(f"Generated dataset: {OUTPUT_ROOT}")
    print("Original STFT spectrograms were not modified.")
    print("CRNN training was not started.")


if __name__ == "__main__":
    main()
