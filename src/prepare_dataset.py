import os
import shutil
import random

SOURCE_DIR = "data/processed/spectrograms"
DEST_DIR = "data/processed/cnn_dataset"

CLASSES = ["fear", "happy", "neutral", "sad"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# Create destination folders
for split in ["train", "val", "test"]:
    for emotion in CLASSES:
        os.makedirs(
            os.path.join(DEST_DIR, split, emotion),
            exist_ok=True
        )

# Process each emotion
for emotion in CLASSES:

    source_folder = os.path.join(SOURCE_DIR, emotion)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(".png")
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"\n{emotion.upper()}")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")

    # Copy files
    for filename in train_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(DEST_DIR, "train", emotion, filename)
        )

    for filename in val_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(DEST_DIR, "val", emotion, filename)
        )

    for filename in test_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(DEST_DIR, "test", emotion, filename)
        )

print("\n======================================")
print("CNN DATASET PREPARATION COMPLETE")
print("======================================")
print(f"Dataset location: {DEST_DIR}")