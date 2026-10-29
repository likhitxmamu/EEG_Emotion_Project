import os
import numpy as np
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed/cnn_dataset"

VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_PATH = "models/eeg_emotion_cnn.keras"

# ============================================================
# SETTINGS
# ============================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

CLASS_NAMES = [
    "fear",
    "happy",
    "neutral",
    "sad"
]

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning validation predictions...")

val_predictions = model.predict(val_ds, verbose=1)
val_pred_classes = np.argmax(val_predictions, axis=1)

print("\nRunning test predictions...")

test_predictions = model.predict(test_ds, verbose=1)
test_pred_classes = np.argmax(test_predictions, axis=1)

# ============================================================
# ACTUAL LABELS
# ============================================================

val_true_classes = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for _, labels in val_ds
])

test_true_classes = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for _, labels in test_ds
])

# ============================================================
# VALIDATION DISTRIBUTION
# ============================================================

print("\n================================================")
print("VALIDATION PREDICTION DISTRIBUTION")
print("================================================")

for i, class_name in enumerate(CLASS_NAMES):
    predicted = np.sum(val_pred_classes == i)
    actual = np.sum(val_true_classes == i)

    print(
        f"{class_name:8s} | "
        f"Actual: {actual:3d} | "
        f"Predicted: {predicted:3d}"
    )

# ============================================================
# TEST DISTRIBUTION
# ============================================================

print("\n================================================")
print("TEST PREDICTION DISTRIBUTION")
print("================================================")

for i, class_name in enumerate(CLASS_NAMES):
    predicted = np.sum(test_pred_classes == i)
    actual = np.sum(test_true_classes == i)

    print(
        f"{class_name:8s} | "
        f"Actual: {actual:3d} | "
        f"Predicted: {predicted:3d}"
    )

# ============================================================
# ACCURACY
# ============================================================

val_accuracy = np.mean(val_pred_classes == val_true_classes)
test_accuracy = np.mean(test_pred_classes == test_true_classes)

print("\n================================================")
print("ACCURACY CHECK")
print("================================================")

print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy:       {test_accuracy * 100:.2f}%")

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n================================================")
print("VALIDATION CONFUSION MATRIX")
print("================================================")

cm_val = tf.math.confusion_matrix(
    val_true_classes,
    val_pred_classes,
    num_classes=len(CLASS_NAMES)
).numpy()

print("Rows = Actual")
print("Columns = Predicted")
print()
print("          " + " ".join(f"{x:>8s}" for x in CLASS_NAMES))

for i, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name:8s} "
        + " ".join(f"{x:8d}" for x in cm_val[i])
    )

print("\n================================================")
print("TEST CONFUSION MATRIX")
print("================================================")

cm_test = tf.math.confusion_matrix(
    test_true_classes,
    test_pred_classes,
    num_classes=len(CLASS_NAMES)
).numpy()

print("Rows = Actual")
print("Columns = Predicted")
print()
print("          " + " ".join(f"{x:>8s}" for x in CLASS_NAMES))

for i, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name:8s} "
        + " ".join(f"{x:8d}" for x in cm_test[i])
    )