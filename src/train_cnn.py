import os
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models

# Access callbacks through the TensorFlow package to avoid editor resolution
# issues while keeping the same runtime behavior.
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed/cnn_dataset"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "eeg_emotion_cnn.keras")


# ============================================================
# SETTINGS
# ============================================================

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 30

CLASS_NAMES = [
    "fear",
    "happy",
    "neutral",
    "sad"
]  


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

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
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    
], name="data_augmentation")


# ============================================================
# CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 4
    layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Classification
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(4, activation="softmax")
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n================ MODEL SUMMARY ================\n")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n================ STARTING TRAINING ================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================================
# TEST
# ============================================================

print("\n================ TESTING MODEL ================\n")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n================================================")
print("CNN TRAINING COMPLETE")
print("================================================")
print(f"Model saved to: {MODEL_PATH}")
# ============================================================
# PREDICTION DIAGNOSTIC
# ============================================================

import numpy as np
from collections import Counter

print("\n================================================")
print("PREDICTION DIAGNOSTIC")
print("================================================")

# Get predictions for validation set
val_predictions = model.predict(val_ds, verbose=0)
val_pred_classes = np.argmax(val_predictions, axis=1)

# Get predictions for test set
test_predictions = model.predict(test_ds, verbose=0)
test_pred_classes = np.argmax(test_predictions, axis=1)

print("\nValidation prediction distribution:")

for i, class_name in enumerate(CLASS_NAMES):
    count = np.sum(val_pred_classes == i)
    print(f"{class_name}: {count}")

print("\nTest prediction distribution:")

for i, class_name in enumerate(CLASS_NAMES):
    count = np.sum(test_pred_classes == i)
    print(f"{class_name}: {count}")

print("\n================================================")
print("EXPECTED CLASS ORDER")
print("================================================")

for i, class_name in enumerate(CLASS_NAMES):
    print(f"{i} = {class_name}")