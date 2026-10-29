import os
import time
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_DIR = "data/processed/crnn_dataset"
MODEL_DIR = "models"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_transformer_model.pth"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_transformer_model.pth"
)

BATCH_SIZE = 32
IMAGE_SIZE = 224

NUM_CLASSES = 4
CLASS_NAMES = ["fear", "happy", "neutral", "sad"]

EPOCHS = 40
LEARNING_RATE = 0.0003

RANDOM_SEED = 42


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 3. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print()
print("=" * 70)
print("TRANSFORMER EEG EMOTION CLASSIFICATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_DIR}")
print()


# ============================================================
# 4. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_DIR}"
    )

for split in ["train", "validation", "test"]:
    split_path = os.path.join(DATASET_DIR, split)

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing dataset split: {split_path}"
        )


# ============================================================
# 5. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.20,
        scale=(0.02, 0.08),
        ratio=(0.5, 2.0),
        value=0
    ),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# 6. LOAD DATASETS
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "validation")
test_dir = os.path.join(DATASET_DIR, "test")


train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=eval_transform
)


print()
print("Classes:")
print(train_dataset.classes)

print()

print(f"Training images   : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")
print(f"Testing images    : {len(test_dataset)}")

print()

print("Class mapping:")

for index, name in enumerate(train_dataset.classes):
    print(f"{index} -> {name}")


# ============================================================
# 7. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. PATCH EMBEDDING
# ============================================================

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=1,
        embedding_dim=128
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (
            image_size // patch_size
        ) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.projection(x)

        # [B, E, H, W]
        x = x.flatten(2)

        # [B, E, N]
        x = x.transpose(1, 2)

        # [B, N, E]

        return x


# ============================================================
# 9. EEG TRANSFORMER
# ============================================================

class EEGTransformer(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        num_classes=4,
        dropout=0.2
    ):

        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=1,
            embedding_dim=embedding_dim
        )

        num_patches = (
            image_size // patch_size
        ) ** 2

        # CLS token
        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embedding_dim
            )
        )

        # Positional embedding
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embedding_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            embedding_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                embedding_dim,
                128
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                128,
                num_classes
            )
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        x = x + self.position_embedding

        x = self.transformer_encoder(x)

        x = self.norm(x[:, 0])

        x = self.classifier(x)

        return x


# ============================================================
# 10. CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING TRANSFORMER MODEL")
print("=" * 70)

model = EEGTransformer(
    image_size=IMAGE_SIZE,
    patch_size=16,
    embedding_dim=64,
    num_heads=4,
    num_layers=2,
    num_classes=NUM_CLASSES,
    dropout=0.1
)

model = model.to(DEVICE)

print(model)


# ============================================================
# 11. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ============================================================
# 12. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 13. VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print()
print("=" * 70)
print("STARTING TRANSFORMER TRAINING")
print("=" * 70)

start_time = time.time()

best_val_accuracy = 0.0
best_model_weights = copy.deepcopy(
    model.state_dict()
)

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()

    scheduler.step(val_accuracy)

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy * 100:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"  ✓ Best model saved "
            f"({best_val_accuracy * 100:.2f}%)"
        )


# ============================================================
# 15. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)


# ============================================================
# 17. TEST MODEL
# ============================================================

print()
print("=" * 70)
print("TESTING TRANSFORMER")
print("=" * 70)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print()

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=train_dataset.classes,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print()

print(cm)


# ============================================================
# 20. PER-EMOTION ACCURACY
# ============================================================

print()
print("=" * 70)
print("PER-EMOTION ACCURACY")
print("=" * 70)

for i, emotion in enumerate(
    train_dataset.classes
):

    class_total = cm[i].sum()

    if class_total > 0:

        class_accuracy = (
            cm[i, i] / class_total
        )

        print(
            f"{emotion:10s}: "
            f"{class_accuracy * 100:.2f}%"
        )


# ============================================================
# 21. TRAINING SUMMARY
# ============================================================

training_time = (
    time.time() - start_time
) / 60


print()
print("=" * 70)
print("TRANSFORMER TRAINING COMPLETE")
print("=" * 70)

print()

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Final test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{macro_f1:.4f}"
)

print(
    f"Training time: "
    f"{training_time:.2f} minutes"
)

print()

print("Best model:")
print(BEST_MODEL_PATH)

print()

print("Final model:")
print(FINAL_MODEL_PATH)

print()

print("=" * 70)
print("SUCCESS")
print("=" * 70)
