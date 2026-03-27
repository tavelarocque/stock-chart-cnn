"""
train.py
--------
Trains the CNN model on the generated stock chart images.

Usage:
    python src/train.py

What it does:
  1. Loads chart images from results/charts/ using Keras image utilities
  2. Splits into train / validation / test sets (70 / 15 / 15)
  3. Applies data augmentation to the training set
  4. Trains the CNN with early stopping and learning rate scheduling
  5. Saves the best model to results/models/best_model.keras
  6. Plots training history and saves to results/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_cnn

# ── Hyperparameters ───────────────────────────────────────────────────────────

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

CHARTS_DIR = "results/charts"
MODELS_DIR = "results/models"


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_metadata(charts_dir: str) -> pd.DataFrame:
    """Load image metadata CSV or scan directory if CSV doesn't exist."""
    metadata_path = os.path.join(charts_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        return pd.read_csv(metadata_path)

    # Fallback: scan directory structure (UP/ and DOWN/ subfolders)
    records = []
    for label_name, label_val in [("UP", 1), ("DOWN", 0)]:
        folder = os.path.join(charts_dir, label_name)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".png"):
                records.append({
                    "image_path": os.path.join(folder, fname),
                    "label": label_val,
                    "label_name": label_name,
                })
    return pd.DataFrame(records)


def load_images(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load all images into memory as numpy arrays."""
    images, labels = [], []
    missing = 0
    for _, row in metadata.iterrows():
        path = row["image_path"]
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            img = keras.utils.load_img(path, target_size=IMAGE_SIZE)
            img_array = keras.utils.img_to_array(img)
            images.append(img_array)
            labels.append(int(row["label"]))
        except Exception:
            missing += 1

    if missing > 0:
        print(f"  ⚠ Skipped {missing} missing/corrupt images")

    return np.array(images), np.array(labels)


# ── Data Augmentation ─────────────────────────────────────────────────────────

def build_augmentation_layer() -> keras.Sequential:
    """Random augmentations applied only during training."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
    ], name="augmentation")


# ── Training ──────────────────────────────────────────────────────────────────

def train(charts_dir: str = CHARTS_DIR, models_dir: str = MODELS_DIR) -> None:
    os.makedirs(models_dir, exist_ok=True)
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("📂 Loading dataset...")
    metadata = load_metadata(charts_dir)
    if metadata.empty:
        print("❌ No data found. Run 'python src/data_collector.py' first.")
        sys.exit(1)

    print(f"   Total samples : {len(metadata):,}")
    print(f"   UP   (bullish): {(metadata['label'] == 1).sum():,}")
    print(f"   DOWN (bearish): {(metadata['label'] == 0).sum():,}")

    print("\n🖼  Loading images into memory (this may take a moment)...")
    X, y = load_images(metadata)
    print(f"   Image array shape: {X.shape}  |  Labels shape: {y.shape}")

    # ── Train / Val / Test Split ──────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=RANDOM_SEED, stratify=y_temp
        # 0.176 of 0.85 ≈ 0.15 of total → final split: 70 / 15 / 15
    )
    print(f"\n   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # ── Class Weights (handles class imbalance) ───────────────────────────────
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"\n   Class weights: DOWN={class_weights[0]:.3f}, UP={class_weights[1]:.3f}")

    # ── Build tf.data pipelines ───────────────────────────────────────────────
    augmentation = build_augmentation_layer()

    def augment(image, label):
        return augmentation(image, training=True), label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train), seed=RANDOM_SEED)
        .batch(BATCH_SIZE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Build Model ───────────────────────────────────────────────────────────
    print("\n🧠 Building model...")
    model = build_cnn(input_shape=(*IMAGE_SIZE, 3), learning_rate=LEARNING_RATE)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_model_path = os.path.join(models_dir, "best_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=10,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(os.path.join(models_dir, "training_log.csv")),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n🚀 Training for up to {EPOCHS} epochs (early stopping enabled)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # ── Evaluate on Test Set ─────────────────────────────────────────────────
    print("\n📊 Evaluating on hold-out test set...")
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    loss, accuracy, auc = model.evaluate(test_ds, verbose=0)
    print(f"   Test Loss    : {loss:.4f}")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"   Test AUC     : {auc:.4f}")

    # Save test arrays for use by predict.py
    np.save(os.path.join(models_dir, "X_test.npy"), X_test)
    np.save(os.path.join(models_dir, "y_test.npy"), y_test)

    # ── Plot Training History ─────────────────────────────────────────────────
    _plot_history(history, save_dir="results")
    print(f"\n✅ Training complete! Model saved to: {best_model_path}")


def _plot_history(history: keras.callbacks.History, save_dir: str = "results") -> None:
    """Save training curves as a PNG."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History — Stock Chart CNN", fontsize=14, fontweight="bold")

    metrics = [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")]
    colors = [("#2196F3", "#FF5722"), ("#4CAF50", "#F44336"), ("#9C27B0", "#FF9800")]

    for ax, (metric, title), (train_c, val_c) in zip(axes, metrics, colors):
        train_vals = history.history.get(metric, [])
        val_vals = history.history.get(f"val_{metric}", [])
        epochs = range(1, len(train_vals) + 1)
        ax.plot(epochs, train_vals, color=train_c, label="Train", linewidth=2)
        ax.plot(epochs, val_vals, color=val_c, label="Validation", linewidth=2, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Training history plot saved to: {out_path}")


if __name__ == "__main__":
    train()
