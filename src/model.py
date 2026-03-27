"""
model.py
--------
Defines the Convolutional Neural Network (CNN) architecture for classifying
stock chart images as UP or DOWN movements.

Architecture overview:
  Input (64x64 RGB image)
    → Conv Block 1: 32 filters, 3x3, ReLU + MaxPool + Dropout
    → Conv Block 2: 64 filters, 3x3, ReLU + MaxPool + Dropout
    → Conv Block 3: 128 filters, 3x3, ReLU + MaxPool + Dropout
    → Flatten
    → Dense 256 + ReLU + Dropout
    → Dense 1 + Sigmoid (binary classification: UP vs DOWN)
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers


def build_cnn(input_shape: tuple = (64, 64, 3), learning_rate: float = 1e-4) -> keras.Model:
    """
    Build and compile the CNN model.

    Args:
        input_shape: Image dimensions (height, width, channels). Default: (64, 64, 3)
        learning_rate: Adam optimizer learning rate. Default: 1e-4

    Returns:
        Compiled Keras model ready for training.
    """
    model = models.Sequential([
        # ── Input ─────────────────────────────────────────────────────────────
        layers.Input(shape=input_shape),

        # ── Rescale pixel values from [0, 255] → [0, 1] ───────────────────────
        layers.Rescaling(1.0 / 255),

        # ── Convolutional Block 1 ─────────────────────────────────────────────
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Convolutional Block 2 ─────────────────────────────────────────────
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Convolutional Block 3 ─────────────────────────────────────────────
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Classifier Head ───────────────────────────────────────────────────
        layers.Flatten(),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),   # Binary: UP (1) vs DOWN (0)
    ], name="StockChartCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    return model


def print_model_summary() -> None:
    """Print model architecture summary."""
    model = build_cnn()
    model.summary()
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {sum(w.numpy().size for w in model.trainable_weights):,}")


if __name__ == "__main__":
    print_model_summary()
