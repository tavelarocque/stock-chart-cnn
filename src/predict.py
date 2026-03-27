"""
predict.py
----------
Loads the trained CNN model and evaluates it on the held-out test set,
generating a full classification report, confusion matrix, and ROC curve.

Also supports predicting on a single new ticker (live data from Yahoo Finance).

Usage:
    # Full evaluation report on test set:
    python src/predict.py

    # Predict on a specific stock (live data):
    python src/predict.py --ticker AAPL
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc as sklearn_auc,
    ConfusionMatrixDisplay,
)
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

MODELS_DIR = "results/models"
RESULTS_DIR = "results"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32


# ── Evaluation on Test Set ────────────────────────────────────────────────────

def evaluate_model(models_dir: str = MODELS_DIR, results_dir: str = RESULTS_DIR) -> None:
    """Load model and test data, produce full evaluation report."""
    model_path = os.path.join(models_dir, "best_model.keras")
    X_test_path = os.path.join(models_dir, "X_test.npy")
    y_test_path = os.path.join(models_dir, "y_test.npy")

    if not all(os.path.exists(p) for p in [model_path, X_test_path, y_test_path]):
        print("❌ Model or test data not found. Run 'python src/train.py' first.")
        sys.exit(1)

    print("📦 Loading model...")
    model = keras.models.load_model(model_path)

    print("📂 Loading test data...")
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    print(f"   Test samples: {len(X_test):,}")

    # ── Predictions ───────────────────────────────────────────────────────────
    test_ds = (
        tf.data.Dataset.from_tensor_slices(X_test)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    y_probs = model.predict(test_ds, verbose=0).flatten()
    y_pred = (y_probs >= 0.5).astype(int)

    # ── Classification Report ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Classification Report")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    overall_accuracy = (y_pred == y_test).mean()
    print(f"  Overall Accuracy: {overall_accuracy * 100:.2f}%")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm, results_dir)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    _plot_roc_curve(y_test, y_probs, results_dir)

    print(f"\n✅ Evaluation plots saved to: {results_dir}/")


def _plot_confusion_matrix(cm: np.ndarray, save_dir: str) -> None:
    """Save a styled confusion matrix plot."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["DOWN (0)", "UP (1)"],
        yticklabels=["DOWN (0)", "UP (1)"],
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Confusion Matrix — Stock Chart CNN", fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")

    # Annotate with percentages
    total = cm.sum()
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j + 0.5, i + 0.65, f"({val/total*100:.1f}%)",
                ha="center", va="center", fontsize=9, color="gray")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Confusion matrix saved to: {out_path}")


def _plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, save_dir: str) -> None:
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = sklearn_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curve — Stock Chart CNN", fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ROC curve saved to: {out_path}")


# ── Live Single-Stock Prediction ──────────────────────────────────────────────

def predict_ticker(ticker: str, models_dir: str = MODELS_DIR) -> None:
    """
    Download the latest 30 days of data for a ticker and predict its direction.
    """
    import yfinance as yf
    import mplfinance as mpf
    from io import BytesIO
    from PIL import Image

    model_path = os.path.join(models_dir, "best_model.keras")
    if not os.path.exists(model_path):
        print("❌ Trained model not found. Run 'python src/train.py' first.")
        sys.exit(1)

    print(f"\n📡 Fetching latest data for {ticker}...")
    df = yf.download(ticker, period="60d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().tail(30)

    if len(df) < 30:
        print(f"❌ Not enough data for {ticker}. Only {len(df)} days available.")
        sys.exit(1)

    # Generate chart image in memory
    buf = BytesIO()
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        rc={"axes.labelsize": 0, "xtick.labelsize": 0, "ytick.labelsize": 0},
    )
    mpf.plot(df, type="candle", style=style, volume=True,
             savefig=dict(fname=buf, dpi=72, bbox_inches="tight"),
             figsize=(2, 2), tight_layout=True)
    buf.seek(0)
    img = Image.open(buf).resize(IMAGE_SIZE).convert("RGB")
    img_array = np.array(img)[np.newaxis, ...]   # Add batch dimension

    # Load model and predict
    print("🧠 Loading model and predicting...")
    model = keras.models.load_model(model_path)
    prob_up = float(model.predict(img_array, verbose=0)[0][0])
    direction = "📈 UP" if prob_up >= 0.5 else "📉 DOWN"
    confidence = prob_up if prob_up >= 0.5 else (1 - prob_up)

    print("\n" + "=" * 45)
    print(f"  Ticker     : {ticker}")
    print(f"  Prediction : {direction}")
    print(f"  Confidence : {confidence * 100:.1f}%")
    print(f"  P(UP)      : {prob_up * 100:.1f}%")
    print(f"  P(DOWN)    : {(1 - prob_up) * 100:.1f}%")
    print("=" * 45)
    print("\n  ⚠️  Disclaimer: This is for educational purposes only.")
    print("     Do not use for actual trading or investment decisions.")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the trained CNN or predict a single stock ticker."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Stock ticker for live prediction (e.g., AAPL). "
             "If not provided, runs full evaluation on the test set.",
    )
    args = parser.parse_args()

    if args.ticker:
        predict_ticker(args.ticker.upper())
    else:
        evaluate_model()
