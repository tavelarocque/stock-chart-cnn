"""
data_collector.py
-----------------
Downloads historical stock price data from Yahoo Finance and generates
labeled candlestick chart images for training the CNN.

Label logic:
  - If the closing price 5 days after a window end is > 2% higher → "UP" (1)
  - If the closing price 5 days after a window end is > 2% lower  → "DOWN" (0)
  - Otherwise → skip (too ambiguous)
"""

import os
import warnings
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

# S&P 500 stocks (a diverse selection across sectors)
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",   # Tech
    "JPM", "BAC", "GS", "WFC", "C",             # Finance
    "JNJ", "PFE", "UNH", "MRK", "ABBV",         # Healthcare
    "XOM", "CVX", "COP", "SLB", "EOG",          # Energy
    "WMT", "HD", "MCD", "NKE", "SBUX",          # Consumer
    "CAT", "DE", "BA", "GE", "MMM",             # Industrials
    "TSLA", "NVDA", "AMD", "INTC", "QCOM",      # Semiconductors
]

WINDOW_SIZE = 30          # Days of price history per chart image
FUTURE_DAYS = 5           # Days ahead to predict
THRESHOLD = 0.02          # 2% movement required to label UP or DOWN
IMAGE_SIZE = (64, 64)     # Chart image dimensions (pixels)
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"


def download_data(tickers, start, end):
    """Download OHLCV data for a list of tickers."""
    data = {}
    print(f"\n📥 Downloading data for {len(tickers)} tickers...")
    for ticker in tqdm(tickers, desc="Fetching"):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            # Flatten MultiIndex columns (newer yfinance versions return these)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Remove duplicate columns if any
            df = df.loc[:, ~df.columns.duplicated()]
            if len(df) > WINDOW_SIZE + FUTURE_DAYS + 10:
                data[ticker] = df
        except Exception as e:
            print(f"  ⚠ Skipped {ticker}: {e}")
    print(f"  ✓ Successfully downloaded {len(data)} tickers")
    return data


def compute_label(df: pd.DataFrame, end_idx: int):
    """
    Return 1 (UP), 0 (DOWN), or None (ambiguous) for a given window end index.
    """
    if end_idx + FUTURE_DAYS >= len(df):
        return None
    current_close = df["Close"].iloc[end_idx]
    future_close = df["Close"].iloc[end_idx + FUTURE_DAYS]
    pct_change = (future_close - current_close) / current_close
    if pct_change > THRESHOLD:
        return 1   # UP
    elif pct_change < -THRESHOLD:
        return 0   # DOWN
    return None    # Ambiguous — skip


def generate_chart_image(df_window: pd.DataFrame, save_path: str) -> bool:
    """
    Generate and save a candlestick chart image from a price window.
    Returns True on success, False on failure.
    """
    try:
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            rc={"axes.labelsize": 0, "xtick.labelsize": 0, "ytick.labelsize": 0},
        )
        mpf.plot(
            df_window,
            type="candle",
            style=style,
            volume=True,
            savefig=dict(fname=save_path, dpi=72, bbox_inches="tight"),
            figsize=(2, 2),
            tight_layout=True,
        )
        return True
    except Exception:
        return False


def generate_dataset(output_dir: str = "results/charts") -> pd.DataFrame:
    """
    Main function: downloads data, generates chart images, and saves a
    metadata CSV with image paths and labels.

    Returns a DataFrame with columns: [image_path, label, ticker, date]
    """
    os.makedirs(os.path.join(output_dir, "UP"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "DOWN"), exist_ok=True)

    all_data = download_data(TICKERS, START_DATE, END_DATE)

    records = []
    total_generated = 0
    total_skipped = 0

    print("\n🖼  Generating chart images...")
    for ticker, df in tqdm(all_data.items(), desc="Processing tickers"):
        # Drop missing values and ensure correct column format
        df = df.dropna()

        for i in range(WINDOW_SIZE, len(df) - FUTURE_DAYS):
            label = compute_label(df, i)
            if label is None:
                total_skipped += 1
                continue

            df_window = df.iloc[i - WINDOW_SIZE : i].copy()
            label_name = "UP" if label == 1 else "DOWN"
            date_str = str(df.index[i].date()).replace("-", "")
            filename = f"{ticker}_{date_str}.png"
            save_path = os.path.join(output_dir, label_name, filename)

            if generate_chart_image(df_window, save_path):
                records.append({
                    "image_path": save_path,
                    "label": label,
                    "label_name": label_name,
                    "ticker": ticker,
                    "date": str(df.index[i].date()),
                })
                total_generated += 1

    metadata = pd.DataFrame(records)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata.to_csv(metadata_path, index=False)

    print(f"\n✅ Dataset generation complete!")
    print(f"   Charts generated : {total_generated:,}")
    print(f"   Ambiguous skipped: {total_skipped:,}")
    if not metadata.empty:
        print(f"   UP  samples      : {(metadata['label'] == 1).sum():,}")
        print(f"   DOWN samples     : {(metadata['label'] == 0).sum():,}")
    else:
        print("   ⚠ No charts generated — check your internet connection and try again.")
    print(f"   Metadata saved to: {metadata_path}")

    return metadata


if __name__ == "__main__":
    metadata = generate_dataset(output_dir="results/charts")
    print(metadata.head())
