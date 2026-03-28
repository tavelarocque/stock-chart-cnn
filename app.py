"""
app.py
------
Streamlit web app for the Stock Chart Pattern CNN.

Type in any stock ticker and get a live prediction with candlestick chart.

Usage:
    streamlit run app.py
"""

import os
import warnings
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import mplfinance as mpf
from PIL import Image
from tensorflow import keras

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Chart CNN",
    page_icon="📈",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📈 Stock Chart Pattern Classifier")
st.markdown(
    "A **Convolutional Neural Network** trained on 30,000+ candlestick chart images "
    "to predict whether a stock will move **UP** or **DOWN** over the next 5 trading days."
)
st.divider()

# ── Model Loading (cached so it only loads once) ──────────────────────────────

MODEL_PATH = "results/models/best_model.keras"
IMAGE_SIZE = (64, 64)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.error(
        "⚠️ Trained model not found. Please run `python3 src/train.py` first.",
        icon="🚨",
    )
    st.stop()

# ── Ticker Input ──────────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.text_input(
        "Enter a stock ticker:",
        value="AAPL",
        placeholder="e.g. AAPL, NVDA, TSLA, MSFT",
        help="Any ticker listed on Yahoo Finance (US stocks work best)"
    ).upper().strip()
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

# ── Prediction Logic ──────────────────────────────────────────────────────────

def fetch_and_predict(ticker_symbol):
    """Download latest data, generate chart, run model, return results."""
    with st.spinner(f"Fetching latest data for {ticker_symbol}..."):
        df = yf.download(ticker_symbol, period="90d", progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna().tail(30)

    if len(df) < 30:
        st.error(f"Not enough data for **{ticker_symbol}** — only {len(df)} days available. Try a different ticker.")
        return

    # ── Generate chart ─────────────────────────────────────────────────────────
    with st.spinner("Generating candlestick chart..."):
        buf = io.BytesIO()
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            rc={"axes.labelsize": 0, "xtick.labelsize": 0, "ytick.labelsize": 0},
        )
        mpf.plot(
            df, type="candle", style=style, volume=True,
            savefig=dict(fname=buf, dpi=150, bbox_inches="tight"),
            figsize=(8, 5), tight_layout=True,
        )
        buf.seek(0)
        chart_img = Image.open(buf).convert("RGB")

    # ── Run model ─────────────────────────────────────────────────────────────
    with st.spinner("Running CNN prediction..."):
        img_small = chart_img.resize(IMAGE_SIZE)
        img_array = np.array(img_small)[np.newaxis, ...]
        prob_up = float(model.predict(img_array, verbose=0)[0][0])
        prob_down = 1 - prob_up
        direction = "UP" if prob_up >= 0.5 else "DOWN"
        confidence = prob_up if prob_up >= 0.5 else prob_down

    # ── Display results ────────────────────────────────────────────────────────
    st.subheader(f"Results for **{ticker_symbol}**")

    # Prediction card
    if direction == "UP":
        st.success(f"### 📈 Prediction: UP  |  Confidence: {confidence*100:.1f}%")
    else:
        st.error(f"### 📉 Prediction: DOWN  |  Confidence: {confidence*100:.1f}%")

    # Probability bar
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("P(UP)", f"{prob_up*100:.1f}%", delta=None)
        st.progress(prob_up)
    with col_b:
        st.metric("P(DOWN)", f"{prob_down*100:.1f}%", delta=None)
        st.progress(prob_down)

    st.divider()

    # Candlestick chart (full resolution)
    st.subheader("📊 Last 30 Days — Candlestick Chart")
    st.image(chart_img, use_column_width=True, caption=f"{ticker_symbol} — 30-day candlestick chart used for prediction")

    # Model input preview
    with st.expander("🔬 What the CNN actually sees (64×64 input)"):
        st.image(img_small.resize((256, 256), Image.NEAREST),
                 caption="Chart resized to 64×64 pixels — the CNN's actual input")

    # Disclaimer
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for **educational purposes only**. "
        "Do not use it for actual trading or investment decisions. "
        "Past chart patterns do not guarantee future returns."
    )


# ── Run on button click or Enter ──────────────────────────────────────────────

if predict_btn and ticker:
    fetch_and_predict(ticker)
elif not predict_btn:
    # Show example info on first load
    st.markdown("### How it works")
    st.markdown(
        """
        1. **Data** — Downloads the last 30 days of OHLCV price data from Yahoo Finance
        2. **Chart** — Converts it into a candlestick chart image (the same way the model was trained)
        3. **CNN** — Feeds the image through a 3-block Convolutional Neural Network
        4. **Prediction** — Outputs the probability the stock will be UP or DOWN in 5 days
        """
    )

    st.markdown("### Model performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", "56.4%", help="Trained on 30,000+ chart images from 35 stocks (2018-2024)")
    col2.metric("Test AUC", "0.58", help="Area Under the ROC Curve — above 0.5 means better than random")
    col3.metric("Training Data", "30K charts", help="Generated from 35 major US stocks over 6 years")

    st.info("💡 Enter a ticker above and click **Predict** to get started!", icon="👆")
