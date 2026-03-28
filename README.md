# 📈 Stock Chart Pattern Recognition with CNN

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

A computer vision project that trains a **Convolutional Neural Network (CNN)** to classify stock candlestick chart images and predict whether a stock will move **UP** or **DOWN** over the next 5 trading days.

> ⚠️ **Disclaimer:** This project is for **educational purposes only**. Do not use it for actual trading or investment decisions.

---

## 🎯 Project Overview

This project demonstrates an end-to-end machine learning pipeline that bridges **computer vision** and **financial data**:

1. **Data Collection** — Downloads historical OHLCV (Open, High, Low, Close, Volume) data for 37 major stocks using `yfinance`
2. **Chart Generation** — Converts raw price data into 64×64 candlestick chart images using `mplfinance`
3. **Labeling** — Automatically labels each chart based on the stock's price movement 5 days later (UP > +2%, DOWN < −2%)
4. **Model Training** — Trains a 3-block CNN with batch normalization, dropout, and data augmentation
5. **Evaluation** — Generates classification reports, confusion matrices, and ROC curves
6. **Live Prediction** — Predicts the direction of any stock ticker using real-time Yahoo Finance data

---

## 🏗️ Project Structure

```
stock-chart-cnn/
├── src/
│   ├── data_collector.py   # Download data & generate chart images
│   ├── model.py            # CNN architecture definition
│   ├── train.py            # Training pipeline with callbacks
│   └── predict.py          # Evaluation & live ticker prediction
├── results/
│   ├── charts/             # Generated candlestick images (UP/ and DOWN/)
│   │   └── metadata.csv    # Image paths, labels, tickers, dates
│   └── models/             # Saved model weights & training logs
├── notebooks/
│   └── exploration.ipynb   # EDA and visualization notebook
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

```
Input: 64×64 RGB Candlestick Chart Image
│
├── Rescaling (pixel values → [0, 1])
│
├── Conv Block 1: Conv2D(32, 3×3) → BatchNorm → MaxPool → Dropout(0.25)
├── Conv Block 2: Conv2D(64, 3×3) → BatchNorm → MaxPool → Dropout(0.25)
├── Conv Block 3: Conv2D(128, 3×3) → BatchNorm → MaxPool → Dropout(0.25)
│
├── Flatten
├── Dense(256, ReLU) + L2 Regularization → Dropout(0.5)
└── Dense(1, Sigmoid) → P(UP)
```

**Training details:**
- Optimizer: Adam (lr = 1e-4, with ReduceLROnPlateau)
- Loss: Binary Cross-Entropy
- Data augmentation: random flips, rotation, zoom, contrast
- Class weighting to handle imbalanced UP/DOWN samples
- Early stopping (patience = 10 epochs on val AUC)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/stock-chart-cnn.git
cd stock-chart-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

This downloads stock data from 2018–2024 and generates ~50,000+ labeled chart images. Takes 20–40 minutes depending on your machine.

```bash
python src/data_collector.py
```

### 4. Train the model

```bash
python src/train.py
```

Training runs for up to 50 epochs with early stopping. The best model is saved to `results/models/best_model.keras`.

### 5. Evaluate on the test set

```bash
python src/predict.py
```

This generates:
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- Full classification report in the terminal

### 6. Predict a live stock

```bash
python src/predict.py --ticker MSFT
```

### 7. Launch the web app 🌐

```bash
streamlit run app.py
```

This opens an interactive browser app where you can type any ticker and instantly see the candlestick chart + CNN prediction with confidence scores.

```
==========================================
  Ticker     : MSFT
  Prediction : 📈 UP
  Confidence : 68.3%
  P(UP)      : 68.3%
  P(DOWN)    : 31.7%
==========================================
```

---

## 📊 Results

| Metric         | Value  |
|----------------|--------|
| Test Accuracy  | ~62%   |
| Test AUC       | ~0.66  |
| F1 Score (UP)  | ~0.63  |
| F1 Score (DOWN)| ~0.61  |

> Note: Beating 50% (random) on stock prediction is genuinely non-trivial. Markets are efficient and noisy — ~62% accuracy represents a meaningful signal from visual chart patterns alone.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `yfinance` | Download historical stock OHLCV data |
| `mplfinance` | Generate candlestick chart images |
| `TensorFlow / Keras` | CNN model building and training |
| `scikit-learn` | Train/test split, class weighting, evaluation metrics |
| `NumPy / Pandas` | Data manipulation |
| `Matplotlib / Seaborn` | Visualization |
| `tqdm` | Progress bars |

---

## 💡 Key Concepts Demonstrated

- **Computer Vision**: Convolutional Neural Networks for image classification
- **Financial Data Engineering**: Converting time-series OHLCV data into visual representations
- **ML Best Practices**: Train/val/test splits, data augmentation, early stopping, learning rate scheduling
- **Class Imbalance Handling**: Computed class weights to prevent model bias
- **Model Evaluation**: Beyond accuracy — AUC, ROC curves, confusion matrices, precision/recall
- **End-to-End Pipeline**: From raw data download to live prediction in 4 commands

---

## 📚 Background

Traditional technical analysis uses candlestick chart patterns (like "Head & Shoulders", "Doji", "Engulfing") to predict price movements. This project asks: **can a CNN learn these patterns automatically from raw chart images**, without any human-defined rules?

This approach is inspired by research in financial deep learning, including papers like:
- *"Algorithmic Trading Using Machine Learning"* (various)
- *"Convolutional Neural Networks for Financial Time Series"*

---

## 🔮 Future Improvements

- Add transfer learning (ResNet50, EfficientNet) for potentially higher accuracy
- Include technical indicators (RSI, MACD, Bollinger Bands) overlaid on charts
- Expand to multi-class prediction (STRONG_UP / UP / NEUTRAL / DOWN / STRONG_DOWN)
- Add a web interface with Streamlit or Gradio for easy demos
- Backtest predictions against historical returns

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome! If you find a bug or have a feature idea, please open an issue first.

---

*Built with Python 🐍 | TensorFlow 🧠 | Real Market Data 📈*
