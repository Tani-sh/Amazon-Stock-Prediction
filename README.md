# 📈 Amazon Stock Prediction

Forecasting Amazon (AMZN) stock closing prices using two approaches — a stacked **LSTM** network and **FBProphet** — and comparing their performance. Also applies PCA for dimensionality reduction on engineered features.

## 🔍 Approach

**LSTM**: A 3-layer stacked LSTM (128 → 64 → 32 units) trained on 60-day look-back windows of normalised closing prices. Uses early stopping and learning rate scheduling.

**FBProphet**: Facebook's time-series model with daily, weekly, and yearly seasonality components. Simpler to set up but less flexible for capturing non-linear patterns.

Both models are evaluated on the same held-out test set for a fair comparison.

## 📊 Results

| Model | MAE | RMSE |
|-------|-----|------|
| LSTM | 1.2 | 1.5 |
| FBProphet | 1.4 | 1.7 |

LSTM edges out Prophet on both metrics, likely due to its ability to learn sequential dependencies in the price data.

## 📁 Project structure

```
├── data_preprocessing.py   # Download data (yfinance), feature engineering, PCA, scaling
├── lstm_model.py           # LSTM architecture, training, prediction
├── prophet_model.py        # FBProphet training and forecasting
├── evaluate.py             # MAE, RMSE, accuracy computation
├── visualize.py            # Prediction vs. actual plots
├── requirements.txt
└── .gitignore
```

## 🚀 Usage

```bash
pip install -r requirements.txt

# Run LSTM model
python lstm_model.py

# Run Prophet model
python prophet_model.py

# Generate comparison plots
python visualize.py
```

📥 Data is pulled automatically from Yahoo Finance via `yfinance`.

## 🔧 Dependencies

`tensorflow`, `prophet`, `yfinance`, `scikit-learn` (PCA, scaling), `numpy`, `pandas`, `matplotlib`
