"""
Data loading, feature engineering, and PCA for stock prediction.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# ── Configuration ─────────────────────────────────────────────────────────────
TICKER = "AMZN"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
SEQUENCE_LENGTH = 60  # days of lookback
PCA_COMPONENTS = 5
TEST_SPLIT = 0.2


def download_data(ticker=TICKER, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """Download historical stock data from Yahoo Finance."""
    print(f"[*] Downloading {ticker} data ({start} → {end})...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.dropna()
    print(f"[*] Downloaded {len(df)} trading days.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features."""
    data = df.copy()

    # Moving averages
    data["MA_7"] = data["Close"].rolling(window=7).mean()
    data["MA_21"] = data["Close"].rolling(window=21).mean()
    data["MA_50"] = data["Close"].rolling(window=50).mean()

    # Exponential moving averages
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    data["MACD"] = data["EMA_12"] - data["EMA_26"]

    # Bollinger Bands
    bb_window = 20
    data["BB_Mid"] = data["Close"].rolling(window=bb_window).mean()
    bb_std = data["Close"].rolling(window=bb_window).std()
    data["BB_Upper"] = data["BB_Mid"] + 2 * bb_std
    data["BB_Lower"] = data["BB_Mid"] - 2 * bb_std

    # RSI (14-day)
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    data["RSI"] = 100 - (100 / (1 + rs))

    # Daily returns and volatility
    data["Daily_Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Daily_Return"].rolling(window=21).std()

    # Volume change
    data["Volume_Change"] = data["Volume"].pct_change()

    data = data.dropna()
    return data


def apply_pca(features: np.ndarray, n_components=PCA_COMPONENTS):
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    variance = sum(pca.explained_variance_ratio_) * 100
    print(f"[*] PCA: {features.shape[1]} features → {n_components} components "
          f"({variance:.1f}% variance retained)")
    return reduced, pca


def prepare_lstm_data(df: pd.DataFrame, target_col="Close", seq_length=SEQUENCE_LENGTH):
    """
    Prepare sequenced data for LSTM training.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    data = df[[target_col]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i - seq_length : i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"[*] LSTM data — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


def prepare_prophet_data(df: pd.DataFrame, target_col="Close"):
    """
    Prepare data for FBProphet (requires 'ds' and 'y' columns).

    Returns:
        train_df, test_df
    """
    prophet_df = df[[target_col]].copy()
    prophet_df.reset_index(inplace=True)
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    split = int(len(prophet_df) * (1 - TEST_SPLIT))
    train_df = prophet_df[:split]
    test_df = prophet_df[split:]

    print(f"[*] Prophet data — Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


if __name__ == "__main__":
    df = download_data()
    df = engineer_features(df)
    print(f"[*] Features: {list(df.columns)}")
    print(f"[*] Shape: {df.shape}")
