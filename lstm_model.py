"""
LSTM model for Amazon stock price prediction.

Usage:
    python lstm_model.py
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from data_preprocessing import download_data, prepare_lstm_data


# ── Configuration ─────────────────────────────────────────────────────────────
EPOCHS = 50
BATCH_SIZE = 32
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT = 0.2


def build_lstm_model(input_shape):
    """
    Build a stacked LSTM model for time-series prediction.

    Architecture:
        LSTM(128, return_sequences) → Dropout(0.2)
        → LSTM(64, return_sequences) → Dropout(0.2)
        → LSTM(32) → Dropout(0.2)
        → Dense(25) → Dense(1)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(LSTM_UNITS_1, return_sequences=True),
        layers.Dropout(DROPOUT),
        layers.LSTM(LSTM_UNITS_2, return_sequences=True),
        layers.Dropout(DROPOUT),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(DROPOUT),
        layers.Dense(25, activation="relu"),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error",
    )

    return model


def train_lstm():
    """Train the LSTM model and return predictions."""
    # Load data
    df = download_data()
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df)

    # Build model
    model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6,
        ),
    ]

    # Train
    print("[*] Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Predict
    predictions_scaled = model.predict(X_test)

    # Inverse transform to get actual prices
    predictions = scaler.inverse_transform(predictions_scaled)
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

    print(f"[✓] LSTM training complete.")
    return predictions.flatten(), actuals.flatten(), history


if __name__ == "__main__":
    from evaluate import compute_metrics

    preds, actuals, _ = train_lstm()
    metrics = compute_metrics(actuals, preds)
    print(f"\n[*] LSTM Results:")
    print(f"    MAE  : {metrics['mae']:.2f}")
    print(f"    RMSE : {metrics['rmse']:.2f}")
    print(f"    Accuracy: {metrics['accuracy']:.1f}%")
