"""
FBProphet model for Amazon stock price prediction.

Usage:
    python prophet_model.py
"""

import numpy as np
import pandas as pd
from prophet import Prophet

from data_preprocessing import download_data, prepare_prophet_data


def train_prophet():
    """Train an FBProphet model and return predictions."""
    # Load data
    df = download_data()
    train_df, test_df = prepare_prophet_data(df)

    # Build and train Prophet model
    print("[*] Training FBProphet model...")
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )

    # Suppress verbose output
    model.fit(train_df)

    # Predict on test period
    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)

    # Extract predictions for test period
    predictions = forecast["yhat"].iloc[-len(test_df):].values
    actuals = test_df["y"].values

    print(f"[✓] FBProphet training complete.")
    return predictions, actuals, model, forecast


if __name__ == "__main__":
    from evaluate import compute_metrics

    preds, actuals, _, _ = train_prophet()
    metrics = compute_metrics(actuals, preds)
    print(f"\n[*] FBProphet Results:")
    print(f"    MAE  : {metrics['mae']:.2f}")
    print(f"    RMSE : {metrics['rmse']:.2f}")
    print(f"    Accuracy: {metrics['accuracy']:.1f}%")
