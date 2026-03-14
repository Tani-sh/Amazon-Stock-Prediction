"""
Evaluation metrics and benchmark comparison for stock prediction models.

Usage:
    python evaluate.py
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, and directional accuracy.

    Directional accuracy = percentage of days where the predicted
    direction of change matches the actual direction.
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Directional accuracy
    actual_direction = np.diff(actual) > 0
    pred_direction = np.diff(predicted) > 0
    accuracy = np.mean(actual_direction == pred_direction) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
    }


def run_benchmark():
    """Run full benchmark comparing LSTM and FBProphet."""
    from lstm_model import train_lstm
    from prophet_model import train_prophet

    print("=" * 60)
    print("  AMAZON STOCK PREDICTION BENCHMARK")
    print("=" * 60)

    # Train LSTM
    print("\n" + "─" * 50)
    print("  Training LSTM...")
    print("─" * 50)
    lstm_preds, lstm_actuals, _ = train_lstm()
    lstm_metrics = compute_metrics(lstm_actuals, lstm_preds)

    # Train Prophet
    print("\n" + "─" * 50)
    print("  Training FBProphet...")
    print("─" * 50)
    prophet_preds, prophet_actuals, _, _ = train_prophet()
    prophet_metrics = compute_metrics(prophet_actuals, prophet_preds)

    # Results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  {'Metric':<20} {'LSTM':>10} {'FBProphet':>12}")
    print(f"  {'─' * 42}")
    print(f"  {'MAE':<20} {lstm_metrics['mae']:>10.2f} {prophet_metrics['mae']:>12.2f}")
    print(f"  {'RMSE':<20} {lstm_metrics['rmse']:>10.2f} {prophet_metrics['rmse']:>12.2f}")
    print(f"  {'Accuracy (%)':<20} {lstm_metrics['accuracy']:>10.1f} {prophet_metrics['accuracy']:>12.1f}")
    print(f"\n{'=' * 60}")

    # Generate comparison plots
    try:
        from visualize import plot_comparison
        plot_comparison(
            lstm_actuals, lstm_preds,
            prophet_actuals, prophet_preds,
            lstm_metrics, prophet_metrics,
        )
        print("[✓] Plots saved to ./plots/")
    except Exception as e:
        print(f"[!] Could not generate plots: {e}")


if __name__ == "__main__":
    run_benchmark()
