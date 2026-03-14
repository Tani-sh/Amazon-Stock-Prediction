"""
Visualisation utilities for stock prediction results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("darkgrid")
PLOT_DIR = "./plots"


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_predictions(actual, predicted, title="Stock Price Prediction", filename="prediction.png"):
    """Plot actual vs. predicted prices."""
    ensure_plot_dir()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual, label="Actual", color="#2196F3", linewidth=1.5)
    ax.plot(predicted, label="Predicted", color="#FF5722", linewidth=1.5, alpha=0.8)
    ax.fill_between(range(len(actual)), actual, predicted, alpha=0.1, color="#FF5722")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(
    lstm_actual, lstm_pred,
    prophet_actual, prophet_pred,
    lstm_metrics, prophet_metrics,
):
    """Generate comparison plots for LSTM vs. FBProphet."""
    ensure_plot_dir()

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # 1. LSTM predictions
    ax1 = axes[0, 0]
    ax1.plot(lstm_actual, label="Actual", color="#2196F3", linewidth=1.2)
    ax1.plot(lstm_pred, label="LSTM Predicted", color="#FF5722", linewidth=1.2, alpha=0.8)
    ax1.set_title(f"LSTM — MAE: {lstm_metrics['mae']:.2f}, RMSE: {lstm_metrics['rmse']:.2f}")
    ax1.legend()
    ax1.set_ylabel("Price (USD)")

    # 2. Prophet predictions
    ax2 = axes[0, 1]
    ax2.plot(prophet_actual, label="Actual", color="#2196F3", linewidth=1.2)
    ax2.plot(prophet_pred, label="Prophet Predicted", color="#4CAF50", linewidth=1.2, alpha=0.8)
    ax2.set_title(f"FBProphet — MAE: {prophet_metrics['mae']:.2f}, RMSE: {prophet_metrics['rmse']:.2f}")
    ax2.legend()

    # 3. Error distribution
    ax3 = axes[1, 0]
    lstm_errors = lstm_actual - lstm_pred
    prophet_errors = prophet_actual - prophet_pred
    ax3.hist(lstm_errors, bins=50, alpha=0.6, label="LSTM", color="#FF5722")
    ax3.hist(prophet_errors, bins=50, alpha=0.6, label="FBProphet", color="#4CAF50")
    ax3.set_title("Prediction Error Distribution")
    ax3.set_xlabel("Error (USD)")
    ax3.set_ylabel("Frequency")
    ax3.legend()

    # 4. Metrics comparison bar chart
    ax4 = axes[1, 1]
    labels = ["MAE", "RMSE"]
    lstm_vals = [lstm_metrics["mae"], lstm_metrics["rmse"]]
    prophet_vals = [prophet_metrics["mae"], prophet_metrics["rmse"]]
    x = np.arange(len(labels))
    width = 0.3
    ax4.bar(x - width / 2, lstm_vals, width, label="LSTM", color="#FF5722", alpha=0.8)
    ax4.bar(x + width / 2, prophet_vals, width, label="FBProphet", color="#4CAF50", alpha=0.8)
    ax4.set_title("Model Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()

    plt.suptitle("Amazon Stock Prediction — LSTM vs. FBProphet", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Comparison plot saved to {PLOT_DIR}/comparison.png")


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n = 200
    actual = np.cumsum(np.random.randn(n)) + 100
    pred_lstm = actual + np.random.randn(n) * 1.5
    pred_prophet = actual + np.random.randn(n) * 2.0

    plot_predictions(actual, pred_lstm, "LSTM Demo", "lstm_demo.png")
    plot_predictions(actual, pred_prophet, "Prophet Demo", "prophet_demo.png")
    print("[✓] Demo plots saved.")
