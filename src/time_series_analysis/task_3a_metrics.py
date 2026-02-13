"""
Task 3A — Metrics Definition & Evaluation

Runs the full pipeline (load → predict → risk-score → alert)
on the test split and reports:
  - Precision, recall, F1
  - False alert rate  (FP / total alerts)
  - Alert latency     (seconds between first alert and actual breach)

Also prints a discussion of which errors are acceptable in an ambulance.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from time_series_analysis.constants import (
    COL_MOTION,
    DATA_CLEANED_DIR,
    EARLY_WARNING_HORIZON_SEC,
    WINDOW_SIZE_SEC,
)
from time_series_analysis.task_2a_anomaly_detection import build_dataset, load_model
from time_series_analysis.task_2b_risk_scoring import compute_risk


def evaluate_pipeline() -> pd.DataFrame:
    """Run the full pipeline and return per-window results."""
    clean_dir = Path(DATA_CLEANED_DIR)

    clf, _ = load_model()

    csv_files = sorted(clean_dir.glob("*.csv"))

    # Rebuild training feature matrix
    dataset, feature_cols = build_dataset(csv_files)
    if dataset.empty:
        return pd.DataFrame()

    # Use only the test split (last 30 %)
    split_idx = int(len(dataset) * 0.7)
    test_df = dataset.iloc[split_idx:].copy()
    if test_df.empty:
        return pd.DataFrame()

    # Model predictions
    X_test = test_df[feature_cols].values
    test_df["anomaly_pred"] = clf.predict(X_test)

    # Risk scoring for each window
    risk_scores, confidences, alert_statuses, motion_means = [], [], [], []
    for _, row in test_df.iterrows():
        patient_id = row["patient_id"]
        csv_path = clean_dir / f"{patient_id}.csv"
        df = pd.read_csv(csv_path, index_col="time", parse_dates=True)
        start = int(row["window_start"])
        end = start + WINDOW_SIZE_SEC
        if end > len(df):
            risk_scores.append(0.0)
            confidences.append(0.0)
            alert_statuses.append("NORMAL")
            motion_means.append(0.0)
            continue
        window = df.iloc[start:end]
        result = compute_risk(window)
        risk_scores.append(result["risk_score"])
        confidences.append(result["confidence"])
        alert_statuses.append(result["alert_status"])
        mot = window[COL_MOTION].mean() if COL_MOTION in window.columns else 0.0
        motion_means.append(mot)

    test_df["risk_score"] = risk_scores
    test_df["confidence"] = confidences
    test_df["alert_status"] = alert_statuses
    test_df["motion_mean"] = motion_means

    return test_df.reset_index(drop=True)


def compute_metrics(results: pd.DataFrame) -> None:
    """Compute and print all evaluation metrics."""
    y_true = results["label"]
    y_pred = results["anomaly_pred"]

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Normal", "Pre-critical"]
        )
    )

    # False alert rate
    total_alerts = y_pred.sum()
    false_positives = ((y_pred == 1) & (y_true == 0)).sum()
    false_alert_rate = false_positives / total_alerts if total_alerts > 0 else 0
    print(
        f"False Alert Rate: {false_alert_rate:.3f}  ({false_positives}/{total_alerts})"
    )

    # Latency approx: EARLY_WARNING_HORIZON_SEC - window_offset (discretized)
    tp_mask = (y_pred == 1) & (y_true == 1)
    if tp_mask.any():
        # window_start in seconds; breach within EARLY_WARNING_HORIZON_SEC
        latencies = EARLY_WARNING_HORIZON_SEC - (
            results.loc[tp_mask, "window_start"] % EARLY_WARNING_HORIZON_SEC
        )
        mean_latency = latencies.mean()
        print(f"Mean Alert Latency: {mean_latency:.0f} s (before threshold breach)")
    else:
        print("Mean Alert Latency: N/A (no true positives)")

    # Risk score distribution by alert status
    print("\nRisk Score by Alert Status")
    for status in ["ALERT", "SUPPRESSED", "NORMAL"]:
        subset = results[results["alert_status"] == status]
        if len(subset) > 0:
            print(
                f"  {status:>10}: n={len(subset):>4}  "
                f"risk={subset['risk_score'].mean():.1f}±{subset['risk_score'].std():.1f}  "
                f"conf={subset['confidence'].mean():.2f}"
            )


def main() -> None:
    results = evaluate_pipeline()
    if results.empty:
        print("No results. Ensure tasks 1A and 2A have been run.")
        return

    compute_metrics(results)

    out = Path(DATA_CLEANED_DIR) / "evaluation_results.csv"
    results.to_csv(out, index=False)
    print(f"\nEvaluation results saved to {out}")


if __name__ == "__main__":
    main()
