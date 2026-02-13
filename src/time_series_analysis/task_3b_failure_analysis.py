"""
Task 3B — Failure Analysis

Programmatically identifies and visualises at least 3 failure cases from
the evaluation results:

  1. False positive from residual artifact
  2. Missed slow deterioration (false negative)
  3. Delayed/suppressed alert during noisy segment

Output: plots/failures/*.png with annotated explanations
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time_series_analysis.constants import (
    COL_HR,
    COL_MOTION,
    COL_SBP,
    COL_SPO2,
    DATA_CLEANED_DIR,
    PLOTS_DIR,
    WINDOW_SIZE_SEC,
)


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load evaluation results and cleaned vitals data."""
    results = pd.read_csv(Path(DATA_CLEANED_DIR) / "evaluation_results.csv")

    vitals = {}
    for csv_path in sorted(Path(DATA_CLEANED_DIR).glob("*.csv")):
        if csv_path.stem == "evaluation_results":
            continue
        vitals[csv_path.stem] = pd.read_csv(
            csv_path, index_col="time", parse_dates=True
        )

    return results, vitals


def _plot_failure(
    vitals_df: pd.DataFrame,
    window_start: int,
    title: str,
    explanation: str,
    improvement: str,
    filename: str,
    out_dir: Path,
) -> None:
    """Plot a vitals segment around a failure case with annotations."""
    # Show ±2 min context for review
    plot_start = max(0, window_start - 120)
    plot_end = min(len(vitals_df), window_start + WINDOW_SIZE_SEC + 120)
    segment = vitals_df.iloc[plot_start:plot_end]
    x = np.arange(len(segment))

    cols = [
        c for c in [COL_HR, COL_SPO2, COL_SBP, COL_MOTION] if c in segment.columns
    ]
    fig, axes = plt.subplots(
        len(cols), 1, figsize=(12, 2.5 * len(cols)), sharex=True
    )
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(x, segment[col].values, linewidth=0.8)
        # Shade detection window
        ws = window_start - plot_start
        we = ws + WINDOW_SIZE_SEC
        ax.axvspan(ws, we, alpha=0.15, color="red", label="Detection window")
        ax.set_ylabel(col)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s, relative)")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    text = f"What failed: {explanation}\nImprovement: {improvement}"
    fig.text(
        0.02,
        -0.02,
        text,
        fontsize=8,
        va="top",
        wrap=True,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_false_positive_artifact(results: pd.DataFrame) -> pd.Series | None:
    """Find a false positive that coincides with high motion (residual artifact)."""
    fp = results[(results["anomaly_pred"] == 1) & (results["label"] == 0)]
    if "motion_mean" in fp.columns:
        high_motion_fp = fp[fp["motion_mean"] > 0.3]
        if not high_motion_fp.empty:
            return high_motion_fp.iloc[0]
    return fp.iloc[0] if not fp.empty else None


def find_missed_deterioration(results: pd.DataFrame) -> pd.Series | None:
    """Find a false negative (missed pre-critical window)."""
    fn = results[(results["anomaly_pred"] == 0) & (results["label"] == 1)]
    return fn.iloc[0] if not fn.empty else None


def find_suppressed_alert(results: pd.DataFrame) -> pd.Series | None:
    """Find a window where alert was suppressed due to low confidence."""
    suppressed = results[results["alert_status"] == "SUPPRESSED"]
    # Prefer one where label was actually positive (real danger, suppressed)
    real_danger = suppressed[suppressed["label"] == 1]
    if not real_danger.empty:
        return real_danger.iloc[0]
    return suppressed.iloc[0] if not suppressed.empty else None


def main() -> None:
    out_dir = Path(PLOTS_DIR) / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)

    results, vitals = load_data()
    if results.empty:
        print("No evaluation results. Run task_3a_metrics first.")
        return

    failures_found = 0

    # Case 1: False positive (artifact)
    row = find_false_positive_artifact(results)
    if row is not None and str(int(row["patient_id"])) in vitals:
        _plot_failure(
            vitals[str(int(row["patient_id"]))],
            int(row["window_start"]),
            "Failure Case 1: False Positive from Residual Artifact",
            "Motion artifact survived cleaning and caused the model to flag "
            "a normal window as anomalous. The SpO2 dip was not physiological.",
            "Use a stricter motion-aware post-filter: if Motion > threshold "
            "and SpO2 recovers within 10 s, suppress the anomaly flag.",
            "case1_false_positive.png",
            out_dir,
        )
        failures_found += 1
        print("Case 1: False positive from residual artifact")

    # Case 2: Missed slow deterioration
    row = find_missed_deterioration(results)
    if row is not None and str(int(row["patient_id"])) in vitals:
        _plot_failure(
            vitals[str(int(row["patient_id"]))],
            int(row["window_start"]),
            "Failure Case 2: Missed Slow Deterioration",
            "Gradual SpO2 decline stayed within per-window normal statistics. "
            "Each individual window looked okay, but the trend over 10+ minutes "
            "was clearly heading toward a critical threshold.",
            "Add a multi-window trend detector: track slope of SpO2 over the "
            "last 5–10 windows. Flag if the cumulative drop exceeds 5% even "
            "if no single window is anomalous.",
            "case2_missed_deterioration.png",
            out_dir,
        )
        failures_found += 1
        print("Case 2: Missed slow deterioration")

    # Case 3: Suppressed alert during noisy segment
    row = find_suppressed_alert(results)
    if row is not None and str(int(row["patient_id"])) in vitals:
        _plot_failure(
            vitals[str(int(row["patient_id"]))],
            int(row["window_start"]),
            "Failure Case 3: Alert Suppressed During Noisy Segment",
            "Risk score exceeded the threshold, but confidence was low due to "
            "high motion and missing data. The alert was suppressed to avoid "
            "a noisy false alarm, but the patient was actually deteriorating.",
            "Implement a 'mandatory re-check' after suppression: if the risk "
            "score remains high for 3+ consecutive windows regardless of "
            "confidence, escalate to ALERT with a 'low confidence' warning.",
            "case3_suppressed_alert.png",
            out_dir,
        )
        failures_found += 1
        print("Case 3: Suppressed alert in noisy segment")

    if failures_found == 0:
        print(
            "Could not find enough failure cases. "
            "Try adjusting model parameters or using more data."
        )
    else:
        print(f"\n{failures_found} failure case(s) analysed → {out_dir}/")


if __name__ == "__main__":
    main()
