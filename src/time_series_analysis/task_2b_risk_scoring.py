"""
Task 2B — Risk Scoring Logic

Combines multiple vitals, trends, and data-quality confidence into a single
triage risk score (0–100) with alert/suppress logic.

Usage: import and call `compute_risk()`, or run standalone on cleaned data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from time_series_analysis.constants import (
    COL_DBP,
    COL_HR,
    COL_MOTION,
    COL_SBP,
    COL_SPO2,
    CONFIDENCE_MIN,
    DATA_CLEANED_DIR,
    DBP_HIGH,
    DBP_LOW,
    HR_HIGH,
    HR_LOW,
    RISK_ALERT_THRESHOLD,
    RISK_WEIGHTS,
    SBP_HIGH,
    SBP_LOW,
    SPO2_LOW,
    TREND_BONUS_MAX,
    VITAL_COLS,
    WINDOW_SIZE_SEC,
    WINDOW_STRIDE_SEC,
)

# Per-vital severity (0–25)

# Normal centre & half-range per vital
_VITAL_RANGES = {
    COL_HR: {"centre": (HR_LOW + HR_HIGH) / 2, "half": (HR_HIGH - HR_LOW) / 2},
    COL_SPO2: {"centre": 97, "half": 97 - SPO2_LOW},  # 97 % is "perfect"
    COL_SBP: {"centre": (SBP_LOW + SBP_HIGH) / 2, "half": (SBP_HIGH - SBP_LOW) / 2},
    COL_DBP: {"centre": (DBP_LOW + DBP_HIGH) / 2, "half": (DBP_HIGH - DBP_LOW) / 2},
}


def _vital_severity(value: float, col: str) -> float:
    """How far is `value` from the normal centre, scaled 0–25."""
    r = _VITAL_RANGES.get(col)
    if r is None or np.isnan(value):
        return 0.0
    deviation = abs(value - r["centre"]) / r["half"]
    return float(np.clip(deviation * 25, 0, 25))


def _trend_bonus(slopes: dict[str, float]) -> float:
    """
    Add up to TREND_BONUS_MAX points if vitals are worsening.

    'Worsening' means:
      - SpO2 trending DOWN  (negative slope is bad)
      - HR trending UP      (positive slope is bad)
      - SBP trending DOWN   (hypotension risk)
    """
    score = 0.0
    # SpO2: negative slope → worsening
    if COL_SPO2 in slopes and slopes[COL_SPO2] < -0.01:
        score += min(abs(slopes[COL_SPO2]) * 100, TREND_BONUS_MAX / 3)
    # HR: positive slope → worsening
    if COL_HR in slopes and slopes[COL_HR] > 0.05:
        score += min(slopes[COL_HR] * 20, TREND_BONUS_MAX / 3)
    # SBP: negative slope → worsening
    if COL_SBP in slopes and slopes[COL_SBP] < -0.05:
        score += min(abs(slopes[COL_SBP]) * 10, TREND_BONUS_MAX / 3)

    return min(score, TREND_BONUS_MAX)


def _confidence(window: pd.DataFrame) -> float:
    """
    Confidence score (0–1) based on data quality:
      - Penalise missing values
      - Penalise high motion (noisy sensor readings)
    """
    missing_penalty = window[VITAL_COLS].isnull().mean().mean()  # 0–1
    motion_penalty = 0.0
    if COL_MOTION in window.columns:
        motion_penalty = window[COL_MOTION].mean() * 0.3  # motion up to 0.3 penalty

    return float(np.clip(1.0 - missing_penalty - motion_penalty, 0, 1))


def compute_risk(window: pd.DataFrame) -> dict:
    """
    Compute risk score, confidence, and alert status for a single window.

    Returns:
        dict with keys: risk_score, confidence, alert_status, details
    """
    # Per-vital severity
    severities = {}
    for col in VITAL_COLS:
        if col in window.columns:
            val = window[col].dropna()
            severities[col] = _vital_severity(
                val.mean() if len(val) else np.nan, col
            )
        else:
            severities[col] = 0.0

    # Weighted sum
    base_score = sum(
        severities[c] * RISK_WEIGHTS.get(c, 0) / 0.25 for c in severities
    )
    # Map weighted severities (0–25) to 0–100 scale

    # Trend (slope over last 3 minutes)
    slopes = {}
    for col in VITAL_COLS:
        if col in window.columns:
            vals = window[col].dropna()
            if len(vals) > 5:
                x = np.arange(len(vals))
                slopes[col] = np.polyfit(x, vals.values, 1)[0]

    bonus = _trend_bonus(slopes)
    risk_score = float(np.clip(base_score + bonus, 0, 100))

    conf = _confidence(window)

    # Alert logic
    if risk_score >= RISK_ALERT_THRESHOLD and conf >= CONFIDENCE_MIN:
        alert_status = "ALERT"
    elif risk_score >= RISK_ALERT_THRESHOLD and conf < CONFIDENCE_MIN:
        alert_status = "SUPPRESSED"
    else:
        alert_status = "NORMAL"

    return {
        "risk_score": round(risk_score, 1),
        "confidence": round(conf, 3),
        "alert_status": alert_status,
        "details": {
            "severities": severities,
            "trend_bonus": round(bonus, 1),
            "slopes": slopes,
        },
    }


def main() -> None:
    """Run risk scoring on all cleaned data and print a summary."""
    clean_dir = Path(DATA_CLEANED_DIR)
    csv_files = sorted(clean_dir.glob("*.csv"))
    if not csv_files:
        print("No cleaned data found. Run task_1b_artifact_detection first.")
        return

    for csv_path in csv_files:
        patient_id = csv_path.stem
        df = pd.read_csv(csv_path, index_col="time", parse_dates=True)
        print(f"\nPatient {patient_id}")

        alerts, suppressed, normal = 0, 0, 0
        for start in range(0, len(df) - WINDOW_SIZE_SEC, WINDOW_STRIDE_SEC):
            window = df.iloc[start : start + WINDOW_SIZE_SEC]
            result = compute_risk(window)

            match result["alert_status"]:
                case "ALERT":
                    alerts += 1
                case "SUPPRESSED":
                    suppressed += 1
                case _:
                    normal += 1

        total = alerts + suppressed + normal
        print(f"  Windows: {total}")
        print(f"  ALERT:      {alerts:>4}  ({100 * alerts / total:.1f}%)")
        print(f"  SUPPRESSED: {suppressed:>4}  ({100 * suppressed / total:.1f}%)")
        print(f"  NORMAL:     {normal:>4}  ({100 * normal / total:.1f}%)")


if __name__ == "__main__":
    main()
