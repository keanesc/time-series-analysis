"""Anomaly detection using gradient-boosted clinical features.

Why gradient boosting, not CNN? At 1 Hz, 60 samples lack temporal micro-patterns
for conv filters. The signal is distance-to-threshold, slope, and variability —
tabular features better suited to HistGradientBoostingClassifier.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

from time_series_analysis.constants import (
    COL_DBP,
    COL_HR,
    COL_MOTION,
    COL_SBP,
    COL_SPO2,
    DATA_CLEANED_DIR,
    DBP_HIGH,
    DBP_LOW,
    EARLY_WARNING_HORIZON_SEC,
    HR_HIGH,
    HR_LOW,
    MODELS_DIR,
    SBP_HIGH,
    SBP_LOW,
    SPO2_LOW,
    WINDOW_SIZE_SEC,
    WINDOW_STRIDE_SEC,
)

CHANNELS = [COL_HR, COL_SPO2, COL_SBP, COL_DBP]

BASELINES = {COL_HR: 80.0, COL_SPO2: 97.0, COL_SBP: 120.0, COL_DBP: 70.0}

_THRESHOLDS = {
    COL_HR: (HR_LOW, HR_HIGH),
    COL_SPO2: (SPO2_LOW, None),
    COL_SBP: (SBP_LOW, SBP_HIGH),
    COL_DBP: (DBP_LOW, DBP_HIGH),
}

_BREACH_CHECKS = [
    (COL_HR, lambda v: (v < HR_LOW) | (v > HR_HIGH)),
    (COL_SPO2, lambda v: v < SPO2_LOW),
    (COL_SBP, lambda v: (v < SBP_LOW) | (v > SBP_HIGH)),
    (COL_DBP, lambda v: (v < DBP_LOW) | (v > DBP_HIGH)),
]


def is_pre_critical(future: pd.DataFrame) -> bool:
    """True if any vital breaches its clinical threshold in *future*."""
    for col, cond in _BREACH_CHECKS:
        if col in future.columns:
            vals = future[col].dropna()
            if len(vals) > 0 and cond(vals).any():
                return True
    return False


def is_currently_stable(window: pd.DataFrame, max_breach_frac: float = 0.20) -> bool:
    breach_count = 0
    total_count = 0
    for col, cond in _BREACH_CHECKS:
        if col in window.columns:
            vals = window[col].dropna()
            if len(vals) > 0:
                breach_count += int(cond(vals).sum())
                total_count += len(vals)
    if total_count == 0:
        return True
    return (breach_count / total_count) < max_breach_frac


def _min_distance_to_threshold(values: np.ndarray, lo, hi) -> float:
    """Smallest distance between any reading and its nearest threshold.

    A low distance means the vital is near a clinical boundary and a
    small perturbation could cause a breach.  This is the single most
    predictive feature for threshold-crossing prediction.
    """
    distances = []
    if lo is not None:
        distances.append(np.min(np.abs(values - lo)))
    if hi is not None:
        distances.append(np.min(np.abs(values - hi)))
    return float(min(distances)) if distances else np.nan


def _linear_slope(values: np.ndarray) -> float:
    """Slope of a least-squares linear fit (units per second)."""
    x = np.arange(len(values), dtype=np.float64)
    # polyfit returns [slope, intercept]
    return float(np.polyfit(x, values, 1)[0])


def extract_window_features(window: pd.DataFrame) -> dict[str, float]:
    """Extract clinically-meaningful features from a single window.

    Features per vital (×4 vitals = 20 features):
      - mean:      Current central tendency
      - slope:     Linear trend direction (+ rising, - falling)
      - min_dist:  Minimum distance to nearest clinical threshold
      - end_val:   Most recent value (closest to prediction point)
      - std:       Variability / instability indicator

    Aggregate features (4):
      - motion_mean:  Mean motion level (artifact risk)
      - missing_frac: Fraction of NaN readings (data quality)
      - n_warning:    Number of vitals within 10% of a threshold
      - max_abs_slope: Largest |slope| across vitals (overall trend strength)
    """
    feats: dict[str, float] = {}

    slopes = []
    for col in CHANNELS:
        if col not in window.columns:
            feats[f"{col}_mean"] = np.nan
            feats[f"{col}_slope"] = np.nan
            feats[f"{col}_min_dist"] = np.nan
            feats[f"{col}_end_val"] = np.nan
            feats[f"{col}_std"] = np.nan
            continue

        vals = window[col].ffill().bfill()
        clean = vals.dropna().values
        if len(clean) < 5:
            feats[f"{col}_mean"] = np.nan
            feats[f"{col}_slope"] = np.nan
            feats[f"{col}_min_dist"] = np.nan
            feats[f"{col}_end_val"] = np.nan
            feats[f"{col}_std"] = np.nan
            continue

        feats[f"{col}_mean"] = float(clean.mean())
        feats[f"{col}_std"] = float(clean.std())
        feats[f"{col}_end_val"] = float(clean[-1])

        slope = _linear_slope(clean)
        feats[f"{col}_slope"] = slope
        slopes.append(abs(slope))

        lo, hi = _THRESHOLDS[col]
        feats[f"{col}_min_dist"] = _min_distance_to_threshold(clean, lo, hi)

    if COL_MOTION in window.columns:
        feats["motion_mean"] = float(window[COL_MOTION].mean())
    else:
        feats["motion_mean"] = 0.0

    avail = [c for c in CHANNELS if c in window.columns]
    feats["missing_frac"] = (
        float(window[avail].isnull().mean().mean()) if avail else 1.0
    )

    # Use proximity thresholds to improve early warning sensitivity
    n_warning = 0
    for col in CHANNELS:
        if f"{col}_min_dist" in feats and not np.isnan(feats[f"{col}_min_dist"]):
            lo, hi = _THRESHOLDS[col]
            ref_range = (hi or lo) - (lo or 0)
            if ref_range > 0 and feats[f"{col}_min_dist"] < 0.10 * ref_range:
                n_warning += 1
    feats["n_warning"] = float(n_warning)

    feats["max_abs_slope"] = float(max(slopes)) if slopes else np.nan

    return feats


def build_dataset(
    csv_files: list[Path],
    window_size: int = WINDOW_SIZE_SEC,
    stride: int = WINDOW_STRIDE_SEC,
    horizon: int = EARLY_WARNING_HORIZON_SEC,
) -> pd.DataFrame:
    """Build a feature matrix from cleaned patient CSVs.

    Returns a DataFrame with one row per (stable) sliding window,
    containing all features, the binary label, patient_id, and
    window_start index.  The second return value is the list of
    feature column names (excludes metadata and label).
    """
    rows = []
    for csv_path in csv_files:
        # Skip non-patient files
        if csv_path.stem == "evaluation_results":
            continue
        patient_id = csv_path.stem
        df = pd.read_csv(csv_path, index_col="time", parse_dates=True)
        n = len(df)

        for start_idx in range(0, n - window_size, stride):
            end_idx = start_idx + window_size
            window = df.iloc[start_idx:end_idx]

            # Skip high-missing/critical windows
            avail = [c for c in CHANNELS if c in window.columns]
            if avail and window[avail].isnull().mean().mean() > 0.5:
                continue
            if not is_currently_stable(window):
                continue

            feats = extract_window_features(window)
            feats["window_start"] = start_idx
            feats["patient_id"] = patient_id

            future_end = min(end_idx + horizon, n)
            future = df.iloc[end_idx:future_end]
            feats["label"] = int(is_pre_critical(future))

            rows.append(feats)

    result = pd.DataFrame(rows)

    feature_cols = [
        c for c in result.columns if c not in ("window_start", "patient_id", "label")
    ]

    return result, feature_cols


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> HistGradientBoostingClassifier:
    """Train a gradient-boosted classifier and print evaluation metrics."""

    clf = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.1,
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Pre-critical"],
            zero_division=0,
        )
    )

    # Probability calibration check
    print(f"\n  Prob|pos  — mean={y_proba[y_test == 1].mean():.3f}")
    neg_probs = y_proba[y_test == 0]
    if len(neg_probs):
        print(f"  Prob|neg  — mean={neg_probs.mean():.3f}")

    return clf


# Inference helpers


def load_model(
    model_dir: str | Path = MODELS_DIR,
) -> tuple[HistGradientBoostingClassifier, list[str]]:
    """Load saved model and return (classifier, feature_column_names)."""
    artifact = joblib.load(Path(model_dir) / "anomaly_model.joblib")
    return artifact["classifier"], artifact["feature_cols"]


def predict_window(
    clf: HistGradientBoostingClassifier,
    feature_cols: list[str],
    window: pd.DataFrame,
) -> int:
    """Run inference on a single window DataFrame.  Returns 0 or 1."""
    feats = extract_window_features(window)
    X = np.array([[feats.get(c, np.nan) for c in feature_cols]])
    return int(clf.predict(X)[0])


def main() -> None:
    clean_dir = Path(DATA_CLEANED_DIR)
    model_dir = Path(MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(clean_dir.glob("*.csv"))
    if not csv_files:
        print("No cleaned data found. Run task_1b first.")
        return

    print("Building feature matrix …")
    dataset, feature_cols = build_dataset(csv_files)
    print(f"Total windows: {len(dataset)}")

    # 70/30 chronological split
    split_idx = int(len(dataset) * 0.7)
    train_df = dataset.iloc[:split_idx]
    test_df = dataset.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(
        f"  Train: {len(X_train)} ({int(y_train.sum())} positive, {y_train.mean():.1%})"
    )
    print(
        f"  Test:  {len(X_test)} ({int(y_test.sum())} positive, {y_test.mean():.1%})"
    )

    clf = train_model(X_train, y_train, X_test, y_test, feature_cols)

    joblib.dump(
        {"classifier": clf, "feature_cols": feature_cols},
        model_dir / "anomaly_model.joblib",
    )
    print(f"\nModel saved to {model_dir / 'anomaly_model.joblib'}")


if __name__ == "__main__":
    main()
