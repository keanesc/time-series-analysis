"""Detect and clean motion artefacts; produce before-vs-after plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time_series_analysis.constants import (
    COL_DBP,
    COL_HR,
    COL_MOTION,
    COL_SBP,
    COL_SPO2,
    DATA_CLEANED_DIR,
    DATA_RAW_DIR,
    HR_SPIKE_THRESHOLD,
    MAX_GAP_FILL_SEC,
    MOTION_ARTIFACT_THRESHOLD,
    PLOTS_DIR,
    SPO2_DROP_THRESHOLD,
    VITAL_COLS,
)


def detect_motion_spo2_artifacts(df: pd.DataFrame) -> pd.Series:
    if COL_SPO2 not in df.columns or COL_MOTION not in df.columns:
        return pd.Series(False, index=df.index)

    high_motion = df[COL_MOTION] > MOTION_ARTIFACT_THRESHOLD
    spo2_diff = df[COL_SPO2].diff().abs()
    return high_motion & (spo2_diff > SPO2_DROP_THRESHOLD)


def detect_hr_spike_artifacts(df: pd.DataFrame) -> pd.Series:
    """Flag single-sample HR spikes that align with high motion to reduce false alarms."""
    if COL_HR not in df.columns or COL_MOTION not in df.columns:
        return pd.Series(False, index=df.index)

    high_motion = df[COL_MOTION] > MOTION_ARTIFACT_THRESHOLD
    hr_diff = df[COL_HR].diff().abs()
    return high_motion & (hr_diff > HR_SPIKE_THRESHOLD)


def detect_missing_segments(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Identify NaN runs in each vital column."""
    masks = {}
    for col in VITAL_COLS:
        if col in df.columns:
            masks[col] = df[col].isna()
    return masks


def clean_artifacts(
    df: pd.DataFrame,
    spo2_mask: pd.Series,
    hr_mask: pd.Series,
) -> pd.DataFrame:
    df = df.copy()
    window = 11
    # Impute SpO2 with centered rolling-median
    if COL_SPO2 in df.columns and spo2_mask.any():
        median = df[COL_SPO2].rolling(window, center=True, min_periods=1).median()
        df.loc[spo2_mask, COL_SPO2] = median[spo2_mask]

    # Preserves trend while reducing transient sensor spikes.
    if COL_HR in df.columns and hr_mask.any():
        median = df[COL_HR].rolling(window, center=True, min_periods=1).median()
        df.loc[hr_mask, COL_HR] = median[hr_mask]

    # FFill short gaps (≤ MAX_GAP_FILL_SEC)
    for col in VITAL_COLS:
        if col in df.columns:
            df[col] = df[col].ffill(limit=MAX_GAP_FILL_SEC)

    return df


def plot_before_after(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    artifact_masks: dict[str, pd.Series],
    patient_id: str,
    out_dir: Path,
) -> None:
    """Generate a before-vs-after subplot for each vital with artifact shading."""
    vitals = [c for c in VITAL_COLS if c in raw.columns]
    fig, axes = plt.subplots(
        len(vitals), 1, figsize=(14, 3 * len(vitals)), sharex=True
    )
    if len(vitals) == 1:
        axes = [axes]

    x = np.arange(len(raw))

    for ax, col in zip(axes, vitals):
        ax.plot(
            x, raw[col].values, alpha=0.5, linewidth=0.5, label="Raw", color="red"
        )
        ax.plot(
            x, cleaned[col].values, linewidth=0.7, label="Cleaned", color="steelblue"
        )

        mask = artifact_masks.get(col, pd.Series(False, index=raw.index))
        if mask.any():
            for start, end in _contiguous_regions(mask):
                ax.axvspan(start, end, alpha=0.15, color="orange", label="")

        ax.set_ylabel(col)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Patient {patient_id} — Artifact Detection", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"{patient_id}_artifacts.png", dpi=150)
    plt.close(fig)


def _contiguous_regions(mask: pd.Series) -> list[tuple[int, int]]:
    indices = np.where(mask.values)[0]
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, breaks)
    return [(g[0], g[-1]) for g in groups]


def main() -> None:
    raw_dir = Path(DATA_RAW_DIR)
    clean_dir = Path(DATA_CLEANED_DIR)
    plot_dir = Path(PLOTS_DIR) / "artifacts"
    clean_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        print("No raw data found. Run task_1a_data_collection first.")
        return

    for csv_path in csv_files:
        patient_id = csv_path.stem
        print(f"Processing {patient_id} …")

        raw = pd.read_csv(csv_path, index_col="time", parse_dates=True)

        spo2_mask = detect_motion_spo2_artifacts(raw)
        hr_mask = detect_hr_spike_artifacts(raw)
        missing = detect_missing_segments(raw)

        n_spo2 = spo2_mask.sum()
        n_hr = hr_mask.sum()
        n_missing = {col: m.sum() for col, m in missing.items()}
        print(
            f"  SpO2 artifacts: {n_spo2}  |  HR artifacts: {n_hr}  |  Missing: {n_missing}"
        )

        cleaned = clean_artifacts(raw, spo2_mask, hr_mask)

        # Build artifact masks for plotting
        artifact_masks = {}
        if COL_SPO2 in raw.columns:
            artifact_masks[COL_SPO2] = spo2_mask | missing.get(
                COL_SPO2, pd.Series(False, index=raw.index)
            )
        if COL_HR in raw.columns:
            artifact_masks[COL_HR] = hr_mask | missing.get(
                COL_HR, pd.Series(False, index=raw.index)
            )
        for col in [COL_SBP, COL_DBP]:
            if col in missing:
                artifact_masks[col] = missing[col]

        plot_before_after(raw, cleaned, artifact_masks, patient_id, plot_dir)

        cleaned.to_csv(clean_dir / f"{patient_id}.csv")
        print("Saved cleaned data and plot")


if __name__ == "__main__":
    main()
