"""Fetch MIMIC-III patient numerics at 1 Hz, simulate motion, and inject artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from time_series_analysis.constants import (
    ALL_COLS,
    COL_DBP,
    COL_HR,
    COL_MOTION,
    COL_SBP,
    COL_SPO2,
    DATA_RAW_DIR,
    PHYSIONET_RECORDS,
)

SIGNAL_MAP: dict[str, str] = {
    "HR": COL_HR,
    "PULSE": COL_HR,
    "SpO2": COL_SPO2,
    "%SpO2": COL_SPO2,
    # Invasive arterial BP
    "ABPSys": COL_SBP,
    "ABP Sys": COL_SBP,
    "ABPDias": COL_DBP,
    "ABP Dias": COL_DBP,
    # Non-invasive cuff BP (more realistic for ambulance context)
    "NBPSys": COL_SBP,
    "NBP Sys": COL_SBP,
    "NBPDias": COL_DBP,
    "NBP Dias": COL_DBP,
}

MAX_SAMPLES = 7200
MIN_SAMPLES = 300


def fetch_record(record_name: str, pn_dir: str) -> pd.DataFrame | None:
    try:
        hdr = wfdb.rdheader(record_name, pn_dir=pn_dir)
        sampto = min(hdr.sig_len, MAX_SAMPLES)
        if sampto < MIN_SAMPLES:
            print(f"{record_name}: only {sampto} samples — too short, skipping")
            return None
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampfrom=0, sampto=sampto)
    except Exception as exc:
        print(f"Could not fetch {record_name}: {exc}")
        return None

    df = pd.DataFrame(record.p_signal, columns=record.sig_name)

    rename = {col: SIGNAL_MAP[col] for col in df.columns if col in SIGNAL_MAP}
    df = df.rename(columns=rename)

    # Drop duplicated signal columns
    df = df.loc[:, ~df.columns.duplicated()]
    available = [c for c in [COL_HR, COL_SPO2, COL_SBP, COL_DBP] if c in df.columns]
    if len(available) < 3:
        print(f"{record_name}: only {available} available — skipping")
        return None

    df = df[available].copy()

    # Create a 1 Hz datetime index
    start = pd.Timestamp("2025-01-01")  # arbitrary anchor
    df.index = pd.date_range(start, periods=len(df), freq="1s")
    df.index.name = "time"

    return df


def simulate_motion(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    baseline = np.cumsum(rng.normal(0, 0.01, n_samples))
    baseline = (
        0.3 * (baseline - baseline.min()) / (baseline.max() - baseline.min() + 1e-9)
    )

    bumps = np.zeros(n_samples)
    n_bumps = max(1, n_samples // 600)  # roughly one bump per 10 min
    for _ in range(n_bumps):
        centre = rng.integers(20, n_samples - 20)
        width = rng.integers(3, 9)
        magnitude = rng.uniform(0.4, 1.0)
        window = np.arange(max(0, centre - width), min(n_samples, centre + width))
        bumps[window] += magnitude * np.exp(
            -0.5 * ((window - centre) / (width / 2)) ** 2
        )

    motion = np.clip(baseline + bumps, 0, 1)
    return motion


def inject_motion_artifacts(
    df: pd.DataFrame, motion: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    df = df.copy()
    high_motion = motion > 0.6

    if COL_SPO2 in df.columns:
        drop = rng.uniform(3, 8, size=len(df)) * high_motion
        df[COL_SPO2] = df[COL_SPO2] - drop

    if COL_HR in df.columns:
        spike = rng.uniform(15, 40, size=len(df)) * high_motion
        df[COL_HR] = df[COL_HR] + spike

    # Inject NaN gaps (~1–4s at ~20% bumps)
    bump_indices = np.where(high_motion)[0]
    if len(bump_indices) > 0:
        n_gaps = max(1, len(bump_indices) // 20)
        gap_starts = rng.choice(bump_indices, size=n_gaps, replace=False)
        for start in gap_starts:
            gap_len = rng.integers(1, 5)
            end = min(start + gap_len, len(df))
            df.iloc[start:end] = np.nan

    return df


def main() -> None:
    out_dir = Path(DATA_RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    summary_rows = []

    for record_name, pn_dir in PHYSIONET_RECORDS:
        print(f"Fetching {record_name} …")
        df = fetch_record(record_name, pn_dir)
        if df is None:
            continue

        motion = simulate_motion(len(df), rng)
        df = inject_motion_artifacts(df, motion, rng)
        df[COL_MOTION] = motion

        cols = [c for c in ALL_COLS if c in df.columns]
        df = df[cols]

        patient_id = record_name.replace("n", "")
        path = out_dir / f"{patient_id}.csv"
        df.to_csv(path)
        print(f"Saved {path}  ({len(df)} samples, {len(df) / 3600:.1f} h)")

        missing_pct = df.isna().mean() * 100
        summary_rows.append(
            {
                "record": record_name,
                "patient_id": patient_id,
                "duration_h": len(df) / 3600,
                "signals": list(df.columns),
                **{f"missing_{c}_%": round(missing_pct.get(c, 0), 1) for c in cols},
            }
        )

    print("\nSummary")
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
