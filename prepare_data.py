#!/usr/bin/env python3
"""
Step 1: Convert wind power .npy data [T, N, F] to TSlib custom CSV format.

Input shape expected:
- T (time steps): 13416
- N (nodes): 10
- F (features): 6, in order: [Power, U10, V10, U100, V100, Other]

Output:
- CSV at ./dataset/wind_dataset.csv
- Columns: date + 60 flattened features (Node{i}_{Feature})
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = ["Power", "U10", "V10", "U100", "V100", "Other"]


def _try_parse_datetime_column(values: np.ndarray) -> pd.DatetimeIndex | None:
    """Return parsed datetime index if values look like timestamps, else None."""
    try:
        parsed = pd.to_datetime(values, errors="raise")
        return parsed
    except Exception:
        return None


def load_npy_with_fallback(input_path: Path) -> tuple[np.ndarray, pd.DatetimeIndex | None]:
    """
    Load npy safely and normalize to numeric [T, N, F_num].

    Supports two common formats:
    1) Pure numeric tensor [T, N, F] (dtype float/int).
    2) Object tensor [T, N, F], where the first feature is embedded timestamp string.
    """
    try:
        raw = np.load(input_path, allow_pickle=False)
    except ValueError:
        raw = np.load(input_path, allow_pickle=True)

    if raw.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={raw.shape}")

    embedded_dates = None

    if raw.dtype == object:
        # Try detecting timestamp at [:, 0, 0]. If parseable, treat it as embedded date.
        maybe_dates = _try_parse_datetime_column(raw[:, 0, 0])
        if maybe_dates is not None:
            embedded_dates = maybe_dates
            # Keep only numeric features.
            numeric = raw[:, :, 1:]
        else:
            numeric = raw

        try:
            data = numeric.astype(np.float32)
        except Exception as exc:
            raise ValueError(
                "Loaded object array but failed to convert to numeric float32. "
                "Please check npy content format."
            ) from exc
    else:
        data = raw.astype(np.float32)

    return data, embedded_dates


def build_columns(num_nodes: int, feature_names: list[str]) -> list[str]:
    """Build flattened column names matching np.reshape(order='C')."""
    cols = []
    for node_idx in range(num_nodes):
        for feat in feature_names:
            cols.append(f"Node{node_idx}_{feat}")
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 3D wind .npy to TSlib CSV")
    parser.add_argument(
        "--input_npy",
        type=str,
        default="./data/wind_train_val_2012-01-02_to_2013-07-13.npy",
        help="Path to input .npy file",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="Time-Series-Library/dataset/wind_dataset.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default="2012-01-02 00:00:00",
        help="Start timestamp for hourly date column",
    )
    args = parser.parse_args()

    input_path = Path(args.input_npy)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data, embedded_dates = load_npy_with_fallback(input_path)

    t, n, f = data.shape

    if f > len(FEATURE_NAMES):
        raise ValueError(
            f"Numeric feature dim={f} is larger than known names={len(FEATURE_NAMES)}. "
            "Please expand FEATURE_NAMES to match your data."
        )

    used_feature_names = FEATURE_NAMES[:f]
    if f != len(FEATURE_NAMES):
        print(
            f"[Warn] Numeric feature dim={f}, using feature names: {used_feature_names}. "
            "If this is unexpected, check original npy schema."
        )

    # Flatten [T, N, F] -> [T, N*F].
    flat = data.reshape(t, n * f)

    # Build feature columns: Node0_Power ... Node9_Other.
    feature_cols = build_columns(num_nodes=n, feature_names=used_feature_names)
    if len(feature_cols) != flat.shape[1]:
        raise RuntimeError("Column count does not match flattened data width")

    df = pd.DataFrame(flat, columns=feature_cols)

    # Build hourly date range and insert as the first column named 'date'.
    if embedded_dates is not None:
        dates = embedded_dates
        print("[Info] Detected embedded timestamp in npy; using embedded date column.")
    else:
        dates = pd.date_range(start=args.start_time, periods=t, freq="H")
    df.insert(0, "date", dates)

    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Data conversion done")
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {df.shape}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
