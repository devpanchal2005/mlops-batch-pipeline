"""
MLOps Batch Job - Rolling-Mean Signal Pipeline
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    """Configure root logger to write to both file and stdout."""
    logger = logging.getLogger("mlops")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler (INFO and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Config loading / validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")

    missing = REQUIRED_CONFIG_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")

    if not isinstance(cfg["seed"], int):
        raise TypeError(f"'seed' must be an integer, got {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got {cfg['window']!r}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"'version' must be a non-empty string, got {cfg['version']!r}")

    return cfg


# ---------------------------------------------------------------------------
# Dataset loading / validation
# ---------------------------------------------------------------------------

def load_dataset(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if path.stat().st_size == 0:
        raise ValueError("Input file is empty.")

    try:
        # encoding="utf-8-sig" silently strips the Windows BOM (\ufeff) that
        # Excel and many Windows tools prepend to CSV files.  Without it,
        # pandas reads the first column name as '\ufeffTimestamp' (or similar),
        # the entire header appears as a single column, and 'close' is not found.
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("CSV parsed successfully but contains no data rows.")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Columns present: {list(df.columns)}"
        )

    if df["close"].isnull().all():
        raise ValueError("Column 'close' exists but contains only null values.")

    return df


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def compute_rolling_mean(close: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling mean with min_periods=window so the first (window-1)
    rows produce NaN. Those rows are excluded from signal computation.
    """
    return close.rolling(window=window, min_periods=window).mean()


def compute_signal(close: pd.Series, rolling_mean: pd.Series) -> pd.Series:
    """
    signal = 1 if close > rolling_mean, else 0.
    Rows where rolling_mean is NaN are set to NaN (excluded from rate calc).
    """
    signal = pd.Series(np.nan, index=close.index)
    valid = rolling_mean.notna()
    signal[valid] = (close[valid] > rolling_mean[valid]).astype(int)
    return signal


# ---------------------------------------------------------------------------
# Metrics writing
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict) -> None:
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MLOps batch job: rolling-mean binary signal pipeline."
    )
    parser.add_argument("--input",   required=True, help="Path to input OHLCV CSV.")
    parser.add_argument("--config",  required=True, help="Path to YAML config file.")
    parser.add_argument("--output",  required=True, help="Path for output metrics JSON.")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for structured log file.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_file)

    t_start = time.perf_counter()
    logger.info("=== Job START ===")
    logger.info("Input:   %s", args.input)
    logger.info("Config:  %s", args.config)
    logger.info("Output:  %s", args.output)
    logger.info("Log:     %s", args.log_file)

    version = "unknown"  # fallback before config is loaded

    try:
        # ------------------------------------------------------------------
        # 1. Load + validate config
        # ------------------------------------------------------------------
        logger.info("Loading config …")
        cfg = load_config(args.config)
        seed    = cfg["seed"]
        window  = cfg["window"]
        version = cfg["version"]
        logger.info(
            "Config loaded - version=%s  seed=%d  window=%d",
            version, seed, window,
        )

        # Set global RNG seed for reproducibility
        np.random.seed(seed)
        logger.info("NumPy random seed set to %d", seed)

        # ------------------------------------------------------------------
        # 2. Load + validate dataset
        # ------------------------------------------------------------------
        logger.info("Loading dataset from '%s' …", args.input)
        df = load_dataset(args.input)
        logger.info("Dataset loaded - %d rows, columns: %s", len(df), list(df.columns))

        close = df["close"].astype(float)

        # ------------------------------------------------------------------
        # 3. Rolling mean
        # ------------------------------------------------------------------
        logger.info("Computing rolling mean (window=%d) …", window)
        rolling_mean = compute_rolling_mean(close, window)
        nan_count = rolling_mean.isna().sum()
        logger.info(
            "Rolling mean computed - %d leading NaN rows excluded from signal.",
            nan_count,
        )

        # ------------------------------------------------------------------
        # 4. Signal generation
        # ------------------------------------------------------------------
        logger.info("Generating binary signal (close > rolling_mean -> 1, else -> 0) …")
        signal = compute_signal(close, rolling_mean)
        valid_signals = signal.dropna()
        rows_with_signal = len(valid_signals)
        signal_rate = float(valid_signals.mean())
        logger.info(
            "Signal generated - valid rows=%d  signal_rate=%.6f",
            rows_with_signal, signal_rate,
        )

        # ------------------------------------------------------------------
        # 5. Metrics + timing
        # ------------------------------------------------------------------
        latency_ms = round((time.perf_counter() - t_start) * 1000)
        rows_processed = len(df)

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          round(signal_rate, 4),
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }

        logger.info(
            "Metrics - rows_processed=%d  signal_rate=%.4f  latency_ms=%d",
            rows_processed, signal_rate, latency_ms,
        )

        write_metrics(args.output, metrics)
        logger.info("Metrics written to '%s'", args.output)

        logger.info("=== Job END - status=success ===")
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: %s", exc)
        error_metrics = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
        }
        try:
            write_metrics(args.output, error_metrics)
            logger.info("Error metrics written to '%s'", args.output)
        except Exception as write_exc:  # noqa: BLE001
            logger.error("Could not write error metrics: %s", write_exc)

        logger.info("=== Job END - status=error ===")
        print(json.dumps(error_metrics, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
