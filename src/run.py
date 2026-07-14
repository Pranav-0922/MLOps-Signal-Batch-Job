#!/usr/bin/env python3
"""
run.py - Minimal MLOps-style batch job.

Loads a YAML config, reads an OHLCV CSV, computes a rolling mean on `close`,
derives a binary signal (close > rolling_mean), and writes structured
metrics JSON + detailed logs. Deterministic given the same input + seed.

Usage:
    python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REQUIRED_CONFIG_FIELDS = ("seed", "window", "version")
REQUIRED_COLUMN = "close"


def parse_args():
    parser = argparse.ArgumentParser(description="MLOps batch signal job")
    parser.add_argument("--input", required=True, help="Path to input CSV (OHLCV data)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to write log file")
    return parser.parse_args()


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    if not isinstance(config, dict):
        raise ValueError("Invalid config structure: expected a YAML mapping of key/value pairs")

    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Config missing required field(s): {', '.join(missing)}")

    if not isinstance(config["seed"], int):
        raise ValueError("Config field 'seed' must be an integer")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError("Config field 'window' must be a positive integer")
    if not isinstance(config["version"], str):
        raise ValueError("Config field 'version' must be a string")

    logger.info(
        "Config loaded + validated: seed=%s, window=%s, version=%s",
        config["seed"], config["window"], config["version"],
    )
    return config


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file is empty or has no parsable columns: {input_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format: {e}")

    if df.empty:
        raise ValueError(f"Input file contains no data rows: {input_path}")

    if REQUIRED_COLUMN not in df.columns:
        raise ValueError(f"Required column '{REQUIRED_COLUMN}' not found in input data")

    if not pd.api.types.is_numeric_dtype(df[REQUIRED_COLUMN]):
        try:
            df[REQUIRED_COLUMN] = pd.to_numeric(df[REQUIRED_COLUMN])
        except (ValueError, TypeError):
            raise ValueError(f"Column '{REQUIRED_COLUMN}' must contain numeric values")

    logger.info("Rows loaded: %d", len(df))
    return df


def compute_rolling_mean_and_signal(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    # First (window - 1) rows will have NaN rolling mean (min_periods=window enforces this
    # explicitly rather than relying on default behavior). These rows are excluded from the
    # signal_rate calculation below to keep the metric well-defined and reproducible.
    df = df.copy()
    df["rolling_mean"] = df[REQUIRED_COLUMN].rolling(window=window, min_periods=window).mean()
    logger.info("Rolling mean computed (window=%d)", window)

    df["signal"] = np.where(df[REQUIRED_COLUMN] > df["rolling_mean"], 1, 0)
    # Rows without a defined rolling mean have no meaningful signal; mark as NaN so
    # they're excluded from signal_rate rather than silently counted as 0.
    df.loc[df["rolling_mean"].isna(), "signal"] = np.nan
    logger.info("Signal generated (1 if close > rolling_mean, else 0)")

    return df


def compute_metrics(df: pd.DataFrame, config: dict, start_time: float, logger: logging.Logger) -> dict:
    valid_signals = df["signal"].dropna()
    rows_processed = len(df)
    signal_rate = float(valid_signals.mean()) if len(valid_signals) > 0 else 0.0
    latency_ms = int(round((time.perf_counter() - start_time) * 1000))

    metrics = {
        "version": config["version"],
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": latency_ms,
        "seed": config["seed"],
        "status": "success",
    }

    logger.info(
        "Metrics summary: rows_processed=%d, signal_rate=%.4f, latency_ms=%d",
        rows_processed, signal_rate, latency_ms,
    )
    return metrics


def write_metrics(output_path: str, metrics: dict, logger: logging.Logger):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", output_path)


def main():
    args = parse_args()
    logger = setup_logging(args.log_file)
    start_time = time.perf_counter()

    logger.info("Job start")
    logger.info("Args: input=%s config=%s output=%s log_file=%s",
                args.input, args.config, args.output, args.log_file)

    try:
        config = load_config(args.config, logger)
        np.random.seed(config["seed"])

        df = load_dataset(args.input, logger)
        df = compute_rolling_mean_and_signal(df, config["window"], logger)
        metrics = compute_metrics(df, config, start_time, logger)

        write_metrics(args.output, metrics, logger)

        print(json.dumps(metrics, indent=2))
        logger.info("Job end: status=success")
        sys.exit(0)

    except Exception as e:
        latency_ms = int(round((time.perf_counter() - start_time) * 1000))
        error_metrics = {
            "version": "v1",
            "status": "error",
            "error_message": str(e),
        }
        try:
            with open(args.output, "w") as f:
                json.dump(error_metrics, f, indent=2)
        except Exception as write_err:
            logger.error("Failed to write error metrics file: %s", write_err)

        logger.error("Exception occurred: %s", e, exc_info=True)
        print(json.dumps(error_metrics, indent=2))
        logger.info("Job end: status=error (latency_ms=%d)", latency_ms)
        sys.exit(1)


if __name__ == "__main__":
    main()
