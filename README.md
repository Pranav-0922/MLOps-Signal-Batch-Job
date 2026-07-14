# MLOps Batch Signal Job

A minimal, reproducible batch job that reads OHLCV data, computes a rolling
mean on `close`, derives a binary trading signal, and writes structured
metrics + logs. Built for Task 0 of the Primetrade.ai ML/MLOps internship
assessment.

## What it does

1. Loads and validates `config/config.yaml` (`seed`, `window`, `version`)
2. Loads and validates `data/data.csv` (must contain a `close` column)
3. Computes a rolling mean on `close` over `window` rows
4. Generates a binary signal: `1` if `close > rolling_mean`, else `0`
5. Writes `metrics.json` (machine-readable) and `run.log` (human-readable)

**Handling of the first `window - 1` rows:** the rolling mean is undefined
until `window` observations are available, so those rows get `NaN` for
`rolling_mean` and `signal`. They are excluded from the `signal_rate`
metric (rather than being counted as 0), so the reported rate reflects only
rows with a well-defined signal. `rows_processed` still counts every row in
the input.

## Repo structure

```
.
├── src/
│   └── run.py            # main script
├── config/
│   └── config.yaml       # seed / window / version
├── data/
│   └── data.csv           # sample OHLCV data (10,000 rows)
├── requirements.txt
├── Dockerfile
├── README.md
├── metrics.json           # sample output from a successful run
└── run.log                # sample log from a successful run
```

## Local run

```bash
pip install -r requirements.txt

python src/run.py \
  --input data/data.csv \
  --config config/config.yaml \
  --output metrics.json \
  --log-file run.log
```

The script prints the final metrics JSON to stdout, writes it to
`--output`, writes logs to `--log-file`, and exits `0` on success or
non-zero on any failure (missing/invalid input, missing `close` column,
invalid/incomplete config, etc.). No paths are hardcoded inside `run.py` —
everything comes from the CLI flags above.

## Docker

Build:

```bash
docker build -t mlops-task .
```

Run:

```bash
docker run --rm mlops-task
```

The image bundles `data/data.csv` and `config/config.yaml` and runs the
exact CLI command internally, so `docker run --rm mlops-task` alone
reproduces the full job, prints the metrics JSON to stdout, and exits `0`
on success.

To pull the generated `metrics.json`/`run.log` out onto your host:

```bash
docker run --rm -v "$(pwd)/out:/app/out" mlops-task \
  python src/run.py --input data/data.csv --config config/config.yaml \
  --output out/metrics.json --log-file out/run.log
```

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4982,
  "latency_ms": 36,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found in input data"
}
```

## Determinism

Given the same `data/data.csv` and `config/config.yaml`, `rows_processed`,
`signal_rate`, and `seed` are identical across runs (verified across
repeated runs, including inside and outside Docker). Only `latency_ms`
varies, since it measures wall-clock runtime rather than a computed result.

## Notes

- `data/data.csv` in this repo is a synthetically generated 10,000-row
  OHLCV dataset (random-walk `close` with derived `open`/`high`/`low`/
  `volume`), generated with a fixed seed for reproducibility, since no
  dataset was provided alongside the assessment brief.
