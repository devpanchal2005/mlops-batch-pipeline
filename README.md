# MLOps Batch Job - Rolling-Mean Signal Pipeline

A minimal, production-style ML batch job that demonstrates **reproducibility**, **observability**, and **deployment readiness**.

---

## Overview

The pipeline:

1. Loads and validates a YAML config (`seed`, `window`, `version`)
2. Reads a 10 000-row OHLCV CSV, validates the `close` column
3. Computes a rolling mean on `close` (configurable window)
4. Generates a binary signal: `1` if `close > rolling_mean`, else `0`
5. Writes structured metrics JSON and a detailed log file

---

## Project structure

```
.
├── run.py           # Main batch job
├── config.yaml      # Job configuration
├── data.csv         # Input OHLCV dataset (10 000 rows)
├── requirements.txt # Python dependencies
├── Dockerfile       # Docker build spec
├── metrics.json     # Sample output from a successful run
├── run.log          # Sample log from a successful run
└── README.md
```

---

## Local run

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

All four flags are required - no paths are hard-coded.

---

## Docker build & run

```bash
# Build the image
docker build -t mlops-task .

# Run (data.csv + config.yaml are baked into the image)
docker run --rm mlops-task
```

The container:

- Exits `0` on success, non-zero on failure
- Prints the final metrics JSON to stdout
- Writes `metrics.json` and `run.log` inside the container

To retrieve output files from a named container:

```bash
docker run --name mlops-run mlops-task
docker cp mlops-run:/app/metrics.json .
docker cp mlops-run:/app/run.log .
docker rm mlops-run
```

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.5002,
  "latency_ms": 50,
  "seed": 42,
  "status": "success"
}
```

Error case:

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Columns present: ['open', 'high']"
}
```

---

## Config reference (`config.yaml`)

| Key       | Type    | Description                            |
| --------- | ------- | -------------------------------------- |
| `seed`    | int     | NumPy random seed for reproducibility  |
| `window`  | int ≥ 1 | Rolling-mean window size (rows)        |
| `version` | string  | Pipeline version tag written to output |

---

## Design decisions

| Topic                 | Decision                                                                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| First `window-1` rows | Rolling mean uses `min_periods=window` -> NaN for first `window-1` rows; those rows are **excluded** from `signal_rate` calculation           |
| Error handling        | Both config and data errors write an error `metrics.json` and exit with code `1`                                                              |
| Determinism           | `numpy.random.seed(seed)` is called immediately after config load; `latency_ms` is wall-clock and varies, all other metrics are deterministic |
| Logging               | Dual-sink (file + stdout); file is DEBUG level, stdout is INFO                                                                                |

---

## Validation errors handled

- Config file not found
- Missing required config keys (`seed`, `window`, `version`)
- Wrong config value types (e.g. non-integer seed)
- Input CSV not found
- Empty input file
- Unparseable CSV
- Missing `close` column
- `close` column all-null
