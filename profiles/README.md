# Profiling artifacts (`profiles/`)

This directory holds **local, machine-specific** benchmark outputs (JSON per run + optional `results.csv`). It is gitignored except for this README.

## Scope (inference-only track)

These tools measure **TensorFlow / Keras `predict` time** for the battery RUL and voltage heads (Part **B** of the internal profiling plan). They do **not** implement CouchDB vs mock vs in-memory **preprocessing** tiers (Part **A**). For end-to-end cold-start including DB fetch and `preprocess_cell`, add a separate benchmark or track an issue.

## Naming

- JSON: `<label>_<ISO-timestamp>.json` (UTC, colons stripped in the filename), e.g. `inference_baseline_2026-04-28T143022Z.json`.
- **label**: short slug you choose when invoking the benchmark.

## Suggested labels (taxonomy)

| Label example | Meaning |
|---------------|--------|
| `inference_baseline` | Default TF threading; sequential vs batched reference |
| `inference_batched_sweep` | `--batch-sizes 64,128,256` knee search |
| `inference_tf_intra_sweep` | Parent run using `--sweep-intra` (one child JSON per combo) |
| `inference_threadpool_w8` | Notes reference thread sweep with max workers 8 |
| `inference_nb1` / `inference_nb4` | Pair `--n-batteries 1` vs `4` with same prefix to estimate per-call overhead |
| `inference_fs_multicell` | `--include-feature-selectors --fs-batched-multicell` |
| `inference_npz_real` | Loaded `--windows-npz` from saved tensors |

## Wall time vs CPU time

JSON metrics include **`wall_ms`** (`time.perf_counter`) and **`cpu_process_ms`** (`time.process_time`). **ThreadPoolExecutor** can **lower wall** while **raising total CPU** — that is expected. Use **`--cpu`** on the comparator when you care about CPU shifts.

## Batch size knee

Batched RUL/VOLT timings use inner `batch_size` / `--batch-sizes`. If **larger** `batch_size` stops helping or gets **slower**, you may be **memory- or cache-bound**; if **faster**, you were **dispatch-bound**. Compare keys `rul_predict_batched_bs*` / `volt_predict_batched_bs*` in one JSON.

## NPZ format (`--windows-npz`)

Save a file with:

- **`windows`**: float32 array, shape `(n_cycles, 50, 12)` for one cell (replicated `N` times via `--n-batteries`), or shape `(n_batteries, n_cycles, 50, 12)`.
- Optional **`charges`**, **`discharges`**, **`summary`**: same layout as `precompute_cell` — `(n_cycles, 4, 500)` ×2 and `(n_cycles, 6)`, one cell (replicated per battery).

**Export snippet** (after you have NumPy arrays in Python, e.g. from a debug session or a one-off script):

```python
np.savez(
    "profiles/cell_B0018_windows.npz",
    windows=windows.astype("float32"),
    charges=charges.astype("float32"),
    discharges=discharges.astype("float32"),
    summary=summary.astype("float32"),
)
```

## Process pools

**`ProcessPoolExecutor` is not supported** here: each worker would load TensorFlow and weights separately (multi-second, high RAM), and **fork + loaded TF** is unsafe on many platforms. The benchmark accepts **`--experimental-process-pool`** only to print a warning and annotate JSON notes.

## Warm-up

Each timed block (RUL sequential, RUL batched per batch size, volt sequential, volt batched, combined RUL+V, threadpool modes) is preceded by the same **`--warmup`** count of **untimed** rounds so the first timed repeat is less affected by one-off overhead. See `warmup_rounds_per_mode` in JSON `config`.

## Measurement key aliases (plan vocabulary)

| Detailed key | Alias / meaning |
|--------------|-----------------|
| `rul_predict_sequential` | `model_predict_rul_sequential_ms` (duplicate blob) |
| `rul_predict_batched` (or `rul_predict_batched_bs*`) | `model_predict_rul_batched_ms` (uses first batch size in sweep) |
| `volt_predict_sequential` | `model_predict_volt_sequential_ms` |
| `volt_predict_batched` (or `*_bs*`) | `model_predict_volt_batched_ms` |

Also see **`wall_ms_per_battery`** under each measurement (total wall / `n_cells`).

## Compare two runs

Default: **`wall_ms`** means, flag changes **> 5%**:

```bash
python scripts/compare_battery_profiles.py profiles/run_a.json profiles/run_b.json
```

Custom threshold or **CPU** time:

```bash
python scripts/compare_battery_profiles.py profiles/a.json profiles/b.json --threshold 10 --cpu
```

## CSV (`results.csv`)

Appended columns include **`rul_batched_speedup`** and **`volt_batched_speedup`** (sequential mean / batched mean, using the **first** batch size in `--batch-sizes`), plus **`best_rul_threadpool_mean_ms`** / **`best_volt_threadpool_mean_ms`** when thread modes ran (minimum wall mean across worker counts).

## Benchmark entrypoint

```bash
uv run benchmark-battery-inference --label my_run --notes "what changed"
```

**TF thread sweep** (one subprocess per pair, fresh interpreter — required for thread config to apply):

```bash
uv run benchmark-battery-inference --label inference_tf --sweep-intra 1,2,4,8 --sweep-inter 1,2
```

**Paired N=1 vs N=4** (two commands, same label prefix + notes):

```bash
uv run benchmark-battery-inference --label inference_nb --n-batteries 1 --notes "N=1 overhead"
uv run benchmark-battery-inference --label inference_nb --n-batteries 4 --notes "N=4 total / 4 = per-battery"
```

Or: `PYTHONPATH=src uv run --group battery python -m servers.battery.benchmark_inference` if you prefer an explicit module path.
