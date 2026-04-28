# Battery MCP Server — Profiling Guide

Comprehensive profiling setup for measuring and improving the performance of
`src/servers/battery/`. Covers preprocessing throughput, TF model inference,
statistical tool execution, and memory footprint. Results are saved as JSON so
any two runs can be diff'd to quantify optimization gains.

---

## 1. Why These Tools (and Why Not Others)

### What we use

| Tool | Why |
|------|-----|
| Python `cProfile` | Built-in call-graph profiler. Zero-overhead outside measurement windows. Shows which functions accumulate the most CPU time. |
| `line_profiler` | Line-by-line timing for hot functions. Essential for identifying which line inside `inp_500` or `preprocess_cycle` is the bottleneck (interp1d construction vs. evaluation). |
| `tracemalloc` | Built-in Python memory tracer. Measures net allocation per section without instrumenting the code. |
| `psutil` | Process-level RSS/VMS memory and CPU frequency. Gives the "wall" memory footprint visible to the OS. |
| `tf.profiler.experimental` | TF op-level trace saved for TensorBoard. Shows how time splits across individual Keras layers (CNN conv ops, batch-norm, etc.) during `feature_selector` and `rul.predict`. |

### Why not PyTorch Profiler

The acctouhou model is saved in Keras 2 `.h5` format and loaded via
`tf_keras` (TensorFlow 2). **PyTorch Profiler only instruments PyTorch ops**
and has no visibility into TensorFlow execution graphs. Using it here would
produce an empty trace.

### Why not NVIDIA Nsight / nvprof

`model_wrapper._load_once()` explicitly checks `tf.config.list_physical_devices("GPU")`.
On this deployment it returns `[]`, so the code forces a `float32` CPU policy
and never dispatches to CUDA. **NVIDIA profilers require CUDA hardware** and
would produce no meaningful data. If a GPU is added in the future,
`nvidia-smi`, Nsight Systems, and the TF Profiler GPU trace would become the
right tools for kernel-level optimization.

---

## 2. Installation

```bash
# Install the battery heavy deps (TF, tf_keras) if not already done
uv sync --group battery

# Install profiling extras
uv sync --group profiling
```

This adds `psutil`, `line-profiler`, and `memory-profiler` to the venv.
`cProfile` and `tracemalloc` are part of the Python standard library and need
no installation.

---

## 3. Profiling Architecture

```
src/servers/battery/profiling/
  __init__.py          package marker
  mock_data.py         Synthetic NASA-like charge/discharge/impedance cycles.
                       No CouchDB required. Data shape mirrors real B0xx cells.
  profiler.py          BatteryProfiler class. Five sections, each independently
                       runnable. Saves structured JSON to profiles/.
  compare.py           Load two JSON result files and print a colour-diff table.
  run_profile.py       CLI wrapper around BatteryProfiler.

profiles/              Created automatically on first run.
  baseline_<ts>.json   Results from the pre-optimisation run.
  optimized_<ts>.json  Results after implementing changes.
  comparison_<ts>.json Optional: saved output of compare.py.
  tf_profile/          TensorBoard-compatible TF op traces (--tf-trace).
```

---

## 4. Profiling Sections

### Section 1 — `preprocessing`

Targets `preprocessing.py`:

| Metric | What it measures |
|--------|-----------------|
| `inp_500.mean_ms` | Cost of one `scipy.interp1d` call (construct + evaluate, 500 output points). Called **4× per cycle** (Q, V, I, T), so 800× per cell at boot. |
| `preprocess_cycle_discharge.mean_ms` | Full `(4, 500)` tensor construction for one discharge cycle including Coulomb counting. |
| `preprocess_cycle_charge.mean_ms` | Same for charge cycles (no `cumulative_trapezoid` but same interpolation). |
| `preprocess_cell_from_couchdb.mean_ms` | End-to-end time per cell: fetch + loop 100 cycles. |
| `preprocess_cell_from_couchdb.cycles_per_sec` | Throughput: paired charge+discharge cycles processed per second. |
| `cprofile.preprocess_cell_from_couchdb` | Full call-graph breakdown (top-25 functions by cumulative time). |
| `line_profiles.inp_500` | Line-level timing showing `interp1d` constructor vs `f(t_new)` cost. |
| `line_profiles.preprocess_cycle` | Shows `cumulative_trapezoid` vs four `inp_500` calls. |

**Key optimization target**: `inp_500` uses `scipy.interp1d` which constructs
a new interpolation object every call. For linear-only interpolation,
`numpy.interp` is 3–8× faster and produces identical results. Additionally,
all four channels share the same time axis `t`, so a vectorized batch
interpolation can eliminate redundant setup work.

### Section 2 — `sliding_windows`

Targets the `np.stack` loop in `model_wrapper.precompute_cell`:

```python
windows = np.stack(
    [_pad_edge(cell_feat[max(0, k - 49): k + 1], 50) for k in range(len(cell_feat))]
)  # → (100, 50, 12)
```

| Metric | What it measures |
|--------|-----------------|
| `sliding_window_stack_100x50x12.mean_ms` | Time to build the full (100, 50, 12) tensor using the current Python list comprehension. |
| `sliding_window_stack_100x50x12.memory_net_mb` | Memory allocated during construction (includes all intermediate arrays from the comprehension). |

**Key optimization target**: `np.lib.stride_tricks.sliding_window_view`
constructs the same sliding window as a zero-copy view — no intermediate
allocations and no Python loop. Expected speedup: 10–50× for this operation.

### Section 3 — `model_inference`

Targets `model_wrapper.py`. Skipped gracefully if weights are unavailable.

| Metric | What it measures |
|--------|-----------------|
| `feature_selector_charge.mean_ms` | `fs_ch.predict(100 cycles, batch_size=128)` |
| `feature_selector_discharge.mean_ms` | `fs_dis.predict(100 cycles, batch_size=128)` |
| `concat_data_normalization.mean_ms` | Normalize + hstack into (100, 12) feature matrix |
| `sliding_window_construction.mean_ms` | Sliding window in inference context |
| `rul_predict.mean_ms` / `.ms_per_cycle` | `predictor.h5` forward pass over 100 windows |
| `volt_predict.mean_ms` / `.ms_per_cycle` | `predictor2.h5` forward pass over 100 windows |
| `precompute_cell_full.mean_ms` | Total end-to-end cost for one cell |
| `boot_all_cells.total_ms` | Simulated `_boot()` for all N cells |

**Reference target from `battery.md`**: ~13 ms/cycle inference.

**Optimization opportunities**:
- Replace `model.predict()` with direct `model(inputs, training=False)` (avoids
  `predict` overhead for already-batched inputs).
- Convert `.h5` weights to `tf.saved_model` format for faster deserialization.
- Use TFLite quantization for 2–4× inference speedup on CPU (INT8 weights).
- Batch all cells through `feature_selector` in one call instead of per-cell loops.

### Section 4 — `statistical_tools`

Targets the numerical cores of the statistics tools in `main.py`:

| Metric | What it measures |
|--------|-----------------|
| `np_polyfit_impedance.mean_ms` | `np.polyfit` for Rct exponential fit (called once per cell per `analyze_impedance_growth`). |
| `capacity_fade_rate_loop.mean_ms` | Python loop over all cells computing fade rate (bottleneck of `detect_capacity_outliers`). |
| `zscore_dict_comprehension.mean_ms` | Dict comprehension for z-score normalization. |
| `summary_row_construction.mean_ms` | Full `get_battery_cycle_summary` row loop. |
| `summary_row_construction.ms_per_row` | Per-cycle cost exposing Python `max()/min()/sum()` on lists. |

**Key optimization targets**:
- `summary_row_construction`: Python `max(temps)`, `min(temps)`, `sum(volts)/len(volts)` on
  Python lists is slow for 500-element arrays. `np.max`, `np.min`, `np.mean` on pre-cast
  arrays run 5–20× faster.
- `capacity_fade_rate_loop`: Sequential per-cell fetch from CouchDB with Python loops.
  Vectorizing with numpy after fetching or using `concurrent.futures.ThreadPoolExecutor`
  can give 3–10× throughput gains for fleet-wide operations.

### Section 5 — `memory`

| Metric | What it measures |
|--------|-----------------|
| `tensor_sizes_bytes` | Theoretical size of key intermediate arrays (float32). |
| `preprocessing_one_cell_net_alloc_mb` | Net MB allocated by `preprocess_cell_from_couchdb` for one cell. |
| `process_rss_mb` | OS-reported resident set size of the Python process after all sections. |

---

## 5. Running the Profiler

### Step 1 — Capture baseline

```bash
# All sections, 3 synthetic cells (no weights needed for preprocessing/stats):
uv run python -m servers.battery.profiling.run_profile --label baseline

# Include model inference (requires weights in BATTERY_MODEL_WEIGHTS_DIR):
uv run python -m servers.battery.profiling.run_profile --label baseline --with-model

# More cells for realistic boot timing:
uv run python -m servers.battery.profiling.run_profile \
    --label baseline --with-model --n-cells 10

# Quick smoke-test (skip slow inference, use fewer cycles):
uv run python -m servers.battery.profiling.run_profile \
    --label smoke --n-cycles 20 --sections preprocessing stats memory
```

Output: `profiles/baseline_<timestamp>Z.json`

### Step 2 — Inspect the baseline

The profiler prints a summary table immediately. For deep-dive, look at:

- **`profiles/baseline_*.json` → `cprofile.preprocess_cell_from_couchdb`**:
  the full call-graph dump showing where the cell preprocessing time goes.
- **`profiles/baseline_*.json` → `line_profiles.inp_500`**:
  line-level breakdown of `scipy.interp1d` construction vs evaluation.
- **`profiles/baseline_*.json` → `cprofile.precompute_cell`**:
  the TF inference call graph.

Open in Python:
```python
import json
data = json.load(open("profiles/baseline_<ts>.json"))
print(data["cprofile"]["preprocess_cell_from_couchdb"])
print(data["line_profiles"]["inp_500"])        # requires line_profiler installed
```

### Step 3 — Optional: TF op-level trace

```bash
uv run python -m servers.battery.profiling.run_profile \
    --label baseline --with-model --tf-trace

# View in TensorBoard:
uv run tensorboard --logdir profiles/tf_profile
# Open http://localhost:6006 → "Profile" tab
```

The TF trace reveals time spent in individual Keras layers (`Conv1D`,
`BatchNormalization`, `LSTM`, etc.) within `feature_selector` and
`rul.predict`. This is the right tool for deciding whether to fuse layers,
apply quantization, or switch to TFLite.

### Step 4 — Implement optimizations

Priority order based on expected impact:

1. **`inp_500` → `np.interp`** (preprocessing.py line 26–27)
   ```python
   # Before
   from scipy.interpolate import interp1d
   def inp_500(x, t):
       f = interp1d(t, x, kind="linear")
       t_new = np.linspace(t.min(), t.max(), num=500)
       return f(t_new)

   # After
   def inp_500(x, t):
       t_new = np.linspace(t[0], t[-1], num=500)
       return np.interp(t_new, t, x)
   ```
   Expected: 3–8× faster per call. With 800 calls/cell × 10 cells, this compounds significantly.

2. **Vectorize all 4 channels in `preprocess_cycle`** (preprocessing.py)
   ```python
   # Before: 4 separate inp_500 calls
   return np.stack([inp_500(Q, t), inp_500(V, t), inp_500(I, t), inp_500(T, t)])

   # After: one vectorized call over a (4, n) array
   channels = np.stack([Q, V, I_arr, T_arr])           # (4, n)
   t_new = np.linspace(t[0], t[-1], 500)
   return np.array([np.interp(t_new, t, ch) for ch in channels])
   # or fully vectorized with scipy.interpolate.interp1d over 2D if desired
   ```

3. **Stride-tricks sliding window** (model_wrapper.py `precompute_cell`)
   ```python
   # Before: O(n) np.stack with list comprehension
   windows = np.stack([_pad_edge(cell_feat[max(0,k-49):k+1], 50) for k in range(100)])

   # After: pad once, then zero-copy strided view
   padded = np.pad(cell_feat, ((49, 0), (0, 0)), mode="edge")   # (149, 12)
   windows = np.lib.stride_tricks.sliding_window_view(padded, (50, 12))\
               .reshape(100, 50, 12)
   ```
   Expected: 10–50× faster; eliminates all intermediate allocations.

4. **`model.predict` → `model(inputs, training=False)`** (model_wrapper.py)
   ```python
   # Before
   result = model.predict(inputs, batch_size=256, verbose=0)

   # After (avoids predict() overhead for pre-batched arrays)
   import tensorflow as tf
   result = model(inputs, training=False).numpy()
   ```

5. **Vectorize `get_battery_cycle_summary` temperature/voltage stats** (main.py)
   ```python
   # Before
   "max_temp_c": max(temps) if temps else None,
   "min_temp_c": min(temps) if temps else None,
   "avg_voltage": (sum(volts) / len(volts)) if volts else None,

   # After
   t_arr = np.asarray(temps)
   v_arr = np.asarray(volts)
   "max_temp_c": float(t_arr.max()) if len(t_arr) else None,
   "min_temp_c": float(t_arr.min()) if len(t_arr) else None,
   "avg_voltage": float(v_arr.mean()) if len(v_arr) else None,
   ```

6. **Parallel CouchDB fetches in `detect_capacity_outliers`** (main.py)
   ```python
   from concurrent.futures import ThreadPoolExecutor
   def _fetch_one(cell):
       return cell, client.fetch_cycles(cell, cycle_type="discharge")
   with ThreadPoolExecutor(max_workers=8) as pool:
       results = dict(pool.map(lambda c: _fetch_one(c), cells))
   ```

### Step 5 — Capture optimized run

```bash
uv run python -m servers.battery.profiling.run_profile \
    --label optimized --with-model
```

Output: `profiles/optimized_<timestamp>Z.json`

### Step 6 — Compare pre/post

```bash
# Auto-picks newest baseline_* and optimized_* files:
uv run python -m servers.battery.profiling.compare \
    "profiles/baseline_*.json" "profiles/optimized_*.json"

# With explicit paths:
uv run python -m servers.battery.profiling.compare \
    profiles/baseline_20260428T120000Z.json \
    profiles/optimized_20260428T150000Z.json

# Save comparison report as JSON:
uv run python -m servers.battery.profiling.compare \
    "profiles/baseline_*.json" "profiles/optimized_*.json" \
    --save profiles/comparison_report.json

# No colour (for CI/log files):
uv run python -m servers.battery.profiling.compare \
    "profiles/baseline_*.json" "profiles/optimized_*.json" --no-color

# Show only changes ≥5%:
uv run python -m servers.battery.profiling.compare \
    "profiles/baseline_*.json" "profiles/optimized_*.json" --min-pct 5.0
```

The comparison table has four status values:
- `IMPROVED` — metric moved in the desired direction by ≥ `--min-pct`%
- `REGRESSED` — metric moved in the wrong direction by ≥ `--min-pct`%
- `—` — change is below the threshold (noise)

The CLI exits with code `1` if any metric regressed, making it suitable for CI gates.

---

## 6. Reading the JSON Output

```json
{
  "label": "baseline",
  "timestamp": "2026-04-28T12:00:00+00:00",
  "config": { "n_cells": 3, "n_cycles": 100, "use_real_model": true },
  "system": {
    "python_version": "3.12.3",
    "platform": "Linux ...",
    "cpu_count_logical": 8,
    "total_ram_gb": 16.0,
    "cpu_freq_mhz": 2400.0
  },
  "sections": {
    "preprocessing": {
      "inp_500": {
        "calls": 500,
        "total_ms": 77.2,
        "mean_ms": 0.1544,   ← main target
        "min_ms": 0.1320,
        "max_ms": 0.4810,
        "throughput_per_sec": 6477
      },
      "preprocess_cycle_discharge": {
        "calls": 100,
        "mean_ms": 2.2840,   ← 4× inp_500 + cumulative_trapezoid
        "memory_net_mb": 3.2
      },
      "preprocess_cell_from_couchdb": {
        "calls": 3,
        "mean_ms": 512.4,    ← full 100-cycle pipeline per cell
        "cycles_per_sec": 195.2
      }
    },
    "sliding_windows": {
      "sliding_window_stack_100x50x12": {
        "mean_ms": 45.2,     ← target for stride_tricks optimization
        "memory_net_mb": 1.1
      }
    },
    "model_inference": {
      "feature_selector_charge":   { "mean_ms": 180.0 },
      "feature_selector_discharge":{ "mean_ms": 190.0 },
      "rul_predict":               { "mean_ms": 95.0, "ms_per_cycle": 0.95 },
      "volt_predict":              { "mean_ms": 110.0, "ms_per_cycle": 1.1 },
      "precompute_cell_full":      { "mean_ms": 590.0, "ms_per_cycle": 5.9 },
      "boot_all_cells":            { "total_ms": 1800.0, "n_cells": 3 }
    },
    "statistical_tools": {
      "np_polyfit_impedance":      { "mean_ms": 0.014 },
      "capacity_fade_rate_loop":   { "mean_ms": 12.3, "ms_per_cell": 4.1 },
      "summary_row_construction":  { "mean_ms": 8.4, "ms_per_row": 0.084 }
    },
    "throughput_summary": {
      "preprocess_cycles_per_sec": 438,
      "inp_500_calls_per_sec":     6477,
      "inference_ms_per_cycle":    5.9
    }
  },
  "memory": {
    "tensor_sizes_bytes": { ... },
    "preprocessing_one_cell_net_alloc_mb": 3.2,
    "process_rss_mb": 512.0
  },
  "cprofile": {
    "preprocess_cell_from_couchdb": "   ncalls  tottime  ... (top 25 functions)",
    "precompute_cell":              "   ncalls  tottime  ..."
  },
  "line_profiles": {
    "inp_500":         "Line #   Hits   Time  ...  % Time  Line Contents",
    "preprocess_cycle": "..."
  }
}
```

---

## 7. Metrics Reference

| Metric | Unit | Lower/Higher better | Notes |
|--------|------|---------------------|-------|
| `*.mean_ms` | ms | lower | Primary timing metric |
| `*.total_ms` | ms | lower | Useful for cumulative cost across all calls |
| `*.throughput_per_sec` | calls/s | higher | Inverse of mean_ms |
| `*.cycles_per_sec` | cycles/s | higher | Cell-level throughput |
| `*.ms_per_cycle` | ms | lower | Per-inference cost (target: ≤13 ms) |
| `*.ms_per_row` | ms | lower | Per-row cost for summary construction |
| `*.memory_net_mb` | MB | lower | Net bytes allocated during the section |
| `process_rss_mb` | MB | lower | OS-level resident memory (whole process) |
| `boot_all_cells.total_ms` | ms | lower | Total cold-start time |

---

## 8. Profiling Without Model Weights

All sections except `model_inference` run entirely on synthetic data —
no CouchDB, no TF weights needed.

```bash
# Profile preprocessing + stats + memory with no external dependencies:
uv run python -m servers.battery.profiling.run_profile \
    --label baseline --sections preprocessing sliding_windows stats memory
```

The `inference` section is skipped with a clear log message when weights are absent.

---

## 9. Profiling with Real NASA Data

To profile with actual data from CouchDB instead of mock data, use the
`profiler.py` API directly:

```python
import sys; sys.path.insert(0, "src")
from servers.battery.profiling.profiler import BatteryProfiler
from servers.battery.couchdb_client import CouchDBClient

# Replace mock fleet with real CouchDB client in BatteryProfiler._mock_client:
profiler = BatteryProfiler(label="baseline_real", n_cells=10, n_cycles=100)
profiler._mock_client = CouchDBClient()        # swap in real client
profiler._fleet = None                         # unused when mock_client is real
profiler.profile_preprocessing()
profiler.profile_statistical_tools()
profiler.save()
```

---

## 10. Running Multiple Comparison Passes

You can capture multiple labels (e.g., one optimization at a time):

```bash
uv run python -m servers.battery.profiling.run_profile --label baseline
# → implement np.interp optimization
uv run python -m servers.battery.profiling.run_profile --label opt1_np_interp
uv run python -m servers.battery.profiling.compare \
    profiles/baseline_*.json profiles/opt1_np_interp_*.json

# → implement stride_tricks optimization
uv run python -m servers.battery.profiling.run_profile --label opt2_stride_tricks
uv run python -m servers.battery.profiling.compare \
    profiles/opt1_np_interp_*.json profiles/opt2_stride_tricks_*.json

# → final comparison vs original baseline
uv run python -m servers.battery.profiling.compare \
    profiles/baseline_*.json profiles/opt2_stride_tricks_*.json \
    --save profiles/final_comparison.json
```

---

## 11. Expected Improvement Targets

Based on static code analysis of the current implementation:

| Optimization | Affected metric | Expected improvement |
|---|---|---|
| `np.interp` in `inp_500` | `inp_500.mean_ms` | 3–8× faster |
| Vectorized channel interp in `preprocess_cycle` | `preprocess_cycle_discharge.mean_ms` | 2–4× faster |
| `stride_tricks` sliding window | `sliding_window_stack_100x50x12.mean_ms` | 10–50× faster |
| `model(x, training=False)` vs `.predict()` | `rul_predict.mean_ms` | 5–20% faster |
| `np.max/min/mean` in summary rows | `summary_row_construction.ms_per_row` | 5–15× faster |
| Parallel CouchDB fetch in outliers | `capacity_fade_rate_loop.mean_ms` | 3–8× faster (I/O bound) |
| TFLite INT8 quantization (future) | `rul_predict.ms_per_cycle` | 2–4× faster |

---

## 12. File Layout

```
src/servers/battery/profiling/
  __init__.py          package marker + usage docs
  mock_data.py         Synthetic NASA-like battery cycle data generator
  profiler.py          BatteryProfiler — 5 sections, JSON output
  compare.py           CLI + importable comparison tool with colour diff
  run_profile.py       CLI entry point (python -m servers.battery.profiling.run_profile)

profiles/              auto-created by the profiler
  baseline_*.json      pre-optimisation results
  optimized_*.json     post-optimisation results
  comparison_*.json    optional saved diff reports
  tf_profile/          TF op traces for TensorBoard (--tf-trace)
```

---

## 13. References

- acctouhou/Prediction_of_battery — pretrained model source (Apache 2.0)
- NASA Prognostics Center — raw cycling data (US Public Domain)
- `battery.md` — server architecture, environment setup, and validation status
- `docs/battery_server_plan.md` — full design rationale
- [Python cProfile docs](https://docs.python.org/3/library/profile.html)
- [line_profiler PyPI](https://pypi.org/project/line-profiler/)
- [TF Profiler guide](https://www.tensorflow.org/guide/profiler)
- [numpy stride_tricks](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html)
