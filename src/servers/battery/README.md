# Battery MCP Server

Lithium-ion battery analytics for AssetOpsBench: RUL, voltage curves, impedance growth, fleet outliers, LLM-narrated diagnosis. Wraps the pretrained acctouhou model (`.h5` / `.npy` artifacts, CPU-only) for the 10 NASA B0xx cells with ≥100 paired charge/discharge cycles; statistical tools work for all 34 cells.

## Design

Boot is minimal: load the four `.h5` models, load the four `.npy` norms, build flexible-shape compiled `tf.function` graphs. **No data fetch, no precompute.** Tools fetch + preprocess + predict on demand.

Three optimizations apply only to the optimized RUL path (`predict_rul_batch`):

1. **Parallel CouchDB fetch** - `ThreadPoolExecutor` overlaps the per-cell network waits.
2. **Compiled TF graphs** - `_CompiledKerasWrapper` wraps each Keras model in a `tf.function` with a flexible-batch input signature, so first inference does not pay per-shape Keras retrace.
3. **Batched predict** - all cells' inputs concatenated into one tensor; 3 TF predicts total instead of 3 × N.

The voltage tools (`predict_voltage_curve`, `predict_voltage_milestones`) deliberately use the **unoptimized** path: serial fetch, raw Keras model, per-cell predict. They serve as the reference "what does an un-optimized on-demand tool look like" - and as the `naive_baseline` rung the benchmark targets.

## Environment variables (add to your `.env`)

| Variable | Default | What it's for |
|----------|---------|---------------|
| `BATTERY_DBNAME` | `battery` | CouchDB database name |
| `BATTERY_MODEL_WEIGHTS_DIR` | `src/servers/battery/artifacts/weights` | directory holding the 4 `.h5` files |
| `BATTERY_NORMS_DIR` | `src/servers/battery/artifacts/norms` | directory holding the 4 `.npy` normalization files |
| `BATTERY_DATA_DIR` | `external/battery/nasa` | flat directory of NASA `B*.json` files (raw input, not artifacts) |
| `BATTERY_FETCH_WORKERS` | `4` | thread-pool size for `predict_rul_batch`'s parallel CouchDB fetch |
| `BATTERY_MODEL_ID` | inherits `FMSR_MODEL_ID` | LLM used by `diagnose_battery` |
| `COUCHDB_URL` / `COUCHDB_USERNAME` / `COUCHDB_PASSWORD` | see `.env.public` | shared with other servers |
| `BATTERY_PARALLEL_FETCH` | `1` | ablation toggle: `0` to force serial CouchDB fetch in `predict_rul_batch` |
| `BATTERY_GRAPH_PRECOMPILE` | `1` | ablation toggle: `0` to skip flexible-shape `tf.function` precompile and use raw Keras |
| `BATTERY_BATCHED_PREDICT` | `1` | ablation toggle: `0` to force per-cell predict in `predict_rul_batch` |
| `BATTERY_DISK_CACHE` | `1` | `0` to skip the `.npz` rul_trajectory cache; tools always run fresh TF predicts |

## First-time setup

1. **Install deps:** `uv sync --group battery` (pulls `tensorflow`, `tf_keras`, `scikit-learn`).
2. **Copy `.env`:** `cp .env.public .env` from the repo root, then fill in your LLM keys.
3. **Download acctouhou artifacts** (Apache 2.0). The 4 `.h5` weights + 4 `.npy` normalization files come from the Google Drive link in the original `Prediction_of_battery/README.md`. Drop them into `BATTERY_MODEL_WEIGHTS_DIR` and `BATTERY_NORMS_DIR`.
4. **Download NASA B0xx cycling data** (US Public Domain) from the NASA Prognostics Center. Put the `B*.json` files into the directory `BATTERY_DATA_DIR` points at (default `external/battery/nasa/`).
5. **Verify the artifacts loaded right:** `./scripts/setup_battery_artifacts.sh`.
6. **Start CouchDB:**
   ```bash
   docker compose -f docker-compose.couchdb.yml up -d
   ```
7. **Load battery data into CouchDB.** This reads from `BATTERY_DATA_DIR` and writes the cells in `BATTERY_CELL_SUBSET`:
   ```bash
   uv run python -m couchdb.init_battery --drop
   ```
8. **Start the server (only if you want to run it standalone, plan-execute starts it automatically):** `uv run battery-mcp-server`.

## Sample plan-execute queries (known to produce real output)

These two queries have been tested end-to-end on `cerebras/llama3.1-8b` and return concrete, data-grounded answers — no LLM hallucinations of accuracy metrics.

**Fleet RUL ranking** (uses `predict_rul_batch`):

```bash
uv run plan-execute "Predict the remaining useful life in cycles for cells B0005, B0006, B0007, B0018, B0033, B0034, B0036, B0054, B0055, and B0056. Rank them from worst to best by remaining cycles, and tell me which 3 cells are closest to end-of-life." --model-id "cerebras/llama3.1-8b"
```

**Outlier detection on the fleet** (uses `detect_capacity_outliers` — statistical, no LFP→NCA chemistry-mismatch caveat):

```bash
uv run plan-execute "Even out of the factory, no two cells are identical, but we need to catch the outliers fast. Using our accelerated testbed data, establish a baseline degradation curve for a standard cell batch. Then, run an anomaly detection prediction to identify which cells are degrading significantly faster than the baseline due to intrinsic manufacturing variability. Flag the top 5% of cells that deviate from the standard State-of-Life (SOL) curve early in their cycle life. Show me the divergence graphs and the error margins. I need to prove to the manufacturing floor that we can spot defective cells within the first 50 cycles." --model-id "cerebras/llama3.1-8b"
```

`uv run pytest src/servers/battery/tests/` runs the preprocessing tests (no backends needed).

## Tools (10)

| Tool | What it does | Optimized? |
|------|--------------|---:|
| `list_batteries` | List cells | – |
| `get_battery_cycle_summary` | Per-cycle Capacity / max_T / avg_V / Rct / Re | – |
| `predict_rul` | Single-cell RUL to 30% fade (1.4 Ah) | compiled graphs |
| `predict_rul_batch` | RUL for many cells in one MCP call | parallel fetch + compiled + batched |
| `predict_voltage_curve` | 100-point V-SOC curve | naive (reference) |
| `predict_voltage_milestones` | First cycle V drops below threshold | naive (reference) |
| `get_actual_voltage_milestones` | Ground-truth V threshold crossings | – |
| `get_impedance_trajectory` | Per-cycle Rct / Re / Rectified_Impedance mag | – |
| `analyze_impedance_growth` | Rct exponential growth fit + alarm | – |
| `detect_capacity_outliers` | Fleet z-score over capacity fade rate | – |
| `diagnose_battery` | Combined RUL + impedance + outlier with LLM narration | – |

`predict_rul_batch` is the preferred entry point for fleet questions. Each MCP tool call spawns a fresh subprocess that pays TF model load + graph compile, so collapsing N calls into 1 is the biggest single optimization for fleet workloads.

## Profiling

Three scripts under [`profiling/`](profiling/). Each writes a small JSON to [`profiles/`](profiles/).

**Optimization ablation** - 5-rung ladder, in-process (no MCP / no LLM):

```bash
uv run python -m servers.battery.profiling.benchmark_optimizations --repeats 3
```

Rungs:

| Rung | parallel_fetch | graph_precompile | batched_predict | disk_cache |
|---|:---:|:---:|:---:|:---:|
| `naive_baseline` | ✗ | ✗ | ✗ | ✗ |
| `+ parallel_fetch` | ✓ | ✗ | ✗ | ✗ |
| `+ graph_precompile` | ✓ | ✓ | ✗ | ✗ |
| `+ batched_predict` | ✓ | ✓ | ✓ | ✗ |
| `+ disk_cache` | ✓ | ✓ | ✓ | ✓ |

Each rung clears `artifacts/cache/`, runs one untimed warmup, then R=3 timed repeats. The cache rung's warmup populates `.npz` files; subsequent timed repeats are warm hits that skip TF predict. Stage breakdown (`fetch_s` / `predict_s`) accompanies each `wall_s`.

**MCP-level batch demo** - compares N per-cell `predict_rul` MCP calls vs one `predict_rul_batch` MCP call. No planner, no LLM — direct `_call_tool` invocation:

```bash
uv run python -m servers.battery.profiling.mcp_batch_demo
```

Default cells = 10 NASA model-ready cells, default repeats = 1. Each per-cell run is ~14 s, so a 10-cell repeat takes ~2.5 minutes; the batched run takes ~15 s. The output JSON reports `mcp_per_cell.wall_s`, `mcp_batched.wall_s`, and `speedup`.

**Scenario runner** - runs plan-execute scenarios end-to-end and dumps per-step timings (no tool responses or plan structure — just wall times):

```bash
uv run python -m servers.battery.profiling.profile_scenario \
    --scenarios 1,2,4,5,6,7,8 \
    --model-id "cerebras/llama3.1-8b"
```

Writes one JSON per scenario into `profiles/scenarios_<ts>/scenario_<n>.json`. Each contains `discovery_s`, `planner_s`, `summary_s`, plus per-step `tool`, `step_s`, `tool_s`. Failures are reported but skipped — the loop continues.

## Validation status

The acctouhou model was trained on the Severson LFP dataset; NASA B0xx cells are NCA 18650. The pretrained pipeline runs end-to-end on NASA data, but **expect significant RUL error without fine-tuning** - on B0005 the predicted RUL at cycle 100 is ~3900 cycles vs ground truth ~24 cycles. This is a chemistry/distribution mismatch, not a pipeline bug.

Statistical tools (`analyze_impedance_growth`, `detect_capacity_outliers`, `get_battery_cycle_summary`) operate directly on NASA measurements and are unaffected.

## File layout

```
src/servers/battery/
  main.py                     FastMCP server, 10 tools, Pydantic result models
  preprocessing.py            NASA JSON cycle -> (4, 500) tensor [Q, V, I, T]
  model_wrapper.py            TF model loader + compiled graph wrappers + predict helpers
  couchdb_client.py           CouchDB fetch helpers
  diagnosis.py                LLM-narrated failure-mode classifier
  chemistries.yaml            Li-ion NCA 18650 thresholds
  artifacts/                  (gitignored)
    weights/                  4 × .h5 pretrained Keras weights
    norms/                    4 × .npy normalization tensors
    cache/                    .npz rul_trajectory per cell (cache_save / cache_load)
  profiles/                   profiling JSON outputs (tracked in git)
  profiling/
    benchmark_optimizations.py  5-rung in-process ablation of the 4 optimizations
    mcp_batch_demo.py           MCP per-cell vs batched: subprocess-spawn savings
    profile_scenario.py         end-to-end plan-execute timing breakdown per scenario
  tests/
```
