# Battery MCP Server

Lithium-ion battery analytics for AssetOpsBench: RUL, voltage curves, impedance growth, fleet outliers, LLM-narrated diagnosis. Wraps the pretrained acctouhou model (`.h5` / `.npy` artifacts, CPU-friendly, ~13 ms/cycle inference) for the 10 NASA B0xx cells with enough cycles (≥100 clean paired charge/discharge cycles); statistical tools work for all 34 cells.

## Environment variables (add to your `.env`)

| Variable | Default | What it's for |
|----------|---------|---------------|
| `BATTERY_DBNAME` | `battery` | CouchDB database name |
| `BATTERY_MODEL_WEIGHTS_DIR` | `external/battery/acctouhou/weights` | directory holding the 4 `.h5` files |
| `BATTERY_NORMS_DIR` | `external/battery/acctouhou/norms` | directory holding the 4 `.npy` normalization files |
| `BATTERY_DATA_DIR` | `external/battery/nasa` | flat directory of NASA `B*.json` files |
| `BATTERY_CELL_SUBSET` | 14-cell prototyping subset | comma-separated asset IDs, or `all` |
| `BATTERY_MODEL_ID` | inherits `FMSR_MODEL_ID` | LLM used by `diagnose_battery` |
| `COUCHDB_URL` / `COUCHDB_USERNAME` / `COUCHDB_PASSWORD` | see `.env.public` | shared with other servers |
| `BATTERY_FS_BATCH_SIZE` | `128` | inner batch for feature_selector `.predict` / `__call__` |
| `BATTERY_HEAD_BATCH_SIZE` | `256` | batch size for RUL / voltage heads |
| `BATTERY_TF_INTRA_OP_THREADS` | TF default | set before load (use `0` or unset to keep default) |
| `BATTERY_TF_INTER_OP_THREADS` | TF default | set before load |
| `BATTERY_GPU_FORCE_FLOAT32` | unset | if `1` and GPU visible, force float32 policy (vs mixed_float16) |
| `BATTERY_KERAS_USE_CALL` | unset | if `1`, use `model(x, training=False)` instead of `predict` where supported |
| `BATTERY_LAZY_VOLTAGE` | unset | if `1`, skip voltage head at boot; fill on first voltage tool (faster boot) |
| `BATTERY_BOOT_PARALLEL_FETCH` | `1` | if `0`, sequential CouchDB preprocess during `_boot` |
| `BATTERY_BOOT_FETCH_WORKERS` | `4` | thread pool size for parallel fetch |
| `BATTERY_BOOT_BATCH_FS` | unset | if `1`, run batched feature selectors across all boot cells (shared `n_cycles`) |
| `TF_ENABLE_ONEDNN_OPTS` | `0` via model_wrapper default | set to `1` in `.env` on **Intel** to try oneDNN (benchmark vs baseline); Apple Silicon often irrelevant |

See [docs/speedup_baseline.md](docs/speedup_baseline.md) (baseline JSON + success criteria), [docs/inference_alt_runtimes.md](docs/inference_alt_runtimes.md) (SavedModel / TFLite / ONNX / XLA notes).

**Concurrency:** prefer **batching** and TF thread envs over `ThreadPoolExecutor` around `predict` unless your benchmark proves a wall-time win ([profiles/README.md](profiles/README.md)).

### After each speed PR

Re-run `benchmark-battery-inference` with the same label prefix and note the JSON path in the PR (see `docs/speedup_baseline.md`).

## First-time setup

1. **Install deps:** `uv sync --group battery` (pulls `tensorflow`, `tf_keras`, `scikit-learn`).
2. **Download acctouhou artifacts** (Apache 2.0) — the 4 `.h5` weights + 4 `.npy` normalization files — from the Google Drive link in `.prediction_of_battery/Prediction_of_battery/README.md`. The zip contains two folders: `pretrained/` (the `.h5` files) and `dataset/` (the `.npy` files plus others we ignore).
3. **Download NASA B0xx cycling data** (US Public Domain) from the NASA Prognostics Center. Flatten all `B*.json` files into `external/battery/nasa/`.
4. **Verify:** `./scripts/setup_battery_artifacts.sh`.
5. **Seed CouchDB:** `./src/couchdb/couchdb_setup.sh` — or run directly: `uv run python src/couchdb/init_battery.py --drop`.
6. **Start the server:** `uv run battery-mcp-server`.

## What you can test

**Smoke tests via plan-execute:**

```bash
uv run plan-execute "What batteries are available at site MAIN?"
uv run plan-execute "Predict RUL for B0018"
uv run plan-execute "When will B0018 voltage drop below 2.8V?"
uv run plan-execute "Diagnose battery B0018"
```

**All 15 scenarios in `battery_scenarios.json`:**

```bash
for i in $(seq 0 14); do
  q=$(jq -r ".battery_management_scenarios[$i].request" battery_scenarios.json)
  echo "=== Scenario $((i+1)) ==="
  uv run plan-execute --show-plan "$q"
done
```

**Pytest:** `uv run pytest src/servers/battery/tests/` (preprocessing tests always run; tool + validation tests skip without CouchDB/weights).

## Inference profiling (Part B + C)

Measure **RUL / voltage** `predict` throughput (sequential vs batched, optional `ThreadPoolExecutor`, TF thread sweeps, optional feature-selector path) and write **JSON + CSV** under `profiles/` (gitignored except `profiles/README.md`).

```bash
uv sync --group battery
uv run benchmark-battery-inference --label inference_baseline --repeats 3 --notes "CPU default threads"
```

- **Compare two JSON runs** (default: flag `wall_ms` mean changes &gt; 5%):  
  `python scripts/compare_battery_profiles.py profiles/a.json profiles/b.json`
- **Labels, NPZ format, batch-size knee, CPU vs wall:** see [profiles/README.md](profiles/README.md).

## Tools (9)

| Tool | What it does | Uses model? |
|------|--------------|-------------|
| `list_batteries` | List available cells with `model_ready` flag | no |
| `get_battery_cycle_summary` | Per-cycle Capacity / max_T / avg_V / Rct / Re | no |
| `predict_rul` | Remaining cycles to 1.4 Ah (30% fade) | yes — `predictor.h5` |
| `predict_rul_batch` | RUL for many cached cells in one call | yes — cache |
| `predict_voltage_curve` | 100-point V-SOC curve for a cycle | yes — `predictor2.h5` |
| `predict_voltage_milestones` | Cycle where V first crosses a threshold | yes — `predictor2.h5` |
| `analyze_impedance_growth` | Rct exponential growth rate + alarm | no — `scipy.polyfit` |
| `detect_capacity_outliers` | Fleet z-score over capacity fade rate | no — numpy |
| `diagnose_battery` | Combined RUL + impedance + outlier with LLM narration | yes + LLM |

Abstract scenarios like *"predict RUL for test cells and list top 12 at risk"* are orchestrated by the plan-execute agent: it calls `list_batteries` → `predict_rul` or `predict_rul_batch` → sorts → returns top N. Tools stay single-cell primitives on purpose except for the optional batch RUL helper.

## Example use cases from `battery_scenarios.json`

1. **Commercial EV Fleet Operator** — "Predict RUL until 30% fade; list top 12 at risk" → `list_batteries` → loop `predict_rul` → LLM ranks
2. **BMS Hardware Engineer** — "Predict EOD time; show voltage drop curves" → `predict_voltage_milestones` + `predict_voltage_curve`
3. **QA Lead** — "Flag cells degrading faster than standard within 50 cycles" → `detect_capacity_outliers(window=50)`
4. **Microgrid Operator** — "Flag cells with exponential Rct spikes" → `analyze_impedance_growth` per cell
5. **Fleet Safety Officer** — "Analyze EIS for electrolyte-resistance trajectory" → `analyze_impedance_growth`
6. **Full diagnosis** — "What's wrong with B0018?" → `diagnose_battery`

## Validation status

The acctouhou model was trained on the Severson LFP dataset; NASA B0xx cells are NCA 18650. The pretrained pipeline runs end-to-end on NASA data, but **expect significant RUL error without fine-tuning** — on B0005 the predicted RUL at cycle 100 is ~3900 cycles vs ground truth ~24 cycles. This is a chemistry/distribution mismatch, not a pipeline bug.

**Tested and confirmed working:**

- Preprocessing pipeline produces the expected `(4, 500)` tensors per cycle.
- Summary features are 6-dim as required by `summary_norm.npy`.
- Inference speed is ~13 ms/cycle on CPU (well under scenario 5's target).
- All 5 preprocessing unit tests pass.

**Not fine-tuned:** if accurate NASA RUL matters, the next step is LoRA-style fine-tuning on B0005/B0006/B0007 and holding out B0018 for validation (out of scope for v1).

**Statistical tools (`analyze_impedance_growth`, `detect_capacity_outliers`, `get_battery_cycle_summary`, `list_batteries`)** don't have this issue — they operate directly on NASA measurements and are unaffected by the chemistry mismatch.

## Troubleshooting

- **Server boots but `predict_rul` returns "Pretrained model unavailable"** → weights aren't in `BATTERY_MODEL_WEIGHTS_DIR` or `BATTERY_NORMS_DIR`. Run `./scripts/setup_battery_artifacts.sh` to diagnose.
- **`list_batteries` returns an error** → CouchDB isn't seeded. Run `uv run python src/couchdb/init_battery.py --drop`.
- **"SpatialDropout1D got an unexpected keyword argument 'trainable'"** → Keras 3 can't deserialize these `.h5` files. The plan installs `tf_keras` (Keras 2 legacy) for this reason; confirm with `uv pip list | grep -i keras`.
- **Planner routes to TSFM instead of battery for RUL queries** → tune the first line of the battery tool docstrings in `main.py` to emphasize "lithium-ion" / "cell" / "RUL".
- **Long cold-start** → normal on first boot; `_boot()` runs inference for all 10 cells once (~20–30 s). Subsequent queries are O(1) cache lookups.

## File layout

```
src/servers/battery/
  __init__.py
  main.py                  FastMCP server, 9 tools, Pydantic result models
  preprocessing.py         NASA JSON cycle → (4, 500) tensor [Q, V, I, T]
  model_wrapper.py         lazy TF/Keras 2 model loader + inference cache
  benchmark_inference.py   Part B+C inference profiling (JSON/CSV under profiles/)
  couchdb_client.py        CouchDB fetch helpers
  diagnosis.py             LLM-narrated failure-mode classifier (3 few-shot)
  chemistries.yaml         Li-ion NCA 18650 thresholds
  tests/
    conftest.py
    test_preprocessing.py  (5 tests — run without backends)
    test_tools.py          (4 tests — need CouchDB + weights)
    test_benchmark_inference.py  profiling helpers / import smoke
    test_speedup_regression.py   precompute ms cap (optional weights)
    test_validation.py     (1 gate — B0018 MAE ≤ 30 cycles)
```

## References

- acctouhou/Prediction_of_battery — pretrained model source (Apache 2.0)
- NASA Prognostics Center of Excellence — raw cycling data (US Public Domain)
- `docs/speedup_baseline.md` — A/B benchmark workflow + success criteria
- `docs/inference_alt_runtimes.md` — SavedModel, TFLite, ONNX, XLA spikes
