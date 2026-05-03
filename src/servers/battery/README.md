# Battery MCP Server

Lithium-ion battery analytics for AssetOpsBench: RUL, voltage curves, impedance growth, fleet outliers, LLM-narrated diagnosis. Wraps the pretrained acctouhou model (`.h5` / `.npy` artifacts, CPU-friendly, ~13 ms/cycle inference) for the 10 NASA B0xx cells with enough cycles (≥100 clean paired charge/discharge cycles); statistical tools work for all 34 cells.

## Environment variables (add to your `.env`)

| Variable | Default | What it's for |
|----------|---------|---------------|
| `BATTERY_DBNAME` | `battery` | CouchDB database name |
| `BATTERY_MODEL_WEIGHTS_DIR` | `src/servers/battery/artifacts/weights` | directory holding the 4 `.h5` files |
| `BATTERY_NORMS_DIR` | `src/servers/battery/artifacts/norms` | directory holding the 4 `.npy` normalization files |
| `BATTERY_DATA_DIR` | `external/battery/nasa` | flat directory of NASA `B*.json` files (raw input, not artifacts) |
| `BATTERY_CELL_SUBSET` | 14-cell prototyping subset | comma-separated asset IDs, or `all` |
| `BATTERY_MODEL_ID` | inherits `FMSR_MODEL_ID` | LLM used by `diagnose_battery` |
| `COUCHDB_URL` / `COUCHDB_USERNAME` / `COUCHDB_PASSWORD` | see `.env.public` | shared with other servers |
| `BATTERY_REBUILD_CACHE` | unset | if `1`, ignore disk cache at boot and recompute (then rewrite `.npz` files). See **Disk cache** below. |
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

## Optimizations attempted that did not help

For the report's negative-result section. Code that produced these is preserved so the experiments can be re-run:

- **TF SavedModel format** - exported the `.h5` weights via `tf.saved_model.save` and re-timed boot. SavedModel boot ran ~17 s vs ~14 s for the `.h5` baseline (slower). Kept `.h5`.
- **Within-process tensor batching** (`precompute_cells_fully_batched`) - collapses 4N -> 4 `model.predict()` calls. Measured 0.95×–1.14× on Apple Silicon CPU (essentially neutral). The script `profiling/benchmark_inference.py` reproduces this and writes JSON to `profiles/`.
- **int8 TFLite quantization** (`profiling/quantize_models.py`) - converts the four Keras models to int8 TFLite. Kept in the codebase as proof of attempt; numerical equivalence and latency on this CPU did not justify replacing the `.h5` path.
- **XLA / ONNX / Core ML** - never implemented; out of scope for v1.

**Concurrency:** prefer **batching** and TF thread envs over `ThreadPoolExecutor` around `predict` unless your benchmark proves a wall-time win ([src/servers/battery/profiles/README.md](src/servers/battery/profiles/README.md)).

### After each speed PR

Re-run `benchmark-battery-inference` with the same label prefix and note the JSON path in the PR. Compare to the prior baseline JSON with `python -m servers.battery.profiling.compare_profiles old.json new.json`.

## First-time setup

1. **Install deps:** `uv sync --group battery` (pulls `tensorflow`, `tf_keras`, `scikit-learn`).
2. **Download acctouhou artifacts** (Apache 2.0) - the 4 `.h5` weights + 4 `.npy` normalization files - from the Google Drive link in `.prediction_of_battery/Prediction_of_battery/README.md`. The zip contains two folders: `pretrained/` (the `.h5` files) and `dataset/` (the `.npy` files plus others we ignore).
3. **Download NASA B0xx cycling data** (US Public Domain) from the NASA Prognostics Center. Flatten all `B*.json` files into `external/battery/nasa/`.
4. **Verify:** `./scripts/setup_battery_artifacts.sh`.
5. **Seed CouchDB:** `./src/couchdb/couchdb_setup.sh` - or run directly: `uv run python src/couchdb/init_battery.py --drop`.
6. **Start the server:** `uv run battery-mcp-server`.

## What you can test

**Smoke tests via plan-execute:**

```bash
uv run plan-execute "What batteries are available at site MAIN?"
uv run plan-execute "Predict RUL for B0018"
uv run plan-execute "When will B0018 voltage drop below 2.8V?"
uv run plan-execute "Diagnose battery B0018"
```

**All 15 scenarios in `src/scenarios/local/battery_utterances.json`:**

```bash
for i in $(seq 0 14); do
  q=$(jq -r ".[$i].query" src/scenarios/local/battery_utterances.json)
  echo "=== Scenario $((i+1)) ==="
  uv run plan-execute --show-plan "$q"
done
```

Or use the `--scenarios` flag to run all of them in one shot:

```bash
uv run plan-execute --scenarios -o results.txt
```

**Pytest:** `uv run pytest src/servers/battery/tests/` (preprocessing tests always run; tool + validation tests skip without CouchDB/weights).

## Profiling

All profiling code lives under `src/servers/battery/profiling/`. JSON outputs go to `src/servers/battery/profiles/` (tracked in git).

**Inference microbenchmark** (RUL / voltage `predict` throughput, sequential vs batched, optional ThreadPoolExecutor, TF thread sweeps):

```bash
uv sync --group battery
uv run benchmark-battery-inference --label inference_baseline --repeats 3 --notes "CPU default threads"
```

**Boot ablation study** (4 sections - CouchDB I/O split, precompute strategy, in-memory cache, disk cache before/after):

```bash
uv run python -m servers.battery.profiling.ablation_boot
```

**End-to-end scenario timings** (real MCP tool call wall_s per `tool_events`):

```bash
uv run python -m servers.battery.profiling.profile_scenarios_detailed --scenarios 0
```

**Compare two profile JSONs** (default: flag `wall_ms` mean changes > 5%):

```bash
python -m servers.battery.profiling.compare_profiles \
  src/servers/battery/profiles/a.json \
  src/servers/battery/profiles/b.json
```

## Disk cache

`_boot()` precompute (preprocess + 4 TF predicts for all model-ready cells) is the dominant cost of every MCP cold-start. Because MCP stdio servers spawn a fresh subprocess per tool call, the in-memory `_CACHE` dict dies on exit and the next call re-pays the boot cost.

The disk cache (`src/servers/battery/artifacts/cache/<cell>.npz` + `<cell>.manifest.json`) persists the precomputed `rul_trajectory` and sliding windows across subprocess restarts. On boot, if the manifest's `n_charge_docs` / `n_discharge_docs` match the current CouchDB state, `_try_load_from_disk` populates `_CACHE` from the `.npz` and the 4 TF predicts are skipped entirely.

Footprint: ~7 KB per cell, ~70 KB for the 10-cell fleet. Gitignored under `src/servers/battery/artifacts/cache/`.

To invalidate: `rm -r src/servers/battery/artifacts/cache/` or set `BATTERY_REBUILD_CACHE=1`. The manifest also detects when CouchDB doc counts change for a cell.

Measured on Apple Silicon CPU, 10 cells (see `src/servers/battery/profiles/ablation_boot_*.json` -> `disk_cache`):

| | Wall ms | TF predict calls |
|---|---:|---:|
| Cold first boot (writes `.npz`) | 7042 | 3 |
| Warm second boot (loads `.npz`) | **4** | **0** |
| Speedup | **1771×** | |

## Tools (9)

| Tool | What it does | Uses model? |
|------|--------------|-------------|
| `list_batteries` | List available cells with `model_ready` flag | no |
| `get_battery_cycle_summary` | Per-cycle Capacity / max_T / avg_V / Rct / Re | no |
| `predict_rul` | Remaining cycles to 1.4 Ah (30% fade) | yes - `predictor.h5` |
| `predict_rul_batch` | RUL for many cached cells in one call | yes - cache |
| `predict_voltage_curve` | 100-point V-SOC curve for a cycle | yes - `predictor2.h5` |
| `predict_voltage_milestones` | Cycle where V first crosses a threshold | yes - `predictor2.h5` |
| `analyze_impedance_growth` | Rct exponential growth rate + alarm | no - `scipy.polyfit` |
| `detect_capacity_outliers` | Fleet z-score over capacity fade rate | no - numpy |
| `diagnose_battery` | Combined RUL + impedance + outlier with LLM narration | yes + LLM |

Abstract scenarios like *"predict RUL for test cells and list top 12 at risk"* are orchestrated by the plan-execute agent: it calls `list_batteries` -> `predict_rul` or `predict_rul_batch` -> sorts -> returns top N. Tools stay single-cell primitives on purpose except for the optional batch RUL helper.

## Example use cases from `src/scenarios/local/battery_utterances.json`

1. **Commercial EV Fleet Operator** - "Predict RUL until 30% fade; list top 12 at risk" -> `list_batteries` -> loop `predict_rul` -> LLM ranks
2. **BMS Hardware Engineer** - "Predict EOD time; show voltage drop curves" -> `predict_voltage_milestones` + `predict_voltage_curve`
3. **QA Lead** - "Flag cells degrading faster than standard within 50 cycles" -> `detect_capacity_outliers(window=50)`
4. **Microgrid Operator** - "Flag cells with exponential Rct spikes" -> `analyze_impedance_growth` per cell
5. **Fleet Safety Officer** - "Analyze EIS for electrolyte-resistance trajectory" -> `analyze_impedance_growth`
6. **Full diagnosis** - "What's wrong with B0018?" -> `diagnose_battery`

## Validation status

The acctouhou model was trained on the Severson LFP dataset; NASA B0xx cells are NCA 18650. The pretrained pipeline runs end-to-end on NASA data, but **expect significant RUL error without fine-tuning** - on B0005 the predicted RUL at cycle 100 is ~3900 cycles vs ground truth ~24 cycles. This is a chemistry/distribution mismatch, not a pipeline bug.

**Tested and confirmed working:**

- Preprocessing pipeline produces the expected `(4, 500)` tensors per cycle.
- Summary features are 6-dim as required by `summary_norm.npy`.
- Inference speed is ~13 ms/cycle on CPU (well under scenario 5's target).
- All 5 preprocessing unit tests pass.

**Not fine-tuned:** if accurate NASA RUL matters, the next step is LoRA-style fine-tuning on B0005/B0006/B0007 and holding out B0018 for validation (out of scope for v1).

**Statistical tools (`analyze_impedance_growth`, `detect_capacity_outliers`, `get_battery_cycle_summary`, `list_batteries`)** don't have this issue - they operate directly on NASA measurements and are unaffected by the chemistry mismatch.

## Troubleshooting

- **Server boots but `predict_rul` returns "Pretrained model unavailable"** -> weights aren't in `BATTERY_MODEL_WEIGHTS_DIR` or `BATTERY_NORMS_DIR`. Run `./scripts/setup_battery_artifacts.sh` to diagnose.
- **`list_batteries` returns an error** -> CouchDB isn't seeded. Run `uv run python src/couchdb/init_battery.py --drop`.
- **"SpatialDropout1D got an unexpected keyword argument 'trainable'"** -> Keras 3 can't deserialize these `.h5` files. The plan installs `tf_keras` (Keras 2 legacy) for this reason; confirm with `uv pip list | grep -i keras`.
- **Planner routes to TSFM instead of battery for RUL queries** -> tune the first line of the battery tool docstrings in `main.py` to emphasize "lithium-ion" / "cell" / "RUL".
- **Long cold-start** -> normal on first boot; `_boot()` runs inference for all 10 cells once (~20–30 s). Subsequent queries are O(1) cache lookups.

## File layout

```
src/servers/battery/
  __init__.py
  main.py                  FastMCP server, 9 tools, Pydantic result models
  preprocessing.py         NASA JSON cycle -> (4, 500) tensor [Q, V, I, T]
  model_wrapper.py         lazy TF/Keras 2 model loader + in-memory + disk cache
  couchdb_client.py        CouchDB fetch helpers
  diagnosis.py             LLM-narrated failure-mode classifier (3 few-shot)
  chemistries.yaml         Li-ion NCA 18650 thresholds
  artifacts/               (gitignored - heavy/local)
    weights/               4 × .h5 pretrained Keras weights
    norms/                 4 × .npy normalization tensors
    saved_models/          TF SavedModel exports (optional)
    quantized/             int8 TFLite exports (optional)
    cache/                 .npz disk cache populated by _boot() (~7 KB/cell)
  profiles/                profiling JSON outputs (tracked in git)
    README.md              what this dir holds + how to regenerate
    ablation_boot_*.json   boot ablation: data load, precompute strategy, in/disk cache
    ablation_post_refactor_*.json   TF tensor batching microbenchmark
    scenario_detailed_*/   end-to-end MCP wall_s per tool call
  profiling/
    __init__.py
    ablation_boot.py                4-section boot ablation (the headline study)
    benchmark_inference.py          inference microbenchmark - TF tensor batching
                                    (entry: benchmark-battery-inference)
    profile_scenarios_detailed.py   end-to-end planner/executor timing per tool call
    compare_profiles.py             diff two profile JSONs side-by-side
    quantize_models.py              int8 TFLite quantization (kept as proof of attempt)
  tests/
    conftest.py
    test_preprocessing.py        (5 tests - run without backends)
    test_tools.py                (6 tests - need CouchDB + weights)
    test_benchmark_inference.py  profiling helpers / import smoke
    test_speedup_regression.py   precompute ms cap (optional weights)
    test_validation.py           (1 gate - B0018 MAE ≤ 30 cycles, documents LFP->NCA mismatch)
```

## References

- acctouhou/Prediction_of_battery - pretrained model source (Apache 2.0)
- NASA Prognostics Center of Excellence - raw cycling data (US Public Domain)
- `src/servers/battery/profiles/README.md` - how to regenerate the ablation JSONs
