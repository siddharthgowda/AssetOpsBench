# AssetOpsBench (school-project fork)

> This is a fork of [AssetOpsBench](https://github.com/IBM/AssetOpsBench) maintained as a Columbia COMS6998 school project. The original project README is preserved as [`OLD_README.md`](OLD_README.md). Everything below describes what was added or changed in this fork - concentrated on the new **battery MCP server**.

---

## What we added

- A new MCP server, `battery`, that joins the existing `iot` / `utilities` / `fmsr` / `tsfm` / `wo` / `vibration` servers and serves Li-ion battery analytics (RUL, voltage curves, impedance, fleet outliers, LLM-narrated diagnosis).
- Profiling + ablation tooling that quantifies where the server's startup time goes and how the optimizations bought it back. The headline study and per-tool measurements live in [`src/servers/battery/profiles/ablation_study/`](src/servers/battery/profiles/ablation_study/).

For the server-level deep dive (env vars, disk cache mechanics, troubleshooting, full file layout), see [`src/servers/battery/README.md`](src/servers/battery/README.md).

---

## The model we chose and why

**Model**: the **acctouhou** pretrained Li-ion RUL + voltage model (Apache 2.0 - sourced from [acctouhou/Prediction_of_battery](https://github.com/acctouhou/Prediction_of_battery)).

What it ships as: four Keras-2 `.h5` weight files (~150 MB total) plus four `.npy` normalization tensors. Two feature-selector models for charge/discharge, one RUL head, one voltage head.

**Why this model:**
- **Lightweight** - runs on CPU at ~7 ms per cycle of inference. No GPU required.
- **Pretrained** - no training pipeline, no labeled-data infrastructure, no GPU cluster needed.
- **Two heads in one package** - RUL trajectory *and* voltage curves, which together cover most of the scenario set.
- **Permissive license** (Apache 2.0) - drop-in for a school project.
- Modest disk footprint (~150 MB weights + 70 KB cache after warmup).

---

## How the server runs the model

The acctouhou pipeline runs once at server boot to fill an in-memory cache. Every subsequent MCP tool call is a sub-microsecond dict lookup against that cache.

The boot pipeline, per cell:

1. **Fetch** the cell's raw cycle docs from CouchDB via `couchdb_client.fetch_cycles`.
2. **Preprocess each cycle** (in `preprocessing.py`):
   - Extract `Voltage_measured`, `Current_measured`, `Temperature_measured`, `Time` arrays from the JSON.
   - **Coulomb counting**: derive Q via `scipy.cumulative_trapezoid(|I|, t) / 3600`.
   - **Interpolate** all four channels (Q, V, I, T) to **500 uniform timesteps** with `scipy.interp1d`.
   - Stack into `(n_cycles, 4, 500)` tensors for charge and discharge.
   - Build a `(n_cycles, 6)` summary feature vector `[Qd, Qc, Tavg, Tmin, Tmax, chargetime]`.
3. **Normalize** the tensors with the four `.npy` norm files.
4. **Run feature selectors**: `feature_selector_ch.h5` and `feature_selector_dis.h5` produce per-cycle features.
5. **Build sliding windows** of 50 cycles × 12 features (numpy).
6. **Run the RUL head** (`predictor.h5`) -> trajectory of remaining cycles per cell.
7. **Run the voltage head** (`predictor2.h5`, lazy by default) -> V-SOC curves per cell.
8. **Cache** the outputs: `rul_trajectory`, `voltage_curves`, and the windows tensor go into both an in-memory `_CACHE` dict and a `.npz` file under `src/servers/battery/artifacts/cache/`. The disk file lets a fresh server subprocess skip the entire pipeline on the next boot.

This is the cost we pay once per boot. Tool calls then read from the cache.

---

## How tool calls work

The server uses **FastMCP** with stdio transport. There is no daemon - the agent spawns a fresh `battery-mcp-server` subprocess per tool call, the call runs, the subprocess exits.

The 9 tools, grouped by what they do:

| Group | Tools | Notes |
|---|---|---|
| Discovery / inspection (no model) | `list_batteries`, `get_battery_cycle_summary` | Reads CouchDB only. |
| Single-cell prediction | `predict_rul`, `predict_voltage_curve`, `predict_voltage_milestones` | Cache lookups. |
| **Batch / fleet** | **`predict_rul_batch`** | Accepts a list of cell IDs; returns N results in one MCP RPC. The key tool for amortizing cold-start cost across a 10-cell question. |
| Statistical (no model) | `analyze_impedance_growth`, `detect_capacity_outliers` | scipy / numpy on raw CouchDB measurements. |
| LLM-narrated | `diagnose_battery` | Combines the above with a few-shot LLM call. |

Why the batch tool matters for performance: each MCP RPC pays the full subprocess cold-start (~7 s with the disk cache, ~15 s without). One batched call serves the whole fleet for the price of one cold-start; ten single-cell calls would pay ten cold-starts.

---

## Repo layout (battery-relevant)

```
src/servers/battery/
  README.md                    full server doc (env vars, disk cache, troubleshooting)
  scenarios_report.md          run report from the 15 scenarios
  main.py                      FastMCP server, 9 tools
  model_wrapper.py             TF/Keras loader + in-memory + disk cache
  preprocessing.py             NASA JSON cycle -> (4, 500) tensor [Q, V, I, T]
  couchdb_client.py            CouchDB fetch helpers
  diagnosis.py                 LLM-narrated failure-mode classifier
  artifacts/                   weights + norms + .npz disk cache (gitignored)
  profiles/ablation_study/     the 4 JSONs the report draws from
  profiling/                   ablation_boot, benchmark_inference, scenario profiler, …
  tests/

src/scenarios/local/battery_utterances.json   15 evaluation scenarios

external/battery/nasa/                        raw NASA B0xx cycle data (gitignored)
```

---

## Where to go next

| For… | Read |
|---|---|
| Per-tool details, env vars, disk cache mechanics | [`src/servers/battery/README.md`](src/servers/battery/README.md) |
| Optimization / ablation measurements | [`src/servers/battery/profiles/ablation_study/`](src/servers/battery/profiles/ablation_study/) |
| Run report across all 15 scenarios | [`src/servers/battery/scenarios_report.md`](src/servers/battery/scenarios_report.md) |
| Upstream AssetOpsBench framework (agent, planner, other servers) | [`OLD_README.md`](OLD_README.md) |
