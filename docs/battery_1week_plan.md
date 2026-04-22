# Battery MCP Server — 1-Week Plan

**Goal:** Everything in the 1-day plan, plus the hardening/coverage work that turns a working prototype into a durable project asset.

**Assumes:** `docs/battery_1day_plan.md` is complete (8 tools, CouchDB-backed, 15 scenarios passing, pytest green, README).

This plan does NOT rewrite anything from the 1-day plan — it adds on top. Days 2–7 below are extensions.

---

## Day 1 — Execute the 1-day plan (12h)

Per `docs/battery_1day_plan.md`. End state:
- 8 working tools
- All 15 scenarios pass end-to-end
- pytest green
- B0018 validation documented (pass or fallback)
- `battery-server-v1` tagged

Everything below assumes this is done.

---

## Day 2 — Expand to all 34 cells where possible (~4h)

Currently only the 10 "usable" cells (≥100 clean discharge cycles) feed the pretrained model. The other 24 cells are still valuable for statistical tools.

### Extend non-model tools to all 34 cells

**`analyze_impedance_growth`** and **`detect_capacity_outliers`** don't need the 100-cycle window. Update their code to fetch any cell from CouchDB:

```python
@mcp.tool()
def analyze_impedance_growth(asset_id: str) -> ImpedanceResult:
    """Works for any cell with ≥3 impedance cycles."""
    cycles = client.fetch_cycles(asset_id, cycle_type='impedance')
    if len(cycles) < 3:
        return ErrorResult(error=f"{asset_id}: only {len(cycles)} impedance cycles")
    # Curve fit on Rct vs cycle_index ...
```

**`get_battery_cycle_summary`** already works for any cell (reads CouchDB directly).

**`list_batteries`** returns a list of dicts with a `model_ready: bool` flag indicating whether pretrained-model tools will work.

### Update expected planner behavior

If the user asks about a short-cycle cell (e.g. B0029), `predict_rul` should return a clear error message pointing to `analyze_impedance_growth` or `detect_capacity_outliers` as alternatives — not crash.

Time: **~4 hours** (mostly testing edge cases per cell).

---

## Day 3 — Scenario 13 fallback + TSFM integration (~5h)

Scenario 13 (RUL from first 25% of cycles) structurally fails the pretrained model because 25% of any cell is <50 cycles, well below the 100-cycle window.

### Route scenario 13 to TSFM with `ttm_52_16_mae`

Add a `method` parameter to `predict_rul`:

```python
@mcp.tool()
def predict_rul(
    asset_id: str,
    from_cycle: int = 100,
    method: str = "auto",
) -> RULResult:
    """RUL prediction. method: 'pretrained' (default), 'tsfm' (for early-cycle), or 'auto'."""
    if method == "tsfm" or (method == "auto" and from_cycle < 100):
        return _predict_rul_via_tsfm(asset_id, from_cycle)
    ...
```

`_predict_rul_via_tsfm` builds a per-cycle capacity-summary DataFrame on the fly, calls the existing TSFM server's `run_tsfm_forecasting` with `model_checkpoint="ttm_52_16_mae"`, and post-processes the forecast to find when Capacity crosses 1.4 Ah.

### Verification

```bash
uv run plan-execute "Predict RUL for B0018 using only the first 30 cycles"
# Expect: predict_rul(B0018, from_cycle=30) → routes via TSFM fallback, returns a number
```

Time: **~5 hours** (TSFM integration has its own data-format quirks).

---

## Day 4 — Docker + CI/CD (~5h)

### Dockerfile update

`src/couchdb/docker-compose.yaml` already references `BATTERY_DBNAME`. Extend the Dockerfile to mount `.pretrained/` and `.data/5. Battery Data Set_json/` so the container can run the server.

```yaml
services:
  couchdb:
    environment:
      BATTERY_DBNAME: battery
      BATTERY_DATA_DIR: /sample_data/battery_nasa
    volumes:
      - ../../.data/5. Battery Data Set_json:/sample_data/battery_nasa:ro
```

And a separate `battery-server` service that runs the MCP server with lazy TF imports.

### GitHub Actions workflow

`.github/workflows/battery-tests.yml`:

```yaml
name: battery-server-tests
on: [push, pull_request]
jobs:
  test:
    steps:
      - uses: actions/checkout@v4
      - run: uv sync --group battery
      - run: uv run pytest src/servers/battery/tests/ -m "not requires_couchdb"
```

(The `requires_couchdb` tests need an integration environment; keep those manual for now.)

Time: **~5 hours**.

---

## Day 5 — Fleet-level tools + performance tuning (~5h)

### New tool: `fleet_rul_ranking(asset_ids=None)`

Batch-processes the RUL for all 10 cells and returns sorted top-N. Avoids 10 separate `predict_rul` calls by looking them all up from `_CACHE`.

```python
@mcp.tool()
def fleet_rul_ranking(asset_ids: list[str] | None = None, top_n: int = 12) -> FleetRankingResult:
    """Rank cells by predicted RUL; return the top N most at risk (lowest RUL)."""
    cells = asset_ids or list(_CACHE.keys())
    rankings = []
    for cell in cells:
        entry = _CACHE.get(cell)
        if entry:
            rankings.append({"asset_id": cell, "rul": float(entry['rul_trajectory'][-1])})
    rankings.sort(key=lambda r: r['rul'])
    return FleetRankingResult(rankings=rankings[:top_n])
```

Unblocks scenarios 1 and 7 without needing the planner to loop.

### Startup precompute in parallel

Currently `_boot()` loops sequentially over 10 cells. With `concurrent.futures`, speed it up by running preprocessing in threads and inference in a single batched call.

### Memory footprint profiling

Log `_CACHE` size at boot. Should be <50 MB total.

Time: **~5 hours**.

---

## Day 6 — Observability + error paths (~4h)

### Tracing

Add `logging.getLogger("battery-mcp-server")` calls to each tool entry/exit. Log:
- Tool invoked, args
- Cache hit/miss
- Inference duration if applicable
- Errors with full traceback

### Graceful degradation for every failure mode

| Failure | Behavior |
|---------|----------|
| Weights dir missing | Model tools return `ErrorResult` with setup hint; statistical tools still work |
| CouchDB down | All tools return `ErrorResult` pointing at `couchdb_setup.sh` |
| Cell has <100 cycles + user calls `predict_rul` | Route to statistical fallback or TSFM (see Day 3) |
| Model inference exception | Log + return `ErrorResult`; don't crash the server |
| LLM unavailable in `diagnose_battery` | Return structured numerical findings without narration |

Time: **~4 hours**.

---

## Day 7 — Docs polish + handoff (~4h)

### Expand `src/servers/battery/README.md`

- Add architecture diagram (Mermaid or ASCII)
- Document each tool's inputs/outputs with examples
- Full troubleshooting section
- Link to `docs/battery_server_plan.md` for design rationale

### Close out `docs/battery_server_plan.md`

- Mark all resolved open questions
- Add final per-cell accuracy table
- Remove speculative sections ("we don't know X" → actual answers)
- Tag the doc "v1 — current as of {date}"

### Write `docs/battery_server_changelog.md`

- Initial v1 release notes
- Known limitations
- Planned v2 features (PBT, fine-tuning, multi-chemistry)

### Final review

- Walk through the repo with fresh eyes for leaks (forgotten debug prints, dead code, commented-out lines)
- Run full test suite one more time
- `git tag battery-server-v1.0`

Time: **~4 hours**.

---

## Total time budget

| Phase | Time |
|-------|------|
| Day 1 — 1-day plan (docs/battery_1day_plan.md) | 12h |
| Day 2 — 34-cell expansion | 4h |
| Day 3 — Scenario 13 TSFM fallback | 5h |
| Day 4 — Docker + CI/CD | 5h |
| Day 5 — Fleet tools + perf tuning | 5h |
| Day 6 — Observability + error paths | 4h |
| Day 7 — Docs polish + handoff | 4h |
| **Total** | **~39h** |

Realistic calendar time for one engineer: 5–7 working days depending on cadence.

---

## Acceptance at end of 1 week

**Everything from the 1-day plan, plus:**

- [ ] All 34 NASA cells usable for statistical tools (`analyze_impedance_growth`, `detect_capacity_outliers`, `get_battery_cycle_summary`, `list_batteries`)
- [ ] Scenario 13 works via TSFM fallback with `ttm_52_16_mae`
- [ ] Docker container boots with battery server and reaches CouchDB
- [ ] GitHub Actions runs battery-server tests on every push
- [ ] `fleet_rul_ranking` returns top-12 in <2 seconds
- [ ] Memory footprint <50 MB at steady state
- [ ] Every error path returns a structured `ErrorResult`, not a traceback
- [ ] README has architecture diagram + full troubleshooting
- [ ] `docs/battery_server_changelog.md` documents v1 release

---

## What's explicitly NOT in the 1-week plan

- **PBT integration** — alternative model documented in `docs/battery_server_plan.md`; revisit only if acctouhou accuracy is insufficient for real use
- **Fine-tuning** acctouhou or PBT on NASA — out of scope, requires its own week
- **Multi-chemistry support** — `chemistries.yaml` is extensible but v1 ships only `li_ion_nca_18650`
- **Production observability** — metrics/tracing beyond structured logging
- **Frontend / visualization tools** — tool outputs are JSON; downstream UI is a separate project
- **Paper/benchmark writeup** — scenario results are logged but not formatted for publication

---

## Risk register (new risks beyond the 1-day plan)

| Risk | Mitigation |
|------|------------|
| TSFM fallback for scenario 13 has its own accuracy issues | Day 3 includes a sanity check vs known EOL; document limits in README |
| Docker image size balloons with TF | Multi-stage build; only copy runtime deps into final image |
| CI/CD times out on model loading | Use a mock mode for unit tests; only integration tests load real weights |
| Fleet ranking scales poorly beyond 100 cells | Current cap is 34; document the assumption; add pagination if needed in v2 |
