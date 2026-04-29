# Speed-up A/B baseline (todo: baseline-json, after-each-change)

Before and after each inference optimization, record a comparable artifact:

```bash
uv sync --group battery
uv run benchmark-battery-inference --label speedup_baseline_v0 --repeats 3 \
  --notes "Machine ID; commit SHA; default env"
```

- Store the JSON path (e.g. under `profiles/`, gitignored) in your PR or lab notebook.
- For production-shaped tensors, export an NPZ (see [profiles/README.md](../profiles/README.md)) and add `--windows-npz ...`.

**After each merged change**, re-run with the same label prefix and a version suffix, e.g. `speedup_baseline_v0_batch256`, and compare:

```bash
python scripts/compare_battery_profiles.py profiles/speedup_baseline_v0_*.json profiles/speedup_baseline_v1_*.json
```

## Success criteria (todo: success-criteria)

A change is “clearly visible” if **either**:

- `combined_rul_then_volt_batched` or batched head `wall_ms.mean` drops by **≥15%** vs baseline (synthetic benchmark), **or**
- `inference_ms_per_cycle` from `_CACHE` on a reference cell (e.g. **B0018**) drops by **≥10%** vs baseline under identical env.

Document the reference cell and baseline JSON filename in the PR.
