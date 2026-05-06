# Battery profiling outputs

JSON results from the scripts under [`../profiling/`](../profiling/). Tracked in git.

Generate with:

```bash
uv run python -m servers.battery.profiling.benchmark_optimizations
uv run python -m servers.battery.profiling.mcp_batch_demo
uv run python -m servers.battery.profiling.profile_scenario --scenarios 1,2,4,5,6,7,8 \
    --model-id "cerebras/llama3.1-8b"
```

Layout:

- `benchmark_<ts>/` — one directory per `benchmark_optimizations` run, with one JSON per rung.
- `mcp_batch_demo_<ts>.json` — one file per `mcp_batch_demo` run.
- `scenarios_<ts>/` — one directory per `profile_scenario` run, with one JSON per scenario containing per-step wall times.
