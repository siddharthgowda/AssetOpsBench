# Battery profiling outputs

JSON results written by the scripts under [`../profiling/`](../profiling/). Tracked in git.

Generate with:

```bash
uv run benchmark-battery-inference --label my_run --notes "what changed"
uv run python -m servers.battery.profiling.ablation_boot
uv run python -m servers.battery.profiling.profile_scenarios_detailed --scenarios 0
```
