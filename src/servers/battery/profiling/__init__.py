"""Battery MCP Server — profiling package.

Run via:
    uv run python -m servers.battery.profiling.run_profile --label baseline
    uv run python -m servers.battery.profiling.run_profile --label optimized
    uv run python -m servers.battery.profiling.compare profiles/baseline_*.json profiles/optimized_*.json

See instructions_profiling.md for the full guide.
"""
