"""Profiling and benchmarking tooling for the battery server.

Modules
-------
ablation_boot                4-section boot ablation (data load, precompute strategy,
                             in-memory cache, disk cache before/after) — the headline study.
benchmark_inference          TF/Keras predict timing (sequential vs batched, sweeps).
                             Entry point: ``benchmark-battery-inference``.
profile_scenarios_detailed   End-to-end planner/executor timing across scenarios with
                             per-tool-call wall_s in ``tool_events``.
compare_profiles             Diff two profile JSONs side-by-side.
quantize_models              Export int8 TFLite versions of the .h5 weights.
"""
