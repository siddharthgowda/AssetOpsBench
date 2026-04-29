"""Benchmark battery TF/Keras inference (Part B + C of profiling plan).

Measures sequential vs batched ``predict`` for RUL/voltage heads, optional
feature-selector path mirroring ``precompute_cell``, ThreadPoolExecutor sweeps,
and optional TF thread / batch-size sweeps (see CLI).

**Wall vs CPU time:** ``wall_ms`` is ``time.perf_counter()`` (what users wait for).
``cpu_process_ms`` is ``time.process_time()`` (can *rise* with threads while wall
drops — not a contradiction). Compare thread modes using both.

Requires dependency group ``battery`` (TensorFlow + tf_keras) and acctouhou
weights under ``BATTERY_MODEL_WEIGHTS_DIR`` / ``BATTERY_NORMS_DIR``.

Example::

    uv run benchmark-battery-inference --label inference_baseline \\
        --n-batteries 4 --repeats 3 --notes "M1 default threads"

**ProcessPoolExecutor** is intentionally not implemented: multiprocessing + TF
implies per-process model load and fork/spawn hazards; see ``--experimental-process-pool``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Repo root: .../src/servers/battery/benchmark_inference.py → … → repo
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _try_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _try_ram_gb() -> int | None:
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) // (1024**3)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return None
    return None


def _parse_csv_ints(s: str | None) -> list[int] | None:
    if not s or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _stats_seconds(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(sum(samples) / len(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "n": int(len(samples)),
    }


def _wall_cpu_ms(
    fn: Callable[[], None], repeats: int
) -> dict[str, Any]:
    wall: list[float] = []
    cpu: list[float] = []
    for _ in range(repeats):
        w0 = time.perf_counter()
        c0 = time.process_time()
        fn()
        wall.append(time.perf_counter() - w0)
        cpu.append(time.process_time() - c0)
    wall_ms = {k: v * 1000.0 if k != "n" else v for k, v in _stats_seconds(wall).items()}
    cpu_ms = {k: v * 1000.0 if k != "n" else v for k, v in _stats_seconds(cpu).items()}
    return {"wall_ms": wall_ms, "cpu_process_ms": cpu_ms}


def _merge_measurement(blob: dict[str, Any], n_batteries: int) -> dict[str, Any]:
    """Attach wall_ms_per_battery (mean/min/max / n_batteries) for interpretation."""
    out = dict(blob)
    wm = blob.get("wall_ms")
    if isinstance(wm, dict) and "mean" in wm and n_batteries > 0:
        out["wall_ms_per_battery"] = {
            "mean": float(wm["mean"]) / n_batteries,
            "min": float(wm["min"]) / n_batteries,
            "max": float(wm["max"]) / n_batteries,
            "n": wm["n"],
        }
    return out


def _alias_model_predict(measurements: dict[str, Any], batch_sizes: list[int]) -> None:
    """Duplicate keys using plan vocabulary (``model_predict_*``)."""
    rul_b = "rul_predict_batched" if len(batch_sizes) == 1 else f"rul_predict_batched_bs{batch_sizes[0]}"
    volt_b = "volt_predict_batched" if len(batch_sizes) == 1 else f"volt_predict_batched_bs{batch_sizes[0]}"
    mapping = {
        "rul_predict_sequential": "model_predict_rul_sequential_ms",
        rul_b: "model_predict_rul_batched_ms",
        "volt_predict_sequential": "model_predict_volt_sequential_ms",
        volt_b: "model_predict_volt_batched_ms",
    }
    for old, new in mapping.items():
        if old in measurements:
            measurements[new] = measurements[old]


def _load_npz_windows(
    path: Path, n_batteries: int, n_cycles: int
) -> tuple[list[np.ndarray], list[np.ndarray] | None, list[np.ndarray] | None, list[np.ndarray] | None]:
    data = np.load(path, allow_pickle=True)
    ch_list: list[np.ndarray] | None = None
    dis_list: list[np.ndarray] | None = None
    summ_list: list[np.ndarray] | None = None

    if "windows" in data.files:
        w0 = np.asarray(data["windows"], dtype=np.float32)
        if w0.ndim == 2:
            if w0.shape != (n_cycles, 50, 12):
                w0 = w0[:n_cycles]
                pad_cycles = n_cycles - w0.shape[0]
                if pad_cycles > 0:
                    w0 = np.pad(
                        w0, ((0, pad_cycles), (0, 0), (0, 0)), mode="edge"
                    )  # (n_cycles, 50, 12)
            windows_list = [w0.copy() for _ in range(n_batteries)]
        elif w0.ndim == 3 and w0.shape[0] == n_batteries:
            windows_list = [np.asarray(w0[i], dtype=np.float32) for i in range(n_batteries)]
        else:
            raise ValueError(f"windows array shape {w0.shape} incompatible with --n-batteries {n_batteries}")
    else:
        raise ValueError("NPZ must contain 'windows' (n_cycles, 50, 12) or (n_batteries, n_cycles, 50, 12)")

    if all(k in data.files for k in ("charges", "discharges", "summary")):
        ch0 = np.asarray(data["charges"], dtype=np.float32)
        d0 = np.asarray(data["discharges"], dtype=np.float32)
        s0 = np.asarray(data["summary"], dtype=np.float32)
        ch_list = [ch0.copy() for _ in range(n_batteries)]
        dis_list = [d0.copy() for _ in range(n_batteries)]
        summ_list = [s0.copy() for _ in range(n_batteries)]

    return windows_list, ch_list, dis_list, summ_list


def _synthetic_raw(n_cycles: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    charges = rng.standard_normal((n_cycles, 4, 500), dtype=np.float64).astype(np.float32)
    discharges = rng.standard_normal((n_cycles, 4, 500), dtype=np.float64).astype(np.float32)
    summary = rng.standard_normal((n_cycles, 6), dtype=np.float64).astype(np.float32)
    return charges, discharges, summary


def _windows_from_cell_feat(cell_feat: np.ndarray) -> np.ndarray:
    from servers.battery.model_wrapper import _pad_edge

    return np.stack(
        [_pad_edge(cell_feat[max(0, k - 49) : k + 1], 50) for k in range(len(cell_feat))]
    )


@dataclass
class _BenchCtx:
    rul: Any
    volt: Any
    windows_list: list[np.ndarray]
    charges_list: list[np.ndarray] | None
    discharges_list: list[np.ndarray] | None
    summary_list: list[np.ndarray] | None
    n_batteries: int
    warmup: int


def _infer_feature_lists(ctx: _BenchCtx, models: dict, norms: dict) -> None:
    from servers.battery.model_wrapper import concat_data, feature_selector

    assert ctx.charges_list and ctx.discharges_list and ctx.summary_list
    for i in range(ctx.n_batteries):
        ch_feat = feature_selector(models["fs_ch"], ctx.charges_list[i], norms["charge"])
        dis_feat = feature_selector(models["fs_dis"], ctx.discharges_list[i], norms["discharge"])
        cf = concat_data(ch_feat, dis_feat, ctx.summary_list[i], norms["summary"])
        ctx.windows_list[i] = _windows_from_cell_feat(cf)


def _run_heads_benchmark(
    ctx: _BenchCtx,
    batch_sizes: list[int],
    repeats: int,
    thread_workers: list[int],
    skip_thread_pool: bool,
    include_fs: bool,
    fs_batched_multicell: bool,
    models: dict,
    norms: dict,
) -> dict[str, Any]:
    measurements: dict[str, Any] = {}
    rul, volt = ctx.rul, ctx.volt
    wl = ctx.windows_list

    def default_bs() -> int:
        return batch_sizes[0]

    # --- Optional: feature selector timings (rebuilds windows; mutates ctx.windows_list) ---
    if include_fs and ctx.charges_list:
        from servers.battery.model_wrapper import concat_data, feature_selector

        if fs_batched_multicell and ctx.n_batteries > 1:

            def run_fs_seq() -> None:
                for i in range(ctx.n_batteries):
                    ch_feat = feature_selector(models["fs_ch"], ctx.charges_list[i], norms["charge"])
                    dis_feat = feature_selector(models["fs_dis"], ctx.discharges_list[i], norms["discharge"])
                    cf = concat_data(ch_feat, dis_feat, ctx.summary_list[i], norms["summary"])
                    ctx.windows_list[i] = _windows_from_cell_feat(cf)

            def run_fs_batched() -> None:
                big_ch = np.concatenate(ctx.charges_list, axis=0)
                big_dis = np.concatenate(ctx.discharges_list, axis=0)
                big_s = np.concatenate(ctx.summary_list, axis=0)
                ch_all = feature_selector(models["fs_ch"], big_ch, norms["charge"])
                dis_all = feature_selector(models["fs_dis"], big_dis, norms["discharge"])
                cf_all = concat_data(ch_all, dis_all, big_s, norms["summary"])
                n_c = ctx.charges_list[0].shape[0]
                for bi in range(ctx.n_batteries):
                    sl = slice(bi * n_c, (bi + 1) * n_c)
                    ctx.windows_list[bi] = _windows_from_cell_feat(cf_all[sl])

            for _ in range(ctx.warmup):
                run_fs_seq()
            measurements["feature_selectors_sequential_per_cell"] = _merge_measurement(
                _wall_cpu_ms(run_fs_seq, repeats), ctx.n_batteries
            )
            for _ in range(ctx.warmup):
                run_fs_batched()
            measurements["feature_selectors_batched_multicell"] = _merge_measurement(
                _wall_cpu_ms(run_fs_batched, repeats), ctx.n_batteries
            )
        else:

            def run_fs_only_seq() -> None:
                for i in range(ctx.n_batteries):
                    ch_feat = feature_selector(models["fs_ch"], ctx.charges_list[i], norms["charge"])
                    dis_feat = feature_selector(models["fs_dis"], ctx.discharges_list[i], norms["discharge"])
                    cf = concat_data(ch_feat, dis_feat, ctx.summary_list[i], norms["summary"])
                    ctx.windows_list[i] = _windows_from_cell_feat(cf)

            for _ in range(ctx.warmup):
                run_fs_only_seq()
            measurements["feature_selectors_sequential_per_cell"] = _merge_measurement(
                _wall_cpu_ms(run_fs_only_seq, repeats), ctx.n_batteries
            )

    # --- RUL / VOLT ---
    def warm_rul_seq(bs: int) -> None:
        for _ in range(ctx.warmup):
            for w in wl:
                rul.predict(w, batch_size=bs, verbose=0)

    def warm_volt_seq(bs: int) -> None:
        for _ in range(ctx.warmup):
            for w in wl:
                s = np.full((w.shape[0], 1), 0.5, dtype=np.float32)
                volt.predict([w, s], batch_size=bs, verbose=0)

    def run_rul_sequential(bs: int) -> None:
        for w in wl:
            rul.predict(w, batch_size=bs, verbose=0)

    def run_rul_batched(bs: int) -> None:
        big = np.concatenate(wl, axis=0)
        rul.predict(big, batch_size=bs, verbose=0)

    def run_volt_sequential(bs: int) -> None:
        for w in wl:
            s = np.full((w.shape[0], 1), 0.5, dtype=np.float32)
            volt.predict([w, s], batch_size=bs, verbose=0)

    def run_volt_batched(bs: int) -> None:
        big_w = np.concatenate(wl, axis=0)
        big_s = np.full((big_w.shape[0], 1), 0.5, dtype=np.float32)
        volt.predict([big_w, big_s], batch_size=bs, verbose=0)

    def run_rul_then_volt_sequential(bs: int) -> None:
        for w in wl:
            rul.predict(w, batch_size=bs, verbose=0)
        for w in wl:
            s = np.full((w.shape[0], 1), 0.5, dtype=np.float32)
            volt.predict([w, s], batch_size=bs, verbose=0)

    def run_rul_then_volt_batched(bs: int) -> None:
        big_w = np.concatenate(wl, axis=0)
        rul.predict(big_w, batch_size=bs, verbose=0)
        big_s = np.full((big_w.shape[0], 1), 0.5, dtype=np.float32)
        volt.predict([big_w, big_s], batch_size=bs, verbose=0)

    bs0 = default_bs()
    warm_rul_seq(bs0)
    measurements["rul_predict_sequential"] = _merge_measurement(
        _wall_cpu_ms(lambda: run_rul_sequential(bs0), repeats), ctx.n_batteries
    )

    for bs in batch_sizes:
        warm_rul_seq(bs)
        key = "rul_predict_batched" if len(batch_sizes) == 1 else f"rul_predict_batched_bs{bs}"
        measurements[key] = _merge_measurement(
            _wall_cpu_ms(lambda b=bs: run_rul_batched(b), repeats), ctx.n_batteries
        )

    warm_volt_seq(bs0)
    measurements["volt_predict_sequential"] = _merge_measurement(
        _wall_cpu_ms(lambda: run_volt_sequential(bs0), repeats), ctx.n_batteries
    )

    for bs in batch_sizes:
        warm_volt_seq(bs)
        key = "volt_predict_batched" if len(batch_sizes) == 1 else f"volt_predict_batched_bs{bs}"
        measurements[key] = _merge_measurement(
            _wall_cpu_ms(lambda b=bs: run_volt_batched(b), repeats), ctx.n_batteries
        )

    warm_rul_seq(bs0)
    warm_volt_seq(bs0)
    measurements["combined_rul_then_volt_sequential"] = _merge_measurement(
        _wall_cpu_ms(lambda: run_rul_then_volt_sequential(bs0), repeats), ctx.n_batteries
    )
    warm_rul_seq(bs0)
    warm_volt_seq(bs0)
    measurements["combined_rul_then_volt_batched"] = _merge_measurement(
        _wall_cpu_ms(lambda: run_rul_then_volt_batched(bs0), repeats), ctx.n_batteries
    )

    if not skip_thread_pool:
        for nw in thread_workers:
            if nw < 1:
                continue

            def make_rul_pool(n: int, bs: int) -> Callable[[], None]:
                def pooled() -> None:
                    with ThreadPoolExecutor(max_workers=n) as ex:
                        list(
                            ex.map(
                                lambda w: rul.predict(w, batch_size=bs, verbose=0),
                                wl,
                            )
                        )

                return pooled

            def make_volt_pool(n: int, bs: int) -> Callable[[], None]:
                def pooled() -> None:
                    def one(w: np.ndarray) -> None:
                        s = np.full((w.shape[0], 1), 0.5, dtype=np.float32)
                        volt.predict([w, s], batch_size=bs, verbose=0)

                    with ThreadPoolExecutor(max_workers=n) as ex:
                        list(ex.map(one, wl))

                return pooled

            warm_rul_seq(bs0)
            measurements[f"rul_predict_threadpool_w{nw}"] = _merge_measurement(
                _wall_cpu_ms(make_rul_pool(nw, bs0), repeats), ctx.n_batteries
            )
            warm_volt_seq(bs0)
            measurements[f"volt_predict_threadpool_w{nw}"] = _merge_measurement(
                _wall_cpu_ms(make_volt_pool(nw, bs0), repeats), ctx.n_batteries
            )

    _alias_model_predict(measurements, batch_sizes)
    return measurements


def _tf_effective_threads() -> tuple[int | None, int | None]:
    try:
        import tensorflow as tf

        intra = tf.config.threading.get_intra_op_parallelism_threads()
        inter = tf.config.threading.get_inter_op_parallelism_threads()
        return intra, inter
    except Exception:
        return None, None


def _write_json_csv(
    *,
    profiles_dir: Path,
    label: str,
    notes: str,
    env: dict[str, Any],
    config: dict[str, Any],
    measurements: dict[str, Any],
    no_csv: bool,
    rul_seq_mean: float,
    rul_bat_mean: float,
    volt_seq_mean: float,
    volt_bat_mean: float,
    best_rul_tp: float | None,
    best_volt_tp: float | None,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "label": label,
        "timestamp": ts,
        "env": env,
        "config": config,
        "measurements": measurements,
        "notes": notes,
    }
    fname = f"{label.replace(' ', '_')}_{ts.replace(':', '')}.json"
    json_path = profiles_dir / fname
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")

    if not no_csv:
        csv_path = profiles_dir / "results.csv"
        speed_rul = (rul_seq_mean / rul_bat_mean) if rul_bat_mean > 0 else 0.0
        speed_volt = (volt_seq_mean / volt_bat_mean) if volt_bat_mean > 0 else 0.0
        row = {
            "label": label,
            "timestamp": ts,
            "git_sha": env.get("git_sha", ""),
            "rul_seq_mean_ms": rul_seq_mean,
            "rul_batched_mean_ms": rul_bat_mean,
            "rul_batched_speedup": round(speed_rul, 4),
            "volt_seq_mean_ms": volt_seq_mean,
            "volt_batched_mean_ms": volt_bat_mean,
            "volt_batched_speedup": round(speed_volt, 4),
            "best_rul_threadpool_mean_ms": best_rul_tp if best_rul_tp is not None else "",
            "best_volt_threadpool_mean_ms": best_volt_tp if best_volt_tp is not None else "",
        }
        fieldnames = list(row.keys())
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"Appended {csv_path}")

    return json_path


def _strip_tf_thread_flags(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a in ("--intra-op-threads", "--inter-op-threads"):
            skip_next = True
            continue
        if a.startswith("--intra-op-threads=") or a.startswith("--inter-op-threads="):
            continue
        out.append(a)
    return out


def _argv_for_sweep_child(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("--sweep-intra") or a.startswith("--sweep-inter"):
            if "=" not in a:
                skip_next = True
            continue
        out.append(a)
    return _strip_tf_thread_flags(out)


def _inject_label_suffix(argv: list[str], suffix: str) -> list[str]:
    has_label = False
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--label" and i + 1 < len(argv):
            has_label = True
            out.extend(["--label", argv[i + 1] + suffix])
            i += 2
            continue
        out.append(argv[i])
        i += 1
    if not has_label:
        out.extend(["--label", f"inference_sweep{suffix}"])
    return out


def _sweep_subprocesses(argv_base: list[str], intra_vals: list[int], inter_vals: list[int]) -> None:
    env = os.environ.copy()
    src = str(_REPO_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    for it in intra_vals:
        for jt in inter_vals:
            suf = f"_i{it}_o{jt}"
            body = _inject_label_suffix(argv_base, suf)
            cmd = (
                [sys.executable, "-m", "servers.battery.benchmark_inference"]
                + body
                + ["--intra-op-threads", str(it), "--inter-op-threads", str(jt)]
            )
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=False)


def run_single(args: argparse.Namespace) -> None:
    profiles_dir = args.profiles_dir or (_REPO_ROOT / "profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)

    if args.experimental_process_pool:
        print(
            "WARNING: --experimental-process-pool is a stub. "
            "ProcessPoolExecutor is not supported (TF reload per process, fork/spawn risk). "
            "Documented in profiles/README.md.",
            file=sys.stderr,
        )

    import tensorflow as tf

    if args.intra_op > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.intra_op)
    if args.inter_op > 0:
        tf.config.threading.set_inter_op_parallelism_threads(args.inter_op)

    from servers.battery.model_wrapper import (
        _MODEL_AVAILABLE,
        _MODELS,
        _NORMS,
        _load_once,
        concat_data,
        feature_selector,
    )

    _load_once()
    if not _MODEL_AVAILABLE or _MODELS is None or _NORMS is None:
        print(
            "Models unavailable. Set BATTERY_MODEL_WEIGHTS_DIR / BATTERY_NORMS_DIR "
            "and ensure weights exist. See battery.md.",
            file=sys.stderr,
        )
        sys.exit(1)

    models = _MODELS
    norms = _NORMS
    rng = np.random.default_rng(args.seed)

    batch_sizes = _parse_csv_ints(args.batch_sizes_csv)
    if not batch_sizes:
        batch_sizes = [args.batch_size]

    windows_list: list[np.ndarray]
    charges_list: list[np.ndarray] | None = None
    discharges_list: list[np.ndarray] | None = None
    summary_list: list[np.ndarray] | None = None
    data_src = "synthetic_windows"

    if args.windows_npz:
        data_src = f"npz:{args.windows_npz.name}"
        windows_list, ch_l, d_l, s_l = _load_npz_windows(
            args.windows_npz, args.n_batteries, args.n_cycles
        )
        charges_list, discharges_list, summary_list = ch_l, d_l, s_l
    else:
        windows_list = []
        for _ in range(args.n_batteries):
            w = rng.standard_normal((args.n_cycles, 50, 12), dtype=np.float64).astype(np.float32)
            windows_list.append(w)

    if args.include_feature_selectors:
        if charges_list is None:
            charges_list = []
            discharges_list = []
            summary_list = []
            for _ in range(args.n_batteries):
                ch, dis, summ = _synthetic_raw(args.n_cycles, rng)
                charges_list.append(ch)
                discharges_list.append(dis)
                summary_list.append(summ)
            for i in range(args.n_batteries):
                ch_feat = feature_selector(models["fs_ch"], charges_list[i], norms["charge"])
                dis_feat = feature_selector(models["fs_dis"], discharges_list[i], norms["discharge"])
                cf = concat_data(ch_feat, dis_feat, summary_list[i], norms["summary"])
                windows_list[i] = _windows_from_cell_feat(cf)
        elif args.windows_npz and charges_list is not None:
            # NPZ provided raw tensors — rebuild windows from features
            ctx0 = _BenchCtx(
                models["rul"],
                models["volt"],
                windows_list,
                charges_list,
                discharges_list,
                summary_list,
                args.n_batteries,
                args.warmup,
            )
            _infer_feature_lists(ctx0, models, norms)

    ctx = _BenchCtx(
        models["rul"],
        models["volt"],
        windows_list,
        charges_list,
        discharges_list,
        summary_list,
        args.n_batteries,
        args.warmup,
    )

    measurements = _run_heads_benchmark(
        ctx,
        batch_sizes=batch_sizes,
        repeats=args.repeats,
        thread_workers=list(args.thread_workers),
        skip_thread_pool=args.skip_thread_pool,
        include_fs=args.include_feature_selectors,
        fs_batched_multicell=args.fs_batched_multicell,
        models=models,
        norms=norms,
    )

    eff_intra, eff_inter = _tf_effective_threads()
    notes_full = args.notes
    if args.experimental_process_pool:
        prefix = (notes_full + " ") if notes_full else ""
        notes_full = prefix + "experimental_process_pool flag set (stub only; not executed)."

    rul_bat_key = (
        "rul_predict_batched"
        if len(batch_sizes) == 1
        else f"rul_predict_batched_bs{batch_sizes[0]}"
    )
    volt_bat_key = (
        "volt_predict_batched"
        if len(batch_sizes) == 1
        else f"volt_predict_batched_bs{batch_sizes[0]}"
    )

    rul_seq_mean = float(measurements["rul_predict_sequential"]["wall_ms"]["mean"])
    rul_bat_mean = float(measurements[rul_bat_key]["wall_ms"]["mean"])
    volt_seq_mean = float(measurements["volt_predict_sequential"]["wall_ms"]["mean"])
    volt_bat_mean = float(measurements[volt_bat_key]["wall_ms"]["mean"])

    best_rul_tp: float | None = None
    best_volt_tp: float | None = None
    if not args.skip_thread_pool:
        rul_tp = [
            float(measurements[k]["wall_ms"]["mean"])
            for k in measurements
            if k.startswith("rul_predict_threadpool_w")
        ]
        volt_tp = [
            float(measurements[k]["wall_ms"]["mean"])
            for k in measurements
            if k.startswith("volt_predict_threadpool_w")
        ]
        if rul_tp:
            best_rul_tp = min(rul_tp)
        if volt_tp:
            best_volt_tp = min(volt_tp)

    try:
        tf_ver = tf.__version__
    except Exception:
        tf_ver = ""

    config: dict[str, Any] = {
        "n_cells": args.n_batteries,
        "n_cycles": args.n_cycles,
        "n_repeats": args.repeats,
        "warmup_rounds_per_mode": args.warmup,
        "batch_sizes": batch_sizes,
        "primary_batch_size": batch_sizes[0],
        "intra_op_threads_requested": args.intra_op or None,
        "inter_op_threads_requested": args.inter_op or None,
        "tf_effective_intra_op_threads": eff_intra,
        "tf_effective_inter_op_threads": eff_inter,
        "thread_workers_swept": list(args.thread_workers) if not args.skip_thread_pool else [],
        "include_feature_selectors": args.include_feature_selectors,
        "fs_batched_multicell": args.fs_batched_multicell,
        "experimental_process_pool": args.experimental_process_pool,
    }

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "tf": tf_ver,
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": _try_ram_gb(),
        "git_sha": _try_git_sha(),
        "data_source": data_src,
    }

    _write_json_csv(
        profiles_dir=profiles_dir,
        label=args.label,
        notes=notes_full,
        env=env,
        config=config,
        measurements=measurements,
        no_csv=args.no_csv,
        rul_seq_mean=rul_seq_mean,
        rul_bat_mean=rul_bat_mean,
        volt_seq_mean=volt_seq_mean,
        volt_bat_mean=volt_bat_mean,
        best_rul_tp=best_rul_tp,
        best_volt_tp=best_volt_tp,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark battery TF/Keras inference (Part B + C).")
    parser.add_argument("--label", default="inference_bench", help="Run label for filenames and JSON.")
    parser.add_argument("--n-batteries", type=int, default=4, dest="n_batteries")
    parser.add_argument("--n-cycles", type=int, default=100, dest="n_cycles")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per mode (after warm-up).")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Identical untimed rounds before each timed block (RUL / volt / combined / thread).",
    )
    parser.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        dest="batch_sizes_csv",
        help='Comma-separated inner batch sizes for batched RUL/VOLT (e.g. "128,256"). Empty = use --batch-size only.',
    )
    parser.add_argument(
        "--windows-npz",
        type=Path,
        default=None,
        dest="windows_npz",
        help="NPZ with 'windows' (n_cycles,50,12); optional 'charges','discharges','summary' for full pipeline.",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=0,
        dest="intra_op",
        help="TensorFlow intra-op threads (0 = TF default). Applied before model load.",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=0,
        dest="inter_op",
        help="TensorFlow inter-op threads (0 = TF default).",
    )
    parser.add_argument(
        "--sweep-intra",
        type=str,
        default="",
        dest="sweep_intra",
        help="Comma-separated intra values; spawns one subprocess per (intra×inter) combo with fresh TF.",
    )
    parser.add_argument(
        "--sweep-inter",
        type=str,
        default="",
        dest="sweep_inter",
        help="Comma-separated inter values; use with --sweep-intra or alone (fixed intra from --intra-op or 0).",
    )
    parser.add_argument(
        "--thread-workers",
        type=int,
        nargs="*",
        default=[2, 4, 8],
        dest="thread_workers",
        help="ThreadPoolExecutor sizes to sweep (default: 2 4 8).",
    )
    parser.add_argument(
        "--skip-thread-pool",
        action="store_true",
        help="Skip ThreadPoolExecutor modes.",
    )
    parser.add_argument(
        "--include-feature-selectors",
        action="store_true",
        help="Time fs_ch + fs_dis + window build (+ optional multicell batched fs).",
    )
    parser.add_argument(
        "--fs-batched-multicell",
        action="store_true",
        help="With --include-feature-selectors and n_batteries>1: compare per-cell fs vs one concat fs run.",
    )
    parser.add_argument(
        "--experimental-process-pool",
        action="store_true",
        help="Document-only stub; warns. ProcessPoolExecutor not supported with TF.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-text note stored in JSON.",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/profiles).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not append profiles/results.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    si = _parse_csv_ints(args.sweep_intra)
    sj = _parse_csv_ints(args.sweep_inter)
    if si is not None or sj is not None:
        intra_vals = si if si is not None else [args.intra_op]
        inter_vals = sj if sj is not None else [args.inter_op]
        if not intra_vals:
            intra_vals = [0]
        if not inter_vals:
            inter_vals = [0]
        child_argv = _argv_for_sweep_child(sys.argv[1:])
        _sweep_subprocesses(child_argv, intra_vals, inter_vals)
        return

    run_single(args)


if __name__ == "__main__":
    main()