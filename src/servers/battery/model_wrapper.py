"""Lazy TF model loader + inference cache for the acctouhou pretrained pipeline.

The core helpers (`mish`, `feature_selector`, `concat_data`, the sliding-window
logic) are copy-adapted from acctouhou's `predict.py`. We load the four .h5 and
four .npy files from BATTERY_MODEL_WEIGHTS_DIR / BATTERY_NORMS_DIR (defaults
to ``src/servers/battery/artifacts/{weights,norms}``) and expose a
module-level `_CACHE` that the MCP server populates at startup.

If either directory is missing or incomplete, `_MODEL_AVAILABLE` stays False and
model-dependent tools return ErrorResult; statistical tools still work.

**Speed-related env vars** (see this server's `README.md`): batch sizes, TF threads, lazy
voltage head, optional Keras ``__call__`` vs ``predict``, GPU float32 force,
boot batched feature selectors.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

# Silence TensorFlow's cosmetic boot-time warnings before any TF import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # override in shell/.env per docs

import numpy as np

logger = logging.getLogger("battery-mcp-server")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


_CACHE: dict[str, dict[str, Any]] = {}
_MODELS: dict[str, Any] | None = None
_NORMS: dict[str, Any] | None = None
_MODEL_AVAILABLE: bool = False
_LAST_BATCH_TIMINGS: dict[str, float] = {}  # populated by precompute_cells_fully_batched


def _fs_batch_size() -> int:
    return max(1, int(os.environ.get("BATTERY_FS_BATCH_SIZE", "128") or 128))


def _head_batch_size() -> int:
    return max(1, int(os.environ.get("BATTERY_HEAD_BATCH_SIZE", "256") or 256))


# Optimization decisions baked in as defaults (formerly env-var gated):
#   - precompute_cells_fully_batched: one TF call per model across all cells
#   - lazy voltage: voltage curves are computed on-demand by ensure_voltage_curves
#   - pre-compiled flexible-shape graphs (_compile_models): no per-shape retracing
#
# Failed/superseded experiments removed but preserved in code for future
# ablation:
#   - SavedModelWrapper class (kept) - instantiate directly to test SavedModel path
#   - TFLiteWrapper class (kept) - instantiate directly to test int8 inference
#   - precompute_cells_batched_fs (kept) - Rushin's partial FS-only batching
#
# See this server's README.md "Optimizations attempted" for measured impact of each.


def _tf_mish(x):
    from tf_keras import backend as K

    return x * K.tanh(K.softplus(x))


class SavedModelWrapper:
    """Drop-in Keras-like wrapper around a tf.saved_model serving signature.

    Bypasses Keras's predict() pipeline (which retraces tf.functions per shape)
    and calls the baked-in graph directly. Trades the per-shape graph trace cost
    for ~zero per-call overhead - useful for cold-start when the same model is
    called once on a single (large) batch.

    Mirrors enough of tf_keras.Model for our internal call sites:
      .predict(x, batch_size=N, verbose=0)  -> numpy array of primary output
      .__call__(x, training=False)          -> tf.Tensor of primary output

    For multi-output models (e.g. voltage predictor), a single primary output
    is selected at construction; auxiliary outputs are dropped. Inputs are
    matched to signature kwargs by sorted name (input_1 < input_2 etc.).
    """

    def __init__(self, savedmodel_path: str, primary_output: str | None = None):
        import tensorflow as tf

        self._loaded = tf.saved_model.load(savedmodel_path)
        self._serving_fn = self._loaded.signatures["serving_default"]
        sig = self._serving_fn.structured_input_signature
        # sig is ((positional_args,), {kwarg_specs}); we only use kwargs.
        self._input_names = sorted(sig[1].keys())
        outputs = self._serving_fn.structured_outputs
        if primary_output is not None:
            if primary_output not in outputs:
                raise KeyError(
                    f"primary_output={primary_output!r} not in {list(outputs.keys())}"
                )
            self._output_name = primary_output
        else:
            if len(outputs) != 1:
                raise ValueError(
                    f"Multi-output model needs explicit primary_output; got {list(outputs.keys())}"
                )
            self._output_name = next(iter(outputs.keys()))

    def _build_inputs(self, x: Any) -> dict:
        import tensorflow as tf

        if isinstance(x, (list, tuple)):
            if len(x) != len(self._input_names):
                raise ValueError(
                    f"Expected {len(self._input_names)} inputs, got {len(x)}"
                )
            return {
                name: tf.constant(np.asarray(arr, dtype=np.float32))
                for name, arr in zip(self._input_names, x)
            }
        if len(self._input_names) != 1:
            raise ValueError(
                f"Single input passed but model expects {len(self._input_names)}"
            )
        return {self._input_names[0]: tf.constant(np.asarray(x, dtype=np.float32))}

    def predict(self, x: Any, batch_size: int | None = None, verbose: int = 0) -> np.ndarray:
        if isinstance(x, (list, tuple)):
            n = len(x[0])
        else:
            n = len(x)
        if batch_size is None or n <= batch_size:
            inputs = self._build_inputs(x)
            return self._serving_fn(**inputs)[self._output_name].numpy()
        chunks: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            if isinstance(x, (list, tuple)):
                chunk = [arr[i : i + batch_size] for arr in x]
            else:
                chunk = x[i : i + batch_size]
            inputs = self._build_inputs(chunk)
            chunks.append(self._serving_fn(**inputs)[self._output_name].numpy())
        return np.concatenate(chunks, axis=0)

    def __call__(self, x: Any, training: bool = False) -> Any:
        inputs = self._build_inputs(x)
        return self._serving_fn(**inputs)[self._output_name]


def _as_numpy(out: Any) -> np.ndarray:
    if hasattr(out, "numpy"):
        return np.asarray(out.numpy())
    return np.asarray(out)


def _forward_fs(model: Any, normalized: np.ndarray) -> np.ndarray:
    return model.predict(normalized, batch_size=_fs_batch_size(), verbose=0)


def _forward_rul(model: Any, windows: np.ndarray) -> np.ndarray:
    return model.predict(windows, batch_size=_head_batch_size(), verbose=0)


def _forward_volt(model: Any, windows: np.ndarray, second_input: np.ndarray) -> Any:
    return model.predict(
        [windows, second_input], batch_size=_head_batch_size(), verbose=0
    )


def feature_selector(model, x: np.ndarray, norm) -> np.ndarray:
    """Acctouhou predict.py charge/discharge branch; ``x`` should be float32 (n, 4, 500)."""
    x = np.asarray(x, dtype=np.float32, order="C")
    normalized = (np.transpose(x, (0, 2, 1)) - norm[0]) / norm[1]
    return _forward_fs(model, normalized)


def concat_data(x1: np.ndarray, x2: np.ndarray, summary: np.ndarray, summary_norm) -> np.ndarray:
    s = np.asarray(summary, dtype=np.float32)
    s_norm = (s - summary_norm[0]) / summary_norm[1]
    return np.hstack((x1, x2, s_norm.astype(np.float32, copy=False)))


def _pad_edge(arr: np.ndarray, target_len: int) -> np.ndarray:
    pad = target_len - len(arr)
    if pad > 0:
        return np.pad(arr, ((0, pad), (0, 0)), mode="edge")
    return arr[-target_len:]


def build_sliding_windows(cell_feat: np.ndarray) -> np.ndarray:
    """Build (n_cycles, 50, 12) windows - preallocated, equivalent to the original stack."""
    n = len(cell_feat)
    out = np.empty((n, 50, 12), dtype=np.float32)
    for k in range(n):
        out[k] = _pad_edge(cell_feat[max(0, k - 49) : k + 1], 50).astype(
            np.float32, copy=False
        )
    return out


def _load_once() -> None:
    """Load .h5 models and .npy norms. Idempotent; sets _MODEL_AVAILABLE on success."""
    global _MODELS, _NORMS, _MODEL_AVAILABLE
    if _MODELS is not None:
        return
    weights_dir = _resolve_path(
        os.environ.get(
            "BATTERY_MODEL_WEIGHTS_DIR", "src/servers/battery/artifacts/weights"
        )
    )
    norms_dir = _resolve_path(
        os.environ.get(
            "BATTERY_NORMS_DIR", "src/servers/battery/artifacts/norms"
        )
    )

    required_weights = ["feature_selector_ch.h5", "feature_selector_dis.h5", "predictor.h5", "predictor2.h5"]
    required_norms = ["charge_norm.npy", "discharge_norm.npy", "summary_norm.npy", "predict_renorm.npy"]
    missing: list[str] = []
    for f in required_weights:
        p = weights_dir / f
        if not p.exists():
            missing.append(str(p))
    for f in required_norms:
        p = norms_dir / f
        if not p.exists():
            missing.append(str(p))
    if missing:
        logger.warning(
            "Battery pretrained model unavailable. Missing files:\n  %s\n"
            "  weights_dir=%s\n  norms_dir=%s\n"
            "Fix: update BATTERY_MODEL_WEIGHTS_DIR and BATTERY_NORMS_DIR in .env, "
            "or run scripts/setup_battery_artifacts.sh to diagnose.",
            "\n  ".join(missing),
            weights_dir,
            norms_dir,
        )
        return

    try:
        intra = int(os.environ.get("BATTERY_TF_INTRA_OP_THREADS", "0") or 0)
        inter = int(os.environ.get("BATTERY_TF_INTER_OP_THREADS", "0") or 0)

        import tensorflow as tf
        import tf_keras  # noqa: F401

        if intra > 0:
            tf.config.threading.set_intra_op_parallelism_threads(intra)
        if inter > 0:
            tf.config.threading.set_inter_op_parallelism_threads(inter)
        if intra > 0 or inter > 0:
            logger.info(
                "TF thread pools (requested): intra=%s inter=%s",
                intra or "default",
                inter or "default",
            )

        gpus = tf.config.list_physical_devices("GPU")
        force_gpu_f32 = os.environ.get("BATTERY_GPU_FORCE_FLOAT32", "").strip() == "1"
        if not gpus:
            tf_keras.mixed_precision.set_global_policy("float32")
            logger.info("No GPU detected; using float32 global dtype policy")
        elif force_gpu_f32:
            tf_keras.mixed_precision.set_global_policy("float32")
            logger.info("GPU present but BATTERY_GPU_FORCE_FLOAT32=1 - using float32 policy")
        else:
            logger.info("Detected %d GPU(s); keeping checkpoint dtype policy (mixed_float16 unless overridden)", len(gpus))

        _MODELS = {
            "fs_ch": tf_keras.models.load_model(
                f"{weights_dir}/feature_selector_ch.h5", compile=False
            ),
            "fs_dis": tf_keras.models.load_model(
                f"{weights_dir}/feature_selector_dis.h5",
                compile=False,
                custom_objects={"mish": _tf_mish},
            ),
            "rul": tf_keras.models.load_model(
                f"{weights_dir}/predictor.h5",
                compile=False,
                custom_objects={"mish": _tf_mish},
            ),
            "volt": tf_keras.models.load_model(
                f"{weights_dir}/predictor2.h5",
                compile=False,
                custom_objects={"mish": _tf_mish},
            ),
        }
        _NORMS = {
            "charge": np.load(f"{norms_dir}/charge_norm.npy", allow_pickle=True).tolist(),
            "discharge": np.load(f"{norms_dir}/discharge_norm.npy", allow_pickle=True).tolist(),
            "summary": np.load(f"{norms_dir}/summary_norm.npy", allow_pickle=True).tolist(),
            "renorm": np.load(f"{norms_dir}/predict_renorm.npy"),
        }
        _MODEL_AVAILABLE = True
        logger.info("Battery model loaded: weights=%s, norms=%s", weights_dir, norms_dir)
        # Pre-compile flexible-shape graphs so first inference doesn't pay
        # per-shape Keras retrace overhead (~450 ms saved on the inference path).
        _compile_models()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Battery pretrained model failed to load (%s: %s). "
            "Statistical tools will still work.",
            type(e).__name__,
            e,
            exc_info=True,
        )
        _MODEL_AVAILABLE = False


class TFLiteWrapper:
    """Drop-in Keras-like wrapper around a tf.lite.Interpreter on a quantized .tflite.

    Mirrors the subset of tf_keras.Model needed by feature_selector / _forward_rul /
    _forward_volt: ``.predict(x)`` returns numpy, ``.__call__(x)`` returns a numpy
    array (we don't expose tf.Tensor outputs to keep the interface simple).
    """

    def __init__(self, tflite_path: str, primary_output_index: int = 0):
        import tensorflow as tf  # noqa: PLC0415

        self._interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._primary_output_idx = primary_output_index
        # Sort inputs by name so multi-input models are deterministic
        self._input_order = sorted(
            range(len(self._input_details)),
            key=lambda i: self._input_details[i]["name"],
        )
        self._last_shapes: list[tuple[int, ...]] = [
            tuple(self._input_details[i]["shape"]) for i in self._input_order
        ]

    def _maybe_resize(self, shapes: list[tuple[int, ...]]) -> None:
        if [tuple(s) for s in shapes] == self._last_shapes:
            return
        for slot, shape in zip(self._input_order, shapes):
            self._interpreter.resize_tensor_input(
                self._input_details[slot]["index"], list(shape)
            )
        self._interpreter.allocate_tensors()
        self._last_shapes = [tuple(s) for s in shapes]

    def predict(self, x: Any, batch_size: int | None = None, verbose: int = 0) -> np.ndarray:
        if isinstance(x, (list, tuple)):
            inputs = list(x)
        else:
            inputs = [x]
        if len(inputs) != len(self._input_order):
            raise ValueError(
                f"Expected {len(self._input_order)} inputs, got {len(inputs)}"
            )
        arrays = [np.ascontiguousarray(np.asarray(a, dtype=np.float32)) for a in inputs]
        self._maybe_resize([a.shape for a in arrays])
        for slot, arr in zip(self._input_order, arrays):
            self._interpreter.set_tensor(self._input_details[slot]["index"], arr)
        self._interpreter.invoke()
        return self._interpreter.get_tensor(
            self._output_details[self._primary_output_idx]["index"]
        )

    def __call__(self, x: Any, training: bool = False) -> np.ndarray:
        return self.predict(x)


class _CompiledKerasWrapper:
    """Wraps a Keras model in a tf.function with explicit input_signature so the
    graph compiles once and is reused across any batch size (no per-shape retrace).

    Exposes the subset of Keras Model API used by our forward functions:
      .predict(x, batch_size=None, verbose=0) -> numpy (or list[numpy] for multi-output)
      .__call__(x, training=False) -> tf.Tensor (or list/tuple for multi-output)
    """

    def __init__(self, model: Any, input_signature: Any, multi_input: bool = False):
        import tensorflow as tf  # noqa: PLC0415

        self._multi_input = multi_input
        self._signature = input_signature

        if multi_input:
            @tf.function(input_signature=input_signature)
            def _fn(*args):
                return model(list(args), training=False)
        else:
            @tf.function(input_signature=input_signature)
            def _fn(x):
                return model(x, training=False)
        self._fn = _fn

    def warmup(self) -> None:
        """Force the graph trace by calling once at batch=1."""
        import tensorflow as tf  # noqa: PLC0415

        if self._multi_input:
            args = [
                tf.zeros([1] + [d if d is not None else 1 for d in s.shape[1:]], dtype=s.dtype)
                for s in self._signature
            ]
            self._fn(*args)
        else:
            s = self._signature[0]
            arg = tf.zeros([1] + [d if d is not None else 1 for d in s.shape[1:]], dtype=s.dtype)
            self._fn(arg)

    def __call__(self, x: Any, training: bool = False) -> Any:
        import tensorflow as tf  # noqa: PLC0415

        if self._multi_input and isinstance(x, (list, tuple)):
            tensors = tuple(tf.constant(np.asarray(a, dtype=np.float32)) for a in x)
            return self._fn(*tensors)
        return self._fn(tf.constant(np.asarray(x, dtype=np.float32)))

    def predict(self, x: Any, batch_size: int | None = None, verbose: int = 0) -> Any:
        out = self.__call__(x)
        if isinstance(out, (list, tuple)):
            return [t.numpy() if hasattr(t, "numpy") else np.asarray(t) for t in out]
        return out.numpy() if hasattr(out, "numpy") else np.asarray(out)


def _compile_models() -> None:
    """Replace _MODELS entries with _CompiledKerasWrapper instances bound to
    flexible-batch input signatures, then trigger compilation. Gated by
    BATTERY_PRECOMPILE_GRAPHS=1.

    After this runs, every subsequent inference call (any batch size) reuses one
    cached graph per model - eliminates the per-shape Keras retrace overhead.
    """
    import tensorflow as tf  # noqa: PLC0415

    if _MODELS is None:
        return
    fs_sig = [tf.TensorSpec(shape=(None, 500, 4), dtype=tf.float32)]
    rul_sig = [tf.TensorSpec(shape=(None, 50, 12), dtype=tf.float32)]
    volt_sig = [
        tf.TensorSpec(shape=(None, 50, 12), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    ]
    t0 = time.perf_counter()
    wrappers: dict[str, _CompiledKerasWrapper] = {}
    try:
        wrappers["fs_ch"] = _CompiledKerasWrapper(_MODELS["fs_ch"], fs_sig, multi_input=False)
        wrappers["fs_dis"] = _CompiledKerasWrapper(_MODELS["fs_dis"], fs_sig, multi_input=False)
        wrappers["rul"] = _CompiledKerasWrapper(_MODELS["rul"], rul_sig, multi_input=False)
        wrappers["volt"] = _CompiledKerasWrapper(_MODELS["volt"], volt_sig, multi_input=True)
        for name, w in wrappers.items():
            w.warmup()
        # Replace
        _MODELS["fs_ch"] = wrappers["fs_ch"]
        _MODELS["fs_dis"] = wrappers["fs_dis"]
        _MODELS["rul"] = wrappers["rul"]
        _MODELS["volt"] = wrappers["volt"]
        logger.info(
            "Pre-compiled flexible-shape graphs in %.0f ms (4 models)",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Pre-compile failed (%s); falling back to plain Keras models", e
        )


def _finalize_volt_output(volt_outputs: Any) -> np.ndarray:
    if isinstance(volt_outputs, (list, tuple)):
        return np.asarray(_as_numpy(volt_outputs[0]))
    return _as_numpy(volt_outputs)


def _run_cell_pipeline(
    cell_id: str,
    charges: np.ndarray,
    discharges: np.ndarray,
    summary: np.ndarray,
) -> None:
    """Per-cell precompute (used as a fallback path; the default boot uses
    precompute_cells_fully_batched). Voltage is deferred to ensure_voltage_curves."""
    assert _MODELS is not None and _NORMS is not None
    t0 = time.perf_counter()
    ch_feat = feature_selector(_MODELS["fs_ch"], charges, _NORMS["charge"])
    dis_feat = feature_selector(_MODELS["fs_dis"], discharges, _NORMS["discharge"])
    cell_feat = concat_data(ch_feat, dis_feat, summary, _NORMS["summary"])
    windows = build_sliding_windows(cell_feat)
    rul_raw = _forward_rul(_MODELS["rul"], windows)
    rul_pred = rul_raw * _NORMS["renorm"][:, 1] + _NORMS["renorm"][:, 0]
    inference_ms = (time.perf_counter() - t0) * 1000.0 / max(len(cell_feat), 1)
    _CACHE[cell_id] = {
        "rul_trajectory": rul_pred[:, 0],
        "voltage_curves": None,
        "_windows_for_volt_lazy": windows,
        "_voltage_lazy_pending": True,
        "inference_ms_per_cycle": float(inference_ms),
    }


# ── Disk cache (.npz per cell) ──────────────────────────────────────────────
# Survives across MCP subprocess restarts. Skips the entire feature-selector +
# RUL predict pipeline at boot when present and fresh. See ``BATTERY_REBUILD_CACHE``.

_DISK_CACHE_DIR = _REPO_ROOT / "src" / "servers" / "battery" / "artifacts" / "cache"


def _disk_cache_path(cell_id: str) -> Path:
    return _DISK_CACHE_DIR / f"{cell_id}.npz"


def _disk_cache_manifest_path(cell_id: str) -> Path:
    return _DISK_CACHE_DIR / f"{cell_id}.manifest.json"


def _rebuild_cache_forced() -> bool:
    return os.environ.get("BATTERY_REBUILD_CACHE", "0").strip() not in ("", "0", "false", "False")


def _try_load_from_disk(cell_id: str, n_charge_docs: int, n_discharge_docs: int) -> bool:
    """Populate _CACHE[cell_id] from disk if a fresh entry exists. Returns True on hit."""
    if _rebuild_cache_forced():
        return False
    npz_path = _disk_cache_path(cell_id)
    man_path = _disk_cache_manifest_path(cell_id)
    if not npz_path.exists() or not man_path.exists():
        return False
    try:
        import json as _json
        manifest = _json.loads(man_path.read_text(encoding="utf-8"))
        if (
            manifest.get("n_charge_docs") != n_charge_docs
            or manifest.get("n_discharge_docs") != n_discharge_docs
        ):
            return False
        npz = np.load(npz_path, allow_pickle=False)
        _CACHE[cell_id] = {
            "rul_trajectory": npz["rul_trajectory"],
            "voltage_curves": npz["voltage_curves"] if "voltage_curves" in npz.files else None,
            "_windows_for_volt_lazy": npz["windows"] if "windows" in npz.files else None,
            "_voltage_lazy_pending": "voltage_curves" not in npz.files,
            "inference_ms_per_cycle": float(manifest.get("inference_ms_per_cycle", 0.0)),
            "_disk_cache_hit": True,
        }
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Disk cache load failed for %s: %s; recomputing", cell_id, exc)
        return False


def _save_to_disk(cell_id: str, n_charge_docs: int, n_discharge_docs: int) -> None:
    """Write _CACHE[cell_id] to disk so the next boot can skip preprocess + predict."""
    entry = _CACHE.get(cell_id)
    if entry is None:
        return
    try:
        _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "rul_trajectory": np.asarray(entry["rul_trajectory"]),
        }
        if entry.get("voltage_curves") is not None:
            arrays["voltage_curves"] = np.asarray(entry["voltage_curves"])
        if entry.get("_windows_for_volt_lazy") is not None:
            arrays["windows"] = np.asarray(entry["_windows_for_volt_lazy"])
        np.savez_compressed(_disk_cache_path(cell_id), **arrays)
        import json as _json
        _disk_cache_manifest_path(cell_id).write_text(
            _json.dumps(
                {
                    "n_charge_docs": n_charge_docs,
                    "n_discharge_docs": n_discharge_docs,
                    "inference_ms_per_cycle": float(entry.get("inference_ms_per_cycle", 0.0)),
                    "has_voltage_curves": entry.get("voltage_curves") is not None,
                    "schema_version": 1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Disk cache save failed for %s: %s", cell_id, exc)


def precompute_cell(
    cell_id: str,
    charges: np.ndarray,
    discharges: np.ndarray,
    summary: np.ndarray,
) -> None:
    """Run the acctouhou pipeline for one cell and populate `_CACHE[cell_id]`.

    Disk-cache fast path: if ``artifacts/cache/<cell_id>.npz`` exists and the manifest
    matches the input shape, load it and skip the entire feature-selector + RUL predict
    pipeline. Otherwise compute as before and write the result for next time.
    Set ``BATTERY_REBUILD_CACHE=1`` to force a fresh compute.

    Set ``BATTERY_LAZY_VOLTAGE=1`` to skip voltage at boot; voltage is computed on
    first call to ``ensure_voltage_curves``.
    """
    n_ch = int(charges.shape[0])
    n_dis = int(discharges.shape[0])
    if _try_load_from_disk(cell_id, n_ch, n_dis):
        return
    _run_cell_pipeline(cell_id, charges, discharges, summary)
    _save_to_disk(cell_id, n_ch, n_dis)


def precompute_cells_batched_fs(
    cells: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Feature selectors on concatenated batches (same n_cycles per cell), then per-cell RUL+VOLT.

    ``cells`` is ``[(cell_id, charges, discharges, summary), ...]`` with identical
    leading dimensions per tensor.
    """
    assert _MODELS is not None and _NORMS is not None
    if not cells:
        return
    n_c = cells[0][1].shape[0]
    for cid, ch, dis, summ in cells:
        if ch.shape[0] != n_c or dis.shape[0] != n_c or summ.shape[0] != n_c:
            raise ValueError(f"Batched FS requires identical n_cycles; bad cell {cid}")

    big_ch = np.concatenate([np.asarray(c[1], dtype=np.float32) for c in cells], axis=0)
    big_dis = np.concatenate([np.asarray(c[2], dtype=np.float32) for c in cells], axis=0)
    big_s = np.concatenate([np.asarray(c[3], dtype=np.float32) for c in cells], axis=0)

    ch_all = feature_selector(_MODELS["fs_ch"], big_ch, _NORMS["charge"])
    dis_all = feature_selector(_MODELS["fs_dis"], big_dis, _NORMS["discharge"])
    cf_all = concat_data(ch_all, dis_all, big_s, _NORMS["summary"])

    for i, (cell_id, _, _, _) in enumerate(cells):
        sl = slice(i * n_c, (i + 1) * n_c)
        cell_feat = cf_all[sl]
        _run_cell_slice(cell_id, cell_feat)


def precompute_cells_fully_batched(
    cells: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Fully-batched precompute: ALL TF model calls (FS + RUL + voltage) run once
    across all cells. Total TF calls: 4 (or 3 with BATTERY_LAZY_VOLTAGE=1) regardless
    of N. Compare to ``precompute_cell``: 4 × N calls.

    Disk-cache fast path: any cell whose ``.npz`` is present and matches the input
    shape is loaded from disk and excluded from the batched compute. If all cells
    hit, no TF predict runs at all.

    Stage timings are stashed in module-global ``_LAST_BATCH_TIMINGS`` so the
    profiler can surface them without re-instrumenting.
    """
    global _LAST_BATCH_TIMINGS
    assert _MODELS is not None and _NORMS is not None
    if not cells:
        return

    # Disk-cache filter: drop cells we can serve from disk before the batched compute.
    miss_cells: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    n_docs_for_save: dict[str, tuple[int, int]] = {}
    for cell_id, ch, dis, summ in cells:
        n_ch = int(ch.shape[0])
        n_dis = int(dis.shape[0])
        n_docs_for_save[cell_id] = (n_ch, n_dis)
        if not _try_load_from_disk(cell_id, n_ch, n_dis):
            miss_cells.append((cell_id, ch, dis, summ))
    if not miss_cells:
        # All cells hit disk cache; skip TF predict entirely.
        _LAST_BATCH_TIMINGS = {
            "n_cells": float(len(cells)),
            "n_cycles_per_cell": float(cells[0][1].shape[0]),
            "feature_selectors_ms": 0.0,
            "sliding_windows_ms": 0.0,
            "rul_predict_ms": 0.0,
            "volt_predict_ms": 0.0,
            "total_ms": 0.0,
            "disk_cache_hits": float(len(cells)),
            "disk_cache_misses": 0.0,
        }
        return
    cells = miss_cells
    n_c = cells[0][1].shape[0]
    for cid, ch, dis, summ in cells:
        if ch.shape[0] != n_c or dis.shape[0] != n_c or summ.shape[0] != n_c:
            raise ValueError(f"Full-batch requires identical n_cycles; bad cell {cid}")

    timings: dict[str, float] = {"n_cells": float(len(cells)), "n_cycles_per_cell": float(n_c)}

    # Stage 1 - batched feature selectors (2 TF calls total)
    t0 = time.perf_counter()
    big_ch = np.concatenate([np.asarray(c[1], dtype=np.float32) for c in cells], axis=0)
    big_dis = np.concatenate([np.asarray(c[2], dtype=np.float32) for c in cells], axis=0)
    big_s = np.concatenate([np.asarray(c[3], dtype=np.float32) for c in cells], axis=0)
    ch_all = feature_selector(_MODELS["fs_ch"], big_ch, _NORMS["charge"])
    dis_all = feature_selector(_MODELS["fs_dis"], big_dis, _NORMS["discharge"])
    cf_all = concat_data(ch_all, dis_all, big_s, _NORMS["summary"])
    timings["feature_selectors_ms"] = (time.perf_counter() - t0) * 1000.0

    # Stage 2 - sliding windows per cell, then stack
    t0 = time.perf_counter()
    windows_list: list[np.ndarray] = []
    for i in range(len(cells)):
        cell_feat = cf_all[i * n_c : (i + 1) * n_c]
        windows_list.append(build_sliding_windows(cell_feat))
    big_windows = np.concatenate(windows_list, axis=0)
    timings["sliding_windows_ms"] = (time.perf_counter() - t0) * 1000.0

    # Stage 3 - batched RUL (1 TF call)
    t0 = time.perf_counter()
    big_rul_raw = _forward_rul(_MODELS["rul"], big_windows)
    big_rul = big_rul_raw * _NORMS["renorm"][:, 1] + _NORMS["renorm"][:, 0]
    timings["rul_predict_ms"] = (time.perf_counter() - t0) * 1000.0

    # Voltage is always deferred to first voltage tool call (ensure_voltage_curves).
    # Saves ~500 ms boot per cell at the cost of ~1 ms tax on first voltage query.
    timings["volt_predict_ms"] = 0.0

    timings["total_ms"] = (
        timings["feature_selectors_ms"]
        + timings["sliding_windows_ms"]
        + timings["rul_predict_ms"]
    )

    # Stage 5 - slice into _CACHE per cell (voltage deferred), then persist to disk
    inference_ms_per_cycle = timings["total_ms"] / max(len(cells) * n_c, 1)
    for i, (cell_id, _, _, _) in enumerate(cells):
        sl = slice(i * n_c, (i + 1) * n_c)
        _CACHE[cell_id] = {
            "rul_trajectory": big_rul[sl, 0],
            "voltage_curves": None,
            "_windows_for_volt_lazy": windows_list[i],
            "_voltage_lazy_pending": True,
            "inference_ms_per_cycle": float(inference_ms_per_cycle),
        }
        n_ch, n_dis = n_docs_for_save.get(cell_id, (0, 0))
        if n_ch and n_dis:
            _save_to_disk(cell_id, n_ch, n_dis)

    timings["disk_cache_hits"] = 0.0
    timings["disk_cache_misses"] = float(len(cells))
    _LAST_BATCH_TIMINGS = timings


def _run_cell_slice(cell_id: str, cell_feat: np.ndarray) -> None:
    """Per-cell post-FS pipeline (used by precompute_cells_batched_fs).
    Voltage is deferred to ensure_voltage_curves on first request."""
    assert _MODELS is not None and _NORMS is not None
    t0 = time.perf_counter()
    windows = build_sliding_windows(cell_feat)
    rul_raw = _forward_rul(_MODELS["rul"], windows)
    rul_pred = rul_raw * _NORMS["renorm"][:, 1] + _NORMS["renorm"][:, 0]
    inference_ms = (time.perf_counter() - t0) * 1000.0 / max(len(cell_feat), 1)
    _CACHE[cell_id] = {
        "rul_trajectory": rul_pred[:, 0],
        "voltage_curves": None,
        "_windows_for_volt_lazy": windows,
        "_voltage_lazy_pending": True,
        "inference_ms_per_cycle": float(inference_ms),
    }


def ensure_voltage_curves(cell_id: str) -> bool:
    """Materialize voltage curves if ``BATTERY_LAZY_VOLTAGE`` left them pending."""
    if not _MODEL_AVAILABLE or _MODELS is None or _NORMS is None:
        return False
    entry = _CACHE.get(cell_id)
    if not entry or not entry.get("_voltage_lazy_pending"):
        return entry is not None and entry.get("voltage_curves") is not None
    windows = entry.get("_windows_for_volt_lazy")
    if windows is None:
        return False
    second_input = np.full((len(windows), 1), 0.5, dtype=np.float32)
    try:
        volt_outputs = _forward_volt(_MODELS["volt"], windows, second_input)
        entry["voltage_curves"] = _finalize_volt_output(volt_outputs)
    except Exception as e:  # noqa: BLE001
        logger.warning("Lazy voltage failed for %s: %s", cell_id, e)
        entry["voltage_curves"] = np.zeros((len(windows), 100), dtype=np.float32)
    entry["_voltage_lazy_pending"] = False
    del entry["_windows_for_volt_lazy"]
    return True
