"""TF model loader + math helpers for the acctouhou pretrained pipeline.

Loads four ``.h5`` Keras models (``feature_selector_ch``, ``feature_selector_dis``,
``predictor`` for RUL, ``predictor2`` for voltage) and four ``.npy`` normalization
tensors, then exposes:

  * ``predict_rul_for_cells(cells, *, use_compiled, batched)`` - the helper the
    optimized RUL tools call. Both flags are knobs the benchmark ablates.
  * ``predict_voltage_for_cell(...)`` - the deliberately-naive single-cell voltage
    predict used by the voltage tools (raw models, no batching, no compiled graphs).
  * ``get_compiled_models()`` - lazily wraps the raw Keras models in
    ``tf.function`` with flexible-batch input signatures so first inference
    doesn't pay per-shape Keras retracing.

Math primitives (``feature_selector``, ``concat_data``, ``build_sliding_windows``)
are copy-adapted from acctouhou's ``predict.py``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

# Silence cosmetic TF boot-time warnings before any TF import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

logger = logging.getLogger("battery-mcp-server")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Module-level state. Two model dicts so the benchmark can ablate compilation:
# raw Keras models stay around even after the compiled versions exist.
_MODELS_RAW: dict[str, Any] | None = None
_MODELS_COMPILED: dict[str, Any] | None = None
_NORMS: dict[str, Any] | None = None
_MODEL_AVAILABLE: bool = False


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _tf_mish(x):
    from tf_keras import backend as K

    return x * K.tanh(K.softplus(x))


# ── math primitives (acctouhou predict.py) ────────────────────────────────────


def feature_selector(model: Any, x: np.ndarray, norm: Any) -> np.ndarray:
    """Apply per-channel z-score normalization and run the FS model.

    ``x`` is float32 (n, 4, 500); the model expects (n, 500, 4)."""
    x = np.asarray(x, dtype=np.float32, order="C")
    normalized = (np.transpose(x, (0, 2, 1)) - norm[0]) / norm[1]
    return np.asarray(model.predict(normalized, batch_size=256, verbose=0))


def concat_data(
    x1: np.ndarray, x2: np.ndarray, summary: np.ndarray, summary_norm: Any
) -> np.ndarray:
    s = np.asarray(summary, dtype=np.float32)
    s_norm = (s - summary_norm[0]) / summary_norm[1]
    return np.hstack((x1, x2, s_norm.astype(np.float32, copy=False)))


def _pad_edge(arr: np.ndarray, target_len: int) -> np.ndarray:
    pad = target_len - len(arr)
    if pad > 0:
        return np.pad(arr, ((0, pad), (0, 0)), mode="edge")
    return arr[-target_len:]


def build_sliding_windows(cell_feat: np.ndarray) -> np.ndarray:
    """Build (n_cycles, 50, 12) windows from per-cycle features."""
    n = len(cell_feat)
    out = np.empty((n, 50, 12), dtype=np.float32)
    for k in range(n):
        out[k] = _pad_edge(cell_feat[max(0, k - 49) : k + 1], 50).astype(
            np.float32, copy=False
        )
    return out


# ── loaders ────────────────────────────────────────────────────────────────────


def _load_once() -> None:
    """Load .h5 models and .npy norms. Idempotent. Sets ``_MODEL_AVAILABLE``."""
    global _MODELS_RAW, _NORMS, _MODEL_AVAILABLE
    if _MODELS_RAW is not None:
        return
    weights_dir = _resolve_path(
        os.environ.get(
            "BATTERY_MODEL_WEIGHTS_DIR", "src/servers/battery/artifacts/weights"
        )
    )
    norms_dir = _resolve_path(
        os.environ.get("BATTERY_NORMS_DIR", "src/servers/battery/artifacts/norms")
    )
    required_w = [
        "feature_selector_ch.h5",
        "feature_selector_dis.h5",
        "predictor.h5",
        "predictor2.h5",
    ]
    required_n = [
        "charge_norm.npy",
        "discharge_norm.npy",
        "summary_norm.npy",
        "predict_renorm.npy",
    ]
    missing = [str(weights_dir / f) for f in required_w if not (weights_dir / f).exists()]
    missing += [str(norms_dir / f) for f in required_n if not (norms_dir / f).exists()]
    if missing:
        logger.warning(
            "Battery pretrained model unavailable. Missing files:\n  %s",
            "\n  ".join(missing),
        )
        return

    try:
        import tf_keras  # noqa: F401

        _MODELS_RAW = {
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
            "discharge": np.load(
                f"{norms_dir}/discharge_norm.npy", allow_pickle=True
            ).tolist(),
            "summary": np.load(f"{norms_dir}/summary_norm.npy", allow_pickle=True).tolist(),
            "renorm": np.load(f"{norms_dir}/predict_renorm.npy"),
        }
        _MODEL_AVAILABLE = True
        logger.info("Battery model loaded: weights=%s, norms=%s", weights_dir, norms_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("Battery model load failed (%s: %s)", type(e).__name__, e)
        _MODEL_AVAILABLE = False


# ── compiled (flexible-shape) wrappers ────────────────────────────────────────


class _CompiledKerasWrapper:
    """Wraps a Keras model in a ``tf.function`` with explicit ``input_signature``
    so the graph compiles once and is reused across any batch size.

    Exposes the subset of the Keras Model API we use:
      ``.predict(x)``  -> numpy
      ``.__call__(x)`` -> tf.Tensor
    """

    def __init__(self, model: Any, input_signature: Any):
        import tensorflow as tf

        self._signature = input_signature

        @tf.function(input_signature=input_signature)
        def _fn(x):
            return model(x, training=False)

        self._fn = _fn

    def warmup(self) -> None:
        import tensorflow as tf

        s = self._signature[0]
        arg = tf.zeros(
            [1] + [d if d is not None else 1 for d in s.shape[1:]], dtype=s.dtype
        )
        self._fn(arg)

    def __call__(self, x: Any, training: bool = False) -> Any:
        import tensorflow as tf

        return self._fn(tf.constant(np.asarray(x, dtype=np.float32)))

    def predict(self, x: Any, batch_size: int | None = None, verbose: int = 0) -> Any:
        out = self.__call__(x)
        return out.numpy() if hasattr(out, "numpy") else np.asarray(out)


def get_compiled_models() -> dict[str, Any] | None:
    """Build (or return cached) flexible-shape compiled wrappers for the three
    optimized-path models: ``fs_ch``, ``fs_dis``, ``rul``.

    The voltage model stays raw because the voltage tools are the deliberate
    unoptimized reference - no need to compile what we want slow on purpose."""
    global _MODELS_COMPILED
    if _MODELS_COMPILED is not None:
        return _MODELS_COMPILED
    if _MODELS_RAW is None or not _MODEL_AVAILABLE:
        return None
    import tensorflow as tf

    fs_sig = [tf.TensorSpec(shape=(None, 500, 4), dtype=tf.float32)]
    rul_sig = [tf.TensorSpec(shape=(None, 50, 12), dtype=tf.float32)]
    t0 = time.perf_counter()
    compiled = {
        "fs_ch": _CompiledKerasWrapper(_MODELS_RAW["fs_ch"], fs_sig),
        "fs_dis": _CompiledKerasWrapper(_MODELS_RAW["fs_dis"], fs_sig),
        "rul": _CompiledKerasWrapper(_MODELS_RAW["rul"], rul_sig),
    }
    for w in compiled.values():
        w.warmup()
    logger.info(
        "Compiled flexible-shape graphs in %.0f ms (3 models)",
        (time.perf_counter() - t0) * 1000,
    )
    _MODELS_COMPILED = compiled
    return _MODELS_COMPILED


# ── public predict helpers ────────────────────────────────────────────────────


def model_available() -> bool:
    return _MODEL_AVAILABLE


def _rul_one_cell(
    charges: np.ndarray,
    discharges: np.ndarray,
    summary: np.ndarray,
    models: dict[str, Any],
) -> np.ndarray:
    """Run FS + RUL for one cell using the supplied model dict."""
    assert _NORMS is not None
    ch_feat = feature_selector(models["fs_ch"], charges, _NORMS["charge"])
    dis_feat = feature_selector(models["fs_dis"], discharges, _NORMS["discharge"])
    cell_feat = concat_data(ch_feat, dis_feat, summary, _NORMS["summary"])
    windows = build_sliding_windows(cell_feat)
    rul_raw = np.asarray(models["rul"].predict(windows, batch_size=256, verbose=0))
    rul = rul_raw * _NORMS["renorm"][:, 1] + _NORMS["renorm"][:, 0]
    return rul[:, 0]


def predict_rul_for_cells(
    cells_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    use_compiled: bool = True,
    batched: bool = True,
) -> list[np.ndarray]:
    """Compute RUL trajectory per cell.

    Args:
      cells_data:    list of ``(charges, discharges, summary)`` per cell.
                     With ``batched=True`` all cells must share ``n_cycles``.
      use_compiled:  ``True`` -> use ``_MODELS_COMPILED`` (flexible-shape
                     ``tf.function`` graphs); ``False`` -> use ``_MODELS_RAW``.
      batched:       ``True`` -> concat all cells, 3 TF predicts total;
                     ``False`` -> per-cell loop, ``3 × N`` TF predicts.

    Returns one 1-D RUL trajectory per cell.
    """
    if not _MODEL_AVAILABLE or _NORMS is None or _MODELS_RAW is None:
        raise RuntimeError("Model not loaded")
    if not cells_data:
        return []

    if use_compiled:
        compiled = get_compiled_models()
        models = compiled if compiled is not None else _MODELS_RAW
    else:
        models = _MODELS_RAW

    if not batched:
        return [_rul_one_cell(ch, dis, s, models) for ch, dis, s in cells_data]

    n_per_cell = cells_data[0][0].shape[0]
    for ch, dis, s in cells_data:
        if ch.shape[0] != n_per_cell or dis.shape[0] != n_per_cell or s.shape[0] != n_per_cell:
            raise ValueError("Batched predict requires identical n_cycles across cells")

    big_ch = np.concatenate([c[0] for c in cells_data], axis=0)
    big_dis = np.concatenate([c[1] for c in cells_data], axis=0)
    big_s = np.concatenate([c[2] for c in cells_data], axis=0)
    ch_all = feature_selector(models["fs_ch"], big_ch, _NORMS["charge"])
    dis_all = feature_selector(models["fs_dis"], big_dis, _NORMS["discharge"])
    cf_all = concat_data(ch_all, dis_all, big_s, _NORMS["summary"])

    windows_list = [
        build_sliding_windows(cf_all[i * n_per_cell : (i + 1) * n_per_cell])
        for i in range(len(cells_data))
    ]
    big_w = np.concatenate(windows_list, axis=0)

    big_rul = np.asarray(models["rul"].predict(big_w, batch_size=256, verbose=0))
    big_rul = big_rul * _NORMS["renorm"][:, 1] + _NORMS["renorm"][:, 0]

    return [
        big_rul[i * n_per_cell : (i + 1) * n_per_cell, 0]
        for i in range(len(cells_data))
    ]


# ── Disk cache (RUL trajectories per cell) ───────────────────────────────────
# Smallest viable cache: one .npz per cell containing the predict output
# (rul_trajectory) and the CouchDB doc counts at compute time (for validation).
# No manifest file, no voltage outputs, no intermediate tensors.

_CACHE_DIR = _REPO_ROOT / "src" / "servers" / "battery" / "artifacts" / "cache"


def _cache_path(cell_id: str) -> Path:
    return _CACHE_DIR / f"{cell_id}.npz"


def cache_load(cell_id: str, n_charge_docs: int, n_discharge_docs: int) -> np.ndarray | None:
    """Return the cached rul_trajectory if present and matches doc counts; else None."""
    path = _cache_path(cell_id)
    if not path.exists():
        return None
    try:
        data = np.load(path)
        if tuple(int(x) for x in data["doc_counts"]) != (n_charge_docs, n_discharge_docs):
            return None
        return np.asarray(data["rul_trajectory"])
    except Exception:  # noqa: BLE001
        return None


def cache_save(
    cell_id: str,
    rul_trajectory: np.ndarray,
    n_charge_docs: int,
    n_discharge_docs: int,
) -> None:
    """Persist rul_trajectory + doc counts as one .npz."""
    path = _cache_path(cell_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rul_trajectory=rul_trajectory.astype(np.float32, copy=False),
        doc_counts=np.array([n_charge_docs, n_discharge_docs], dtype=np.int64),
    )


def cache_clear() -> None:
    """Wipe every .npz in the cache dir. Used between benchmark rungs."""
    if not _CACHE_DIR.exists():
        return
    for f in _CACHE_DIR.glob("*.npz"):
        f.unlink()


def predict_voltage_for_cell(
    charges: np.ndarray,
    discharges: np.ndarray,
    summary: np.ndarray,
) -> np.ndarray:
    """Naive single-cell voltage predict.

    No batching, no compiled graphs - the deliberate unoptimized reference for
    the voltage tools. Returns ``(n_cycles, 100)`` voltage curves."""
    if not _MODEL_AVAILABLE or _MODELS_RAW is None or _NORMS is None:
        raise RuntimeError("Model not loaded")
    ch_feat = feature_selector(_MODELS_RAW["fs_ch"], charges, _NORMS["charge"])
    dis_feat = feature_selector(_MODELS_RAW["fs_dis"], discharges, _NORMS["discharge"])
    cell_feat = concat_data(ch_feat, dis_feat, summary, _NORMS["summary"])
    windows = build_sliding_windows(cell_feat)
    second = np.full((len(windows), 1), 0.5, dtype=np.float32)
    out = _MODELS_RAW["volt"].predict([windows, second], batch_size=256, verbose=0)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return np.asarray(out)
