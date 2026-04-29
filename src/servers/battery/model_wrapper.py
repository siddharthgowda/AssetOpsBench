"""Lazy TF model loader + inference cache for the acctouhou pretrained pipeline.

The core helpers (`mish`, `feature_selector`, `concat_data`, the sliding-window
logic) are copy-adapted from acctouhou's `predict.py`. We load the four .h5 and
four .npy files from BATTERY_MODEL_WEIGHTS_DIR / BATTERY_NORMS_DIR and expose a
module-level `_CACHE` that the MCP server populates at startup.

If either directory is missing or incomplete, `_MODEL_AVAILABLE` stays False and
model-dependent tools return ErrorResult; statistical tools still work.

**Speed-related env vars** (see `battery.md`): batch sizes, TF threads, lazy
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


def _fs_batch_size() -> int:
    return max(1, int(os.environ.get("BATTERY_FS_BATCH_SIZE", "128") or 128))


def _head_batch_size() -> int:
    return max(1, int(os.environ.get("BATTERY_HEAD_BATCH_SIZE", "256") or 256))


def _keras_use_call() -> bool:
    return os.environ.get("BATTERY_KERAS_USE_CALL", "").strip() == "1"


def _lazy_voltage() -> bool:
    return os.environ.get("BATTERY_LAZY_VOLTAGE", "").strip() == "1"


def _tf_mish(x):
    from tf_keras import backend as K

    return x * K.tanh(K.softplus(x))


def _as_numpy(out: Any) -> np.ndarray:
    if hasattr(out, "numpy"):
        return np.asarray(out.numpy())
    return np.asarray(out)


def _forward_fs(model: Any, normalized: np.ndarray) -> np.ndarray:
    bs = _fs_batch_size()
    if _keras_use_call():
        return _as_numpy(model(normalized, training=False))
    return model.predict(normalized, batch_size=bs, verbose=0)


def _forward_rul(model: Any, windows: np.ndarray) -> np.ndarray:
    bs = _head_batch_size()
    if _keras_use_call():
        return _as_numpy(model(windows, training=False))
    return model.predict(windows, batch_size=bs, verbose=0)


def _forward_volt(model: Any, windows: np.ndarray, second_input: np.ndarray) -> Any:
    bs = _head_batch_size()
    if _keras_use_call():
        return model((windows, second_input), training=False)
    return model.predict([windows, second_input], batch_size=bs, verbose=0)


def feature_selector(model, x: np.ndarray, norm) -> np.ndarray:
    """Acctouhou predict.py charge/discharge branch; ``x`` should be float32 (n, 4, 500)."""
    x = np.asarray(x, dtype=np.float32, order="C")
    normalized = (np.transpose(x, (0, 2, 1)) - norm[0]) / norm[1]
    return _forward_fs(model, normalized)


def concat_data(x1: np.ndarray, x2: np.ndarray, summary: np.ndarray, summary_norm) -> np.ndarray:
    s = np.asarray(summary, dtype=np.float32, copy=False)
    s_norm = (s - summary_norm[0]) / summary_norm[1]
    return np.hstack((x1, x2, s_norm.astype(np.float32, copy=False)))


def _pad_edge(arr: np.ndarray, target_len: int) -> np.ndarray:
    pad = target_len - len(arr)
    if pad > 0:
        return np.pad(arr, ((0, pad), (0, 0)), mode="edge")
    return arr[-target_len:]


def build_sliding_windows(cell_feat: np.ndarray) -> np.ndarray:
    """Build (n_cycles, 50, 12) windows — preallocated, equivalent to the original stack."""
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
        os.environ.get("BATTERY_MODEL_WEIGHTS_DIR", "external/battery/acctouhou/weights")
    )
    norms_dir = _resolve_path(
        os.environ.get("BATTERY_NORMS_DIR", "external/battery/acctouhou/norms")
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
            logger.info("GPU present but BATTERY_GPU_FORCE_FLOAT32=1 — using float32 policy")
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
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Battery pretrained model failed to load (%s: %s). "
            "Statistical tools will still work.",
            type(e).__name__,
            e,
            exc_info=True,
        )
        _MODEL_AVAILABLE = False


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
    assert _MODELS is not None and _NORMS is not None
    t0 = time.perf_counter()
    ch_feat = feature_selector(_MODELS["fs_ch"], charges, _NORMS["charge"])
    dis_feat = feature_selector(_MODELS["fs_dis"], discharges, _NORMS["discharge"])
    cell_feat = concat_data(ch_feat, dis_feat, summary, _NORMS["summary"])
    windows = build_sliding_windows(cell_feat)
    rul_raw = _forward_rul(_MODELS["rul"], windows)
    rul_pred = rul_raw * _NORMS["renorm"][:, 1] + _NORMS["renorm"][:, 0]

    lazy = _lazy_voltage()
    if lazy:
        inference_ms = (time.perf_counter() - t0) * 1000.0 / max(len(cell_feat), 1)
        _CACHE[cell_id] = {
            "rul_trajectory": rul_pred[:, 0],
            "voltage_curves": None,
            "_windows_for_volt_lazy": windows,
            "_voltage_lazy_pending": True,
            "inference_ms_per_cycle": float(inference_ms),
        }
        return

    second_input = np.full((len(windows), 1), 0.5, dtype=np.float32)
    try:
        volt_outputs = _forward_volt(_MODELS["volt"], windows, second_input)
        volt_curves = _finalize_volt_output(volt_outputs)
    except Exception as e:  # noqa: BLE001
        logger.warning("Voltage predictor failed for %s: %s; voltage curves unavailable", cell_id, e)
        volt_curves = np.zeros((len(windows), 100), dtype=np.float32)
    inference_ms = (time.perf_counter() - t0) * 1000.0 / max(len(cell_feat), 1)
    _CACHE[cell_id] = {
        "rul_trajectory": rul_pred[:, 0],
        "voltage_curves": volt_curves,
        "_voltage_lazy_pending": False,
        "inference_ms_per_cycle": float(inference_ms),
    }


def precompute_cell(
    cell_id: str,
    charges: np.ndarray,
    discharges: np.ndarray,
    summary: np.ndarray,
) -> None:
    """Run the acctouhou pipeline for one cell and populate `_CACHE[cell_id]`.

    Set ``BATTERY_LAZY_VOLTAGE=1`` to skip voltage at boot; voltage is computed on
    first call to ``ensure_voltage_curves``.
    """
    _run_cell_pipeline(cell_id, charges, discharges, summary)


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


def _run_cell_slice(cell_id: str, cell_feat: np.ndarray) -> None:
    assert _MODELS is not None and _NORMS is not None
    t0 = time.perf_counter()
    windows = build_sliding_windows(cell_feat)
    rul_raw = _forward_rul(_MODELS["rul"], windows)
    rul_pred = rul_raw * _NORMS["renorm"][:, 1] + _NORMS["renorm"][:, 0]

    if _lazy_voltage():
        inference_ms = (time.perf_counter() - t0) * 1000.0 / max(len(cell_feat), 1)
        _CACHE[cell_id] = {
            "rul_trajectory": rul_pred[:, 0],
            "voltage_curves": None,
            "_windows_for_volt_lazy": windows,
            "_voltage_lazy_pending": True,
            "inference_ms_per_cycle": float(inference_ms),
        }
        return

    second_input = np.full((len(windows), 1), 0.5, dtype=np.float32)
    try:
        volt_outputs = _forward_volt(_MODELS["volt"], windows, second_input)
        volt_curves = _finalize_volt_output(volt_outputs)
    except Exception as e:  # noqa: BLE001
        logger.warning("Voltage predictor failed for %s: %s", cell_id, e)
        volt_curves = np.zeros((len(windows), 100), dtype=np.float32)
    inference_ms = (time.perf_counter() - t0) * 1000.0 / max(len(cell_feat), 1)
    _CACHE[cell_id] = {
        "rul_trajectory": rul_pred[:, 0],
        "voltage_curves": volt_curves,
        "_voltage_lazy_pending": False,
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
