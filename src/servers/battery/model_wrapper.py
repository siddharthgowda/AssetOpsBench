"""Lazy TF model loader + inference cache for the acctouhou pretrained pipeline.

The core helpers (`mish`, `feature_selector`, `concat_data`, the sliding-window
logic) are copy-adapted from acctouhou's `predict.py`. We load the four .h5 and
four .npy files from BATTERY_MODEL_WEIGHTS_DIR / BATTERY_NORMS_DIR and expose a
module-level `_CACHE` that the MCP server populates at startup.

If either directory is missing or incomplete, `_MODEL_AVAILABLE` stays False and
model-dependent tools return ErrorResult; statistical tools still work.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

# Silence TensorFlow's cosmetic boot-time warnings before any TF import.
# TF_CPP_MIN_LOG_LEVEL=3 suppresses INFO/WARNING/ERROR from the C++ layer
# (including the "mixed_float16 may run slowly" message which is purely
# hypothetical — we force float32 at load time).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # extra noise suppression

import numpy as np

logger = logging.getLogger("battery-mcp-server")

# Repo root: src/servers/battery/model_wrapper.py → src/servers/battery → src/servers → src → repo
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _resolve_path(p: str) -> Path:
    """Resolve relative env-var paths against the repo root, not CWD."""
    path = Path(p)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path

# Module-level state (populated by _load_once)
_CACHE: dict[str, dict[str, Any]] = {}
_MODELS: dict[str, Any] | None = None
_NORMS: dict[str, Any] | None = None
_MODEL_AVAILABLE: bool = False


def _tf_mish(x):
    """Custom activation from acctouhou predict.py:25-26. Registered via custom_objects.

    Uses tf_keras (Keras 2 legacy) because acctouhou's .h5 weights were saved with
    Keras 2 and are incompatible with Keras 3's deserializer.
    """
    from tf_keras import backend as K

    return x * K.tanh(K.softplus(x))


def feature_selector(model, x: np.ndarray, norm) -> np.ndarray:
    """Copy of acctouhou predict.py:30-32."""
    normalized = (np.transpose(x, (0, 2, 1)) - norm[0]) / norm[1]
    return model.predict(normalized, batch_size=128, verbose=0)


def concat_data(x1: np.ndarray, x2: np.ndarray, summary: np.ndarray, summary_norm) -> np.ndarray:
    """Copy of acctouhou predict.py:33-35."""
    s_norm = (np.asarray(summary) - summary_norm[0]) / summary_norm[1]
    return np.hstack((x1, x2, s_norm))


def _pad_edge(arr: np.ndarray, target_len: int) -> np.ndarray:
    pad = target_len - len(arr)
    if pad > 0:
        return np.pad(arr, ((0, pad), (0, 0)), mode="edge")
    return arr[-target_len:]


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

    # Explicit file-level precheck so the user sees WHICH file is actually missing
    # instead of a generic "module not found" style message from tf.
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
        # Lazy imports — keep the server bootable even without TF/tf_keras installed.
        # tf_keras is Keras 2 (legacy), required because acctouhou's .h5 weights
        # were saved with Keras 2 and Keras 3 can't deserialize them.
        import tensorflow as tf
        import tf_keras  # noqa: F401

        # The acctouhou weights were saved with a mixed_float16 dtype policy,
        # which runs *slowly* on CPU and emits a noisy warning. If no GPU is
        # visible, force a float32 global policy before loading. This also
        # matters for portability: users on any CPU (Linux / Windows / Apple
        # Silicon) should get a clean boot.
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            tf_keras.mixed_precision.set_global_policy("float32")
            logger.info("No GPU detected; using float32 global dtype policy")
        else:
            logger.info("Detected %d GPU(s); keeping default dtype policy", len(gpus))

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
        logger.info(
            "Battery model loaded: weights=%s, norms=%s", weights_dir, norms_dir
        )
    except Exception as e:  # noqa: BLE001
        # Include full traceback so users see the actual root cause (version
        # mismatch, deserializer error, etc.) instead of a one-line message.
        logger.warning(
            "Battery pretrained model failed to load (%s: %s). "
            "Statistical tools will still work.",
            type(e).__name__,
            e,
            exc_info=True,
        )
        _MODEL_AVAILABLE = False


def precompute_cell(
    cell_id: str,
    charges: np.ndarray,
    discharges: np.ndarray,
    summary: np.ndarray,
) -> None:
    """Run the full acctouhou pipeline for one cell and populate `_CACHE[cell_id]`.

    charges / discharges: (100, 4, 500); summary: (100, 6).

    The predictor (RUL head) has input shape (None, 50, 12) → output (None, 2).
    The predictor2 (voltage head) has two inputs: (None, 50, 12) + (None, 1) → three
    outputs. The scalar second input appears to be an SOC/cycle-phase query; we pass
    a neutral value (0.5) to get a representative voltage curve at mid-discharge.
    """
    assert _MODELS is not None and _NORMS is not None, "_load_once() must be called first"
    t0 = time.perf_counter()
    ch_feat = feature_selector(_MODELS["fs_ch"], charges, _NORMS["charge"])        # (100, 3)
    dis_feat = feature_selector(_MODELS["fs_dis"], discharges, _NORMS["discharge"])  # (100, 3)
    cell_feat = concat_data(ch_feat, dis_feat, summary, _NORMS["summary"])           # (100, 12)
    # Sliding 50-cycle window, edge-padded for the first few cycles
    windows = np.stack(
        [_pad_edge(cell_feat[max(0, k - 49) : k + 1], 50) for k in range(len(cell_feat))]
    )                                                                                 # (100, 50, 12)
    # RUL head
    rul_pred = _MODELS["rul"].predict(windows, batch_size=256, verbose=0)             # (100, 2)
    # De-normalize RUL: predict_renorm is [[mean_rul, scale_rul], [mean_s, scale_s]]
    # predict.py applies: rul_pred * renorm[:,1] + renorm[:,0]
    rul_pred = rul_pred * _NORMS["renorm"][:, 1] + _NORMS["renorm"][:, 0]
    # Voltage head: needs a scalar second input (SOC query, ~[0,1]). Use 0.5 (mid).
    second_input = np.full((len(windows), 1), 0.5, dtype=np.float32)
    try:
        volt_outputs = _MODELS["volt"].predict(
            [windows, second_input], batch_size=256, verbose=0
        )
        # Returns a list/tuple of 3 outputs; first is the voltage curve (100 points)
        if isinstance(volt_outputs, (list, tuple)):
            volt_curves = np.asarray(volt_outputs[0])
        else:
            volt_curves = np.asarray(volt_outputs)
    except Exception as e:  # noqa: BLE001
        logger.warning("Voltage predictor failed for %s: %s; voltage curves unavailable", cell_id, e)
        volt_curves = np.zeros((len(windows), 100), dtype=np.float32)
    inference_ms = (time.perf_counter() - t0) * 1000.0 / max(len(cell_feat), 1)
    _CACHE[cell_id] = {
        "rul_trajectory": rul_pred[:, 0],
        "voltage_curves": volt_curves,                         # (100, 100) predicted V across SOC grid
        "inference_ms_per_cycle": float(inference_ms),
    }
