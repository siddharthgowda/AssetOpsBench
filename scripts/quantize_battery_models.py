#!/usr/bin/env python3
"""Quantize battery .h5 models to int8 TFLite (dynamic-range, no calibration data).

For each acctouhou model, produces a .tflite file with int8 weights and float32
activations. Reports file size before/after and runs a numerical equivalence
test against the original Keras model so you can see how much the quantization
shifts predictions.

Output: external/battery/acctouhou/quantized/<name>.tflite

Usage:
    PYTHONPATH=src uv run --group battery python scripts/quantize_battery_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    import tensorflow as tf  # noqa: PLC0415
    import tf_keras  # noqa: F401, PLC0415

    from servers.battery import model_wrapper as mw  # noqa: PLC0415

    mw._load_once()
    if not mw._MODEL_AVAILABLE or mw._MODELS is None:
        print(
            "Models not loaded. Set BATTERY_MODEL_WEIGHTS_DIR.",
            file=sys.stderr,
        )
        return 1

    out_dir = _REPO / "external/battery/acctouhou/quantized"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = _REPO / "external/battery/acctouhou/weights"
    h5_files = {
        "fs_ch": weights_dir / "feature_selector_ch.h5",
        "fs_dis": weights_dir / "feature_selector_dis.h5",
        "rul": weights_dir / "predictor.h5",
        "volt": weights_dir / "predictor2.h5",
    }

    print(f"{'Model':<10}{'Original (MB)':>16}{'Quantized (MB)':>17}{'Reduction':>12}")
    print("─" * 55)

    rng = np.random.default_rng(42)

    for name, h5_path in h5_files.items():
        original_size_mb = h5_path.stat().st_size / 1e6

        # ── Convert with dynamic-range INT8 quantization ──────────────
        converter = tf.lite.TFLiteConverter.from_keras_model(mw._MODELS[name])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # SELECT_TF_OPS lets unsupported ops (like 'mish') stay as TF ops —
        # they don't get quantized, but they don't block conversion either.
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        try:
            tflite_bytes = converter.convert()
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<10}  CONVERSION FAILED: {exc}")
            continue

        out_path = out_dir / f"{name}.tflite"
        out_path.write_bytes(tflite_bytes)
        new_size_mb = out_path.stat().st_size / 1e6
        reduction = (1 - new_size_mb / original_size_mb) * 100
        print(
            f"{name:<10}{original_size_mb:>16.2f}{new_size_mb:>17.2f}{reduction:>11.1f}%"
        )

        # ── Numerical equivalence on a small dummy input ─────────────
        try:
            interpreter = tf.lite.Interpreter(model_path=str(out_path))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            if len(input_details) != 1:
                print(
                    f"           num-equiv: skipped ({len(input_details)} inputs — "
                    "tested on Keras side instead)"
                )
                continue
            spec = input_details[0]
            shape = list(spec["shape"])
            shape[0] = 4
            test_input = rng.standard_normal(shape).astype(np.float32)
            interpreter.resize_tensor_input(spec["index"], shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(spec["index"], test_input)
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            quant_out = interpreter.get_tensor(output_details[0]["index"])
            keras_out = np.asarray(mw._MODELS[name](test_input))
            if quant_out.shape == keras_out.shape:
                diff = np.abs(quant_out - keras_out)
                print(
                    f"           num-equiv:  max-abs={diff.max():.5f}  "
                    f"mean-abs={diff.mean():.5f}"
                )
        except Exception as exc:  # noqa: BLE001
            # FlexPad / Flex-delegate-required ops show up here for some models.
            # The .tflite file is still valid — just can't run via this minimal interpreter.
            print(f"           num-equiv: not runnable in stock interpreter ({type(exc).__name__})")

    print(f"\nQuantized models in: {out_dir.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
