# Alternate inference runtimes (SavedModel, TFLite, ONNX, XLA)

These are **spikes** — validate accuracy on NASA tensors before relying on them.

## SavedModel export (todo: savedmodel-export)

```bash
PYTHONPATH=src uv run --group battery python scripts/export_battery_savedmodel.py \
  --out external/battery/acctouhou/saved_models
```

Then time `tf.saved_model.load` + serving signatures vs `tf_keras.models.load_model` + `predict`.

## `tf.function` / warm inference (todo: tf-function-wrap)

Wrap fixed-shape calls (e.g. `(100, 50, 12)` windows) in `tf.function` and compare cold vs warm latency. Check numerical parity against `model.predict`.

## XLA (todo: xla-experiment)

Set `TF_XLA_FLAGS=--tf_xla_cpu_global_jit` or use experimental jit paths where your TF build allows. Often fragile with legacy Keras saved graphs — keep behind an env flag and benchmark.

## TensorFlow Lite (todo: tflite-spike)

1. Convert RUL (and optionally voltage / feature selectors) with the TF Lite converter.
2. Run CPU delegate; compare outputs vs Keras on 2–3 cells (MAE on RUL trajectory).

Dependencies and exact flags depend on TF version; treat as research.

## ONNX Runtime (todo: onnx-spike)

1. `python -m tf2onnx.convert --keras ...` (or SavedModel input).
2. `onnxruntime.InferenceSession`; compare latency and MAE.

## Core ML (todo: coreml-note)

Apple-only deployment path. Risks: custom `mish`, multi-input voltage head, and shape contracts — expect conversion glue and on-device validation.

## Quantization MAE budget (todo: quantization-mae)

For INT8 / dynamic-range PTQ: define max acceptable **absolute RUL error** (cycles) and **voltage curve L2** on a fixed calibration + holdout split. Document in `battery.md`.

## Distillation (todo: distill-backlog)

Training a smaller student from the four `.h5` models is **not** in scope for v1; track as future work if you need a large step-change in throughput.
