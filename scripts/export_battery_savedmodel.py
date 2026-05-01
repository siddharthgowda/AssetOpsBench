#!/usr/bin/env python3
"""Export acctouhou Keras `.h5` models to TensorFlow SavedModel directories.

Usage (from repo root, with weights present)::

    PYTHONPATH=src uv run --group battery python scripts/export_battery_savedmodel.py \\
        --out external/battery/acctouhou/saved_models

Time inference via ``tf.saved_model.load`` + callables and compare to ``predict``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO / "external/battery/acctouhou/saved_models",
    )
    args = parser.parse_args()
    os.environ.setdefault("BATTERY_MODEL_WEIGHTS_DIR", str(_REPO / "external/battery/acctouhou/weights"))
    sys.path.insert(0, str(_REPO / "src"))
    import tensorflow as tf  # noqa: PLC0415
    import tf_keras  # noqa: F401, PLC0415

    from servers.battery import model_wrapper as mw  # noqa: PLC0415

    mw._load_once()
    if not mw._MODEL_AVAILABLE or mw._MODELS is None:
        print("Models not loaded; set BATTERY_MODEL_WEIGHTS_DIR.", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    m = mw._MODELS
    for name, model in m.items():
        dest = args.out / name
        dest.mkdir(parents=True, exist_ok=True)
        try:
            model.save(dest, save_format="tf")
            print("Wrote", dest)
        except Exception as e:  # noqa: BLE001
            print(f"Failed {name}: {e}", file=sys.stderr)
            # Legacy fallback: some graphs may need export via tf.saved_model.save(model, ...)
            try:
                tf.saved_model.save(model, str(dest))
                print("Wrote (saved_model)", dest)
            except Exception as e2:  # noqa: BLE001
                print(f"Fallback failed {name}: {e2}", file=sys.stderr)
    print("Done. Benchmark load + inference vs baseline `.h5` path.")


if __name__ == "__main__":
    main()
