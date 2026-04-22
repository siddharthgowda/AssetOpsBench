"""LLM-narrated battery diagnosis. Orchestrates predict_rul + analyze_impedance_growth +
detect_capacity_outliers, then asks an LLM to classify the primary degradation mode.

Copy-adapt of FMSR's LLM pattern (src/servers/fmsr/main.py:93-142):
- Lazy LLM init
- Direct litellm call (no nested planner)
- Graceful degradation if LLM is unavailable (falls back to rule-based classification)

Exactly 3 few-shot examples per plan — capacity fade, lithium plating, impedance growth.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Union

import yaml

logger = logging.getLogger("battery-mcp-server")

_CHEMS = yaml.safe_load((Path(__file__).parent / "chemistries.yaml").read_text())
_NCA = _CHEMS["li_ion_nca_18650"]

_DEFAULT_MODEL_ID = (
    os.environ.get("BATTERY_MODEL_ID")
    or os.environ.get("FMSR_MODEL_ID")
    or "watsonx/meta-llama/llama-3-3-70b-instruct"
)

_llm = None
try:
    from llm import LiteLLMBackend

    _llm = LiteLLMBackend(_DEFAULT_MODEL_ID)
    logger.info("Battery diagnosis LLM initialized: %s", _DEFAULT_MODEL_ID)
except Exception as e:  # noqa: BLE001
    logger.warning("Battery diagnosis LLM unavailable (%s); will use rule-based fallback", e)

_PROMPT_TEMPLATE = f"""You classify Li-ion NCA 18650 battery degradation modes from numerical findings.

Thresholds (NASA B0xx NCA 18650):
- EOL capacity: {_NCA['eol_capacity_ah']} Ah (80% of {_NCA['nominal_capacity_ah']} Ah nominal)
- Rct alarm: >{_NCA['rct_alarm_multiplier']}× initial
- Coulombic efficiency drop alarm: >{_NCA['ce_anomaly_drop_pct']}%

Examples (exactly 3):
1. Input: {{"rul_cycles": 80, "rct_growth_per_cycle": 0.003, "rct_alarm": false, "outlier_z": 0.4}}
   Output: {{"primary_mode": "capacity_fade", "severity": "routine", "explanation": "Linear SEI-driven fade within expected envelope for NCA; impedance stable and not outlying."}}
2. Input: {{"rul_cycles": 60, "rct_growth_per_cycle": 0.004, "rct_alarm": false, "outlier_z": 2.3}}
   Output: {{"primary_mode": "lithium_plating", "severity": "alarm", "explanation": "Z-score >2 with only modest Rct growth suggests plating, typically from fast-charge or low-temperature operation."}}
3. Input: {{"rul_cycles": 90, "rct_growth_per_cycle": 0.02, "rct_alarm": true, "outlier_z": 0.6}}
   Output: {{"primary_mode": "impedance_growth", "severity": "alarm", "explanation": "DCIR rising faster than capacity fade indicates electrode degradation; elevated thermal-runaway risk."}}

Now classify the input below. Respond with ONLY a JSON object having keys primary_mode, severity, explanation. No prose outside the JSON.

Input: {{findings}}
Output:"""

_RECS = {
    "capacity_fade": [
        "Continue routine monitoring.",
        "Plan replacement before SOH drops to 80%.",
    ],
    "lithium_plating": [
        "Reduce charge C-rate.",
        "Avoid charging below 10°C.",
        "Inspect BMS fast-charge logic.",
    ],
    "impedance_growth": [
        "Flag for safety inspection.",
        "Not recommended for second-life grid storage.",
        "Increase monitoring frequency.",
    ],
    "healthy": [
        "No action needed.",
        "Continue baseline monitoring.",
    ],
}


def _extract_json(raw: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


def diagnose(asset_id: str):
    # Avoid circular import — import the tool wrappers lazily.
    from .main import (
        DiagnosisResult,
        ErrorResult,
        analyze_impedance_growth,
        detect_capacity_outliers,
        predict_rul,
    )

    rul = predict_rul(asset_id)
    imp = analyze_impedance_growth(asset_id)
    outlier = detect_capacity_outliers([asset_id])

    if isinstance(rul, ErrorResult) and isinstance(imp, ErrorResult):
        return ErrorResult(
            error=(
                f"Insufficient data to diagnose {asset_id}. "
                f"RUL error: {rul.error}. Impedance error: {imp.error}."
            )
        )

    findings: dict = {"asset_id": asset_id}
    rct_alarm = False
    if not isinstance(rul, ErrorResult):
        findings["rul_cycles"] = rul.rul_cycles
        if rul.mae_cycles is not None:
            findings["mae_cycles"] = rul.mae_cycles
    if not isinstance(imp, ErrorResult):
        findings["rct_growth_per_cycle"] = imp.rct_growth_per_cycle
        findings["rct_alarm"] = imp.alarm
        rct_alarm = imp.alarm
    if not isinstance(outlier, ErrorResult):
        findings["outlier_z"] = outlier.z_scores.get(asset_id, 0.0)

    # LLM path
    if _llm is not None:
        try:
            raw = _llm.generate(_PROMPT_TEMPLATE.format(findings=json.dumps(findings)))
            parsed = _extract_json(raw)
            if parsed.get("primary_mode") in _RECS:
                return DiagnosisResult(
                    asset_id=asset_id,
                    primary_mode=parsed["primary_mode"],
                    severity=parsed.get("severity", "unknown"),
                    explanation=parsed.get("explanation", ""),
                    recommendations=_RECS.get(parsed["primary_mode"], []),
                    numerical_findings={
                        k: float(v) for k, v in findings.items() if isinstance(v, (int, float))
                    },
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM diagnosis failed for %s (%s); falling back to rule-based", asset_id, e)

    # Rule-based fallback
    rul_cycles = findings.get("rul_cycles")
    outlier_z = findings.get("outlier_z", 0.0)
    if rct_alarm:
        mode = "impedance_growth"
        severity = "alarm"
        explanation = "Rule-based: Rct > 1.5× initial indicates electrode degradation."
    elif outlier_z > 2.0:
        mode = "lithium_plating"
        severity = "alarm"
        explanation = "Rule-based: fleet-outlier z > 2 with stable Rct suggests plating."
    elif isinstance(rul_cycles, (int, float)) and rul_cycles < 30:
        mode = "capacity_fade"
        severity = "alarm"
        explanation = "Rule-based: <30 cycles to EOL indicates advanced fade."
    else:
        mode = "healthy" if findings else "capacity_fade"
        severity = "routine"
        explanation = "Rule-based fallback (LLM unavailable or returned invalid JSON)."

    return DiagnosisResult(
        asset_id=asset_id,
        primary_mode=mode,
        severity=severity,
        explanation=explanation,
        recommendations=_RECS.get(mode, []),
        numerical_findings={
            k: float(v) for k, v in findings.items() if isinstance(v, (int, float))
        },
    )
