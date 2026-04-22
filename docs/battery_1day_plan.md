# Battery MCP Server — 1-Day Plan (~12 hours)

**Goal:** Fully functional `battery` MCP server in a single day (~12 hours of focused work): CouchDB-backed, real Severson normalization, 8 tools including LLM-narrated diagnosis, formal validation against B0018, all 15 `battery_scenarios.json` scenarios running end-to-end, pytest suite, README. Every piece of code below is a copy-adapt from an existing pattern in the repo or from `.prediction_of_battery/` — almost no greenfield work.

**What already exists (no download, no setup work needed):**
- `.pretrained/{feature_selector_ch.h5, feature_selector_dis.h5, predictor.h5, predictor2.h5}` — model weights
- `.data/5. Battery Data Set_json/` — 34 raw NASA JSON files across 6 batches
- `.prediction_of_battery/Prediction_of_battery/1_Predicting/predict.py` — inference code to copy verbatim
- `.prediction_of_battery/Prediction_of_battery/4_data processing/data_processing.ipynb` — `inp_500` + preprocessing code to copy
- `src/servers/vibration/` — exact template for the server structure
- `src/couchdb/init_asset_data.py` — CouchDB bulk-insert pattern to copy
- `src/servers/fmsr/main.py` — LLM-inside-tool pattern to copy

**What the user must provide before starting:**
Download the four `.npy` normalization files from the acctouhou Google Drive into `.pretrained/`:
`charge_norm.npy`, `discharge_norm.npy`, `summary_norm.npy`, `predict_renorm.npy`. Takes ~2 minutes.
Without these, predictions are approximate. **Do this before H0.**

---

## Non-destructive contract

**All changes are additive.** The core framework is untouched.

| File | Type of change | Risk |
|------|---------------|------|
| `src/servers/battery/*` (new dir) | add | none |
| `src/couchdb/init_battery.py` (new) | add | none |
| `pyproject.toml` | append 1 line to `[project.scripts]` + new `[dependency-groups] battery` | none (TF is lazy-imported) |
| `src/agent/plan_execute/executor.py` | add 1 line to `DEFAULT_SERVER_PATHS` dict | none (existing keys unchanged) |
| `src/couchdb/couchdb_setup.sh` | append a new block at end | none |
| `.env.public` | append a new `# ── Battery server` block | none |
| `.gitignore` | already done (hidden dirs ignored) | none |

**Files I will NOT touch:** any other server's `main.py`, IoT / WO / FMSR / vibration / utilities code, `agent/` core (except the 1-line dict entry), existing tests, docker-compose, Dockerfiles.

---

## Part 1 — Core server (Hours 0–6)

### Hour 0 — setup (15 min)

```bash
# Verify weights present
ls .pretrained/
# Expect: feature_selector_ch.h5 feature_selector_dis.h5 predictor.h5 predictor2.h5
#         charge_norm.npy discharge_norm.npy summary_norm.npy predict_renorm.npy

# Verify raw data present
ls ".data/5. Battery Data Set_json/1. BatteryAgingARC-FY08Q4/"
# Expect: B0005.json B0006.json B0007.json B0018.json + READMEs

# Create skeleton
mkdir -p src/servers/battery/tests

# Add deps (additive)
uv sync --group battery
```

`pyproject.toml` edits (additive):

```toml
[project.scripts]
# ... existing entries unchanged ...
battery-mcp-server = "servers.battery.main:main"

[dependency-groups]
# ... existing entries unchanged ...
battery = ["tensorflow>=2.15", "scikit-learn>=1.3"]
```

Register in planner (1 line addition to `src/agent/plan_execute/executor.py:27-34`):

```python
DEFAULT_SERVER_PATHS: dict[str, Path | str] = {
    "iot": "iot-mcp-server",
    "utilities": "utilities-mcp-server",
    "fmsr": "fmsr-mcp-server",
    "tsfm": "tsfm-mcp-server",
    "wo": "wo-mcp-server",
    "vibration": "vibration-mcp-server",
    "battery": "battery-mcp-server",   # ← ADD THIS LINE
}
```

### Hour 0.5–1 — CouchDB ingestion (~30 min)

**New file: `src/couchdb/init_battery.py`**

Direct copy of `init_asset_data.py` with NASA-specific doc transformation. Pseudocode:

```python
# Copy all helpers verbatim from init_asset_data.py:
#   _db_url, _ensure_db, _create_indexes, _bulk_insert

def nasa_cycle_to_docs(cell_id: str, raw: dict) -> list[dict]:
    """Convert one NASA cell's raw cycles into CouchDB documents."""
    docs = []
    for i, c in enumerate(raw[cell_id]['cycle']):
        doc = {
            "asset_id": cell_id,
            "cycle_index": i,
            "cycle_type": c['type'],                      # charge | discharge | impedance
            "ambient_temperature": c['ambient_temperature'],
            "timestamp": _matlab_time_to_iso(c['time']),  # [Y,M,D,h,m,s] → ISO8601
            "data": c['data'],                            # preserve full within-cycle arrays
        }
        # Quality filter: drop too-short discharge cycles at ingest
        if c['type'] == 'discharge':
            if len(c['data'].get('Voltage_measured', [])) < 100:
                continue
        docs.append(doc)
    return docs

def main():
    data_dir = os.environ.get("BATTERY_DATA_DIR", ".data/5. Battery Data Set_json")
    db_name = os.environ.get("BATTERY_DBNAME", "battery")
    all_docs = []
    for cell_file in glob(f"{data_dir}/*/B*.json"):
        cell_id = Path(cell_file).stem       # "B0005"
        with open(cell_file) as f:
            raw = json.load(f)
        all_docs.extend(nasa_cycle_to_docs(cell_id, raw))
    _ensure_db(db_name, drop=True)
    _bulk_insert(db_name, all_docs)
    _create_indexes(db_name, [["asset_id", "cycle_type", "cycle_index"]])
```

**Edit `src/couchdb/couchdb_setup.sh`:** append (additive)

```bash
BATTERY_DATA_DIR="/sample_data/battery_nasa"
if [ -d "$BATTERY_DATA_DIR" ]; then
  echo "Loading battery data..."
  python3 /couchdb/init_battery.py \
    --data-dir "$BATTERY_DATA_DIR" \
    --db "${BATTERY_DBNAME:-battery}" \
    --drop
fi
```

**Edit `.env.public`:** append new section

```bash
# ── Battery server ───────────────────────────────────────────────────────────
BATTERY_DBNAME=battery
BATTERY_MODEL_WEIGHTS_DIR=.pretrained
BATTERY_DATA_DIR=.data/5. Battery Data Set_json
```

**Run:**

```bash
uv run python src/couchdb/init_battery.py
# Expect: "Inserted batch N/M" lines, finishing with "Done. 34 cells, ~XXXX docs."
```

Time: **~30 min** (pattern is 95% identical to `init_asset_data.py`).

### Hour 1–2 — Preprocessing + model inference (~60 min)

**New file: `src/servers/battery/preprocessing.py`**

Verbatim copy of `inp_500` from `data_processing.ipynb`, plus a NASA-specific adapter.

```python
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

def inp_500(x, t):
    """Copied verbatim from acctouhou data_processing.ipynb."""
    f = interp1d(t, x, kind='linear')
    t_new = np.linspace(t.min(), t.max(), num=500)
    return f(t_new)

def preprocess_cycle(data: dict) -> np.ndarray | None:
    """One NASA cycle → (4, 500) tensor [Q, V, I, T]. None if unusable."""
    t = np.asarray(data['Time'], dtype=float)
    if len(t) < 100:
        return None
    V = np.asarray(data['Voltage_measured'], dtype=float)
    I = np.asarray(data['Current_measured'], dtype=float)
    T = np.asarray(data['Temperature_measured'], dtype=float)
    Q = cumulative_trapezoid(np.abs(I), t, initial=0) / 3600.0  # Ah
    return np.stack([inp_500(arr, t) for arr in (Q, V, I, T)])

def preprocess_cell_from_couchdb(cell_id: str, client) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (charge[100,4,500], discharge[100,4,500], summary[100,4])."""
    charges = client.fetch_cycles(cell_id, cycle_type='charge')
    discharges = client.fetch_cycles(cell_id, cycle_type='discharge')
    n = min(len(charges), len(discharges), 100)
    ch, dis, summ = [], [], []
    for i in range(n):
        c_t = preprocess_cycle(charges[i]['data'])
        d_t = preprocess_cycle(discharges[i]['data'])
        if c_t is None or d_t is None:
            continue
        ch.append(c_t); dis.append(d_t)
        summ.append([
            charges[i]['data']['Time'][-1],                              # chargetime
            discharges[i]['data'].get('Capacity', 0.0),                  # discharge capacity (scalar)
            min(discharges[i]['data']['Temperature_measured']),
            max(discharges[i]['data']['Temperature_measured']),
        ])
    if len(ch) < 100:
        raise ValueError(f"{cell_id}: only {len(ch)} clean paired cycles")
    return np.array(ch[:100]), np.array(dis[:100]), np.array(summ[:100])
```

**New file: `src/servers/battery/model_wrapper.py`**

Direct copy-adapt of `.prediction_of_battery/Prediction_of_battery/1_Predicting/predict.py` lines 19–35 (helpers) and 48–63 (sliding window).

```python
# Copy verbatim from acctouhou predict.py:
from tensorflow.keras import backend as K

def mish(x): return x * K.tanh(K.softplus(x))

def feature_selector(model, x, norm):
    normalized = (np.transpose(x, (0, 2, 1)) - norm[0]) / norm[1]
    return model.predict(normalized, batch_size=128, verbose=0)

def concat_data(x1, x2, summary, summary_norm):
    s_norm = (summary - summary_norm[0]) / summary_norm[1]
    return np.hstack((x1, x2, s_norm))

# Module-level cache populated at server startup
_CACHE: dict[str, dict] = {}
_MODELS = None
_NORMS = None
_MODEL_AVAILABLE = False

def _load_once(weights_dir: str):
    global _MODELS, _NORMS, _MODEL_AVAILABLE
    if _MODELS is not None:
        return
    try:
        _MODELS = {
            'fs_ch':  tf.keras.models.load_model(f"{weights_dir}/feature_selector_ch.h5", compile=False),
            'fs_dis': tf.keras.models.load_model(f"{weights_dir}/feature_selector_dis.h5", compile=False, custom_objects={'mish': mish}),
            'rul':    tf.keras.models.load_model(f"{weights_dir}/predictor.h5", compile=False, custom_objects={'mish': mish}),
            'volt':   tf.keras.models.load_model(f"{weights_dir}/predictor2.h5", compile=False, custom_objects={'mish': mish}),
        }
        _NORMS = {
            'charge':    np.load(f"{weights_dir}/charge_norm.npy", allow_pickle=True).tolist(),
            'discharge': np.load(f"{weights_dir}/discharge_norm.npy", allow_pickle=True).tolist(),
            'summary':   np.load(f"{weights_dir}/summary_norm.npy", allow_pickle=True).tolist(),
            'renorm':    np.load(f"{weights_dir}/predict_renorm.npy"),
        }
        _MODEL_AVAILABLE = True
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Pretrained weights unavailable: {e}. Model tools disabled; statistical tools still work.")

def precompute_cell(cell_id, charges, discharges, summary):
    """Populate _CACHE[cell_id] with RUL trajectory + voltage curves + inference_ms."""
    t0 = time.perf_counter()
    ch_feat  = feature_selector(_MODELS['fs_ch'],  charges,  _NORMS['charge'])
    dis_feat = feature_selector(_MODELS['fs_dis'], discharges, _NORMS['discharge'])
    cell_feat = concat_data(ch_feat, dis_feat, summary, _NORMS['summary'])  # (100, 12)
    # Sliding 50-cycle window, edge-pad last ones
    windows = np.stack([_pad_edge(cell_feat[max(0, k-49):k+1], 50) for k in range(len(cell_feat))])
    rul_pred  = _MODELS['rul'].predict(windows,  batch_size=256, verbose=0)
    volt_pred = _MODELS['volt'].predict(windows, batch_size=256, verbose=0)
    rul_pred  = rul_pred  * _NORMS['renorm'][:,1] + _NORMS['renorm'][:,0]
    inference_ms = (time.perf_counter() - t0) * 1000 / len(cell_feat)
    _CACHE[cell_id] = {
        'rul_trajectory': rul_pred[:, 0],
        'voltage_curves': volt_pred,
        'inference_ms_per_cycle': inference_ms,
    }
```

Time: **~60 min** (90% is copy from `predict.py`; Coulomb-counting for Q is the only new code).

### Hour 2–4 — MCP server with 8 tools (~2 hours)

**New file: `src/servers/battery/main.py`** — mirror `src/servers/vibration/main.py`.

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("battery")

# ── Pydantic result models ───────────────────────────────────────
class ErrorResult(BaseModel): error: str
class BatteryListResult(BaseModel): cells: list[str]
class CycleSummaryResult(BaseModel): asset_id: str; rows: list[dict]
class RULResult(BaseModel): asset_id: str; rul_cycles: float; from_cycle: int; inference_ms: float
class VoltageCurveResult(BaseModel): asset_id: str; cycle_index: int; voltage: list[float]
class MilestonesResult(BaseModel): asset_id: str; crossings: dict[float, int]
class ImpedanceResult(BaseModel): asset_id: str; rct_growth_rate: float; alarm: bool
class OutlierResult(BaseModel): cells: list[str]; z_scores: dict[str, float]
class DiagnosisResult(BaseModel): asset_id: str; primary_mode: str; severity: str; explanation: str; recommendations: list[str]

# ── Server startup ───────────────────────────────────────────────
_USABLE = ["B0005","B0006","B0007","B0018","B0033","B0034","B0036","B0054","B0055","B0056"]

def _boot():
    weights = os.environ.get("BATTERY_MODEL_WEIGHTS_DIR", ".pretrained")
    _load_once(weights)
    client = CouchDBClient()
    for cell in _USABLE:
        try:
            ch, dis, summ = preprocess_cell_from_couchdb(cell, client)
            if _MODEL_AVAILABLE:
                precompute_cell(cell, ch, dis, summ)
        except Exception as e:
            logger.warning(f"Skipped {cell}: {e}")
    logger.info(f"{len(_CACHE)} cells ready")

_boot()

# ── Tools ────────────────────────────────────────────────────────

@mcp.tool()
def list_batteries(site_name: str = "MAIN") -> BatteryListResult:
    """List available lithium-ion battery cells for RUL, voltage curve, impedance analysis."""
    return BatteryListResult(cells=list(_CACHE.keys()))

@mcp.tool()
def get_battery_cycle_summary(asset_id: str) -> CycleSummaryResult:
    """Return per-cycle Capacity, max_T, avg_V, Rct for a battery cell."""
    ...

@mcp.tool()
def predict_rul(asset_id: str, from_cycle: int = 100) -> RULResult:
    """Remaining useful life in cycles for a lithium-ion cell."""
    if not _MODEL_AVAILABLE:
        return ErrorResult(error="Pretrained model unavailable; see README for setup")
    entry = _CACHE.get(asset_id)
    if not entry: return ErrorResult(error=f"No cached inference for {asset_id}")
    return RULResult(asset_id=asset_id, rul_cycles=float(entry['rul_trajectory'][from_cycle-1]),
                     from_cycle=from_cycle, inference_ms=entry['inference_ms_per_cycle'])

@mcp.tool()
def predict_voltage_curve(asset_id: str, cycle_index: int) -> VoltageCurveResult:
    """500-point voltage-vs-SOC curve for a specified cycle."""
    ...

@mcp.tool()
def predict_voltage_milestones(asset_id: str, thresholds: list[float] = [2.9, 2.8, 2.7]) -> MilestonesResult:
    """Cycle where predicted voltage first drops below each threshold (EOD timing)."""
    ...

@mcp.tool()
def analyze_impedance_growth(asset_id: str) -> ImpedanceResult:
    """Exponential curve fit on Rct across impedance cycles; returns growth rate + alarm."""
    ...

@mcp.tool()
def detect_capacity_outliers(asset_ids: list[str] | None = None, window: int = 50) -> OutlierResult:
    """Flag cells with fade rate >2 std from fleet mean."""
    ...

@mcp.tool()
def diagnose_battery(asset_id: str) -> DiagnosisResult:
    """Run all analyses + LLM-narrate findings: primary failure mode, severity, recommendations."""
    from .diagnosis import diagnose
    return diagnose(asset_id)

def main(): mcp.run(transport="stdio")

if __name__ == "__main__": main()
```

**New file: `src/servers/battery/couchdb_client.py`** — direct copy of `src/servers/vibration/couchdb_client.py`, rename `VIBRATION_DBNAME` → `BATTERY_DBNAME`, adapt `fetch_` methods for cycle-indexed selectors.

Time: **~2 hours** for all 8 tools + client.

### Hour 4–5 — `diagnose_battery` + `chemistries.yaml` (~1 hour)

**New file: `src/servers/battery/chemistries.yaml`**

```yaml
li_ion_nca_18650:
  nominal_capacity_ah: 2.0
  eol_threshold_soh: 0.80
  eol_capacity_ah: 1.4
  cutoff_voltage_min: 2.7
  voltage_milestones: [2.9, 2.8, 2.7]
  rct_alarm_multiplier: 1.5
  re_alarm_multiplier: 2.0
  ce_anomaly_drop_pct: 2.0
  temperature_alarm_c: 45
```

**New file: `src/servers/battery/diagnosis.py`** — copy FMSR's LLM pattern from `src/servers/fmsr/main.py:93-142`.

```python
# LLM init (copy from fmsr/main.py)
_DEFAULT_MODEL_ID = os.environ.get("BATTERY_MODEL_ID") or os.environ.get("FMSR_MODEL_ID", "watsonx/meta-llama/llama-3-3-70b-instruct")
# ... _build_llm() copy from fmsr ...

_PROMPT = """You classify Li-ion NCA battery degradation modes from numerical findings.

Thresholds:
{thresholds}

Examples:
1. Input: {{SOH:0.84, rct_growth:0.03, ce:0.992, knee:false}}
   Output: {{primary_mode:"capacity_fade", severity:"routine", explanation:"Linear SEI-driven fade"}}
2. Input: {{SOH:0.89, rct_growth:0.04, ce_min:0.968, fade_accel:true}}
   Output: {{primary_mode:"lithium_plating", severity:"alarm", explanation:"CE drop with accelerating fade indicates plating"}}
3. Input: {{SOH:0.91, rct_growth:0.18, ce:0.990, fade_linear:true}}
   Output: {{primary_mode:"impedance_growth", severity:"alarm", explanation:"DCIR rising faster than fade suggests electrode degradation"}}

Now classify:
Input: {findings}
Output (JSON):"""

def diagnose(asset_id: str) -> DiagnosisResult:
    rul = predict_rul(asset_id)
    imp = analyze_impedance_growth(asset_id)
    outlier = detect_capacity_outliers([asset_id])
    findings = {"asset_id": asset_id, "rul": rul.rul_cycles,
                "rct_growth": imp.rct_growth_rate,
                "outlier_z": outlier.z_scores.get(asset_id, 0.0)}
    raw = _llm.generate(_PROMPT.format(thresholds=_THRESHOLDS,
                                       findings=json.dumps(findings)))
    parsed = json.loads(_extract_json(raw))
    return DiagnosisResult(asset_id=asset_id, **parsed,
                           recommendations=_rec_for_mode(parsed['primary_mode']))
```

Time: **~1 hour** (FMSR has the exact pattern).

### Hour 5–6 — smoke test core functionality (~1 hour)

```bash
uv run plan-execute "What batteries are available at site MAIN?"
# Expect: list_batteries → BatteryListResult with 10 cells

uv run plan-execute "Predict RUL for B0018"
# Expect: predict_rul → RULResult; rul_cycles >0, inference_ms <100

uv run plan-execute "When will B0018 voltage drop below 2.8V?"
# Expect: predict_voltage_milestones → MilestonesResult

uv run plan-execute "Diagnose battery B0018"
# Expect: diagnose_battery → DiagnosisResult with primary_mode + explanation
```

**If the planner routes wrong** (e.g., picks TSFM for RUL): tune the docstring of the battery tool to include keywords like "lithium-ion", "RUL", "remaining useful life", "voltage drop", "EOD". The planner reads the first line of each tool's docstring.

**End of Part 1 (Hour 6):** Core server functional. 4 smoke-test scenarios work. Time to pause and commit.

---

## Part 2 — Validation, coverage, polish (Hours 6–12)

### Hour 6–8 — Formal B0018 validation gate (~2 hours)

**New file: `src/servers/battery/tests/test_validation.py`**

Why B0018: cleanest cell (132 discharge cycles, all ≥100 samples, standard 2A profile, 24°C). Fairest test of LFP→NCA transfer.

```python
def test_b0018_rul_mae():
    """Validation gate: predict_rul on B0018 at cycle 100 within ±30 cycles of ground truth."""
    from servers.battery.main import predict_rul
    from servers.battery.couchdb_client import CouchDBClient

    client = CouchDBClient()
    dis = client.fetch_cycles("B0018", "discharge")
    capacities = [d['data']['Capacity'] for d in dis if 'Capacity' in d['data']]
    eol = next((i for i, c in enumerate(capacities) if c < 1.4), len(capacities))
    gt_rul_at_cycle_100 = eol - 100

    pred = predict_rul("B0018", from_cycle=100)
    assert abs(pred.rul_cycles - gt_rul_at_cycle_100) <= 30, \
        f"RUL MAE too large: predicted {pred.rul_cycles}, ground truth {gt_rul_at_cycle_100}"
```

**Decision gate:**
- **Pass (MAE ≤ 30 cycles)** → acctouhou is v1, document the number in README.
- **Fail** → document in README, add fallback inside `predict_rul`:

```python
# Fallback inside predict_rul if _MODEL_AVAILABLE=False or validation fails:
caps = fetch_capacities(asset_id)[:from_cycle]
params, _ = scipy.optimize.curve_fit(_exp_decay, range(len(caps)), caps)
return _extrapolate_to_threshold(params, threshold=1.4) - from_cycle
```

Time: **~2 hours** (1h to write test, 1h buffer for fallback if needed).

### Hour 8–10 — Run all 15 scenarios (~2 hours)

```bash
for i in $(seq 0 14); do
    question=$(jq -r ".battery_management_scenarios[$i].request" battery_scenarios.json)
    echo "=== Scenario $((i+1)) ==="
    uv run plan-execute --show-plan "$question" 2>&1 | tee -a scenario_results.log
done
```

**Expected routing:**

| # | Primary tool | Server |
|---|-------------|--------|
| 1 | `predict_rul` | battery |
| 2 | `predict_voltage_milestones` | battery |
| 3 | `predict_rul` + `detect_capacity_outliers` | battery |
| 4 | `analyze_sensitivity` | tsfm |
| 5 | `predict_rul` (reports inference_ms) | battery |
| 6 | `predict_voltage_milestones` | battery |
| 7 | `predict_rul` | battery |
| 8 | `predict_voltage_milestones` | battery |
| 9 | `detect_capacity_outliers` | battery |
| 10 | `analyze_sensitivity` + `predict_rul` | tsfm + battery |
| 11 | `analyze_impedance_growth` + `predict_rul` | battery |
| 12 | `analyze_impedance_growth` | battery |
| 13 | `predict_rul(from_cycle=N*0.25)` — may fall back if <100 cycles | battery or tsfm |
| 14 | `analyze_impedance_growth` | battery |
| 15 | `predict_rul` + statistical baseline | battery |

**If routing is wrong:** tune tool docstrings with more-specific keywords. Don't touch the planner itself.

Time: **~2 hours** (most scenarios run cleanly; 3-5 may need docstring tuning).

### Hour 10–11 — pytest suite + error handling (~1 hour)

**`src/servers/battery/tests/conftest.py`** — copy pattern from `src/servers/vibration/tests/conftest.py`.

```python
@pytest.fixture(autouse=True)
def skip_if_no_backend():
    if not os.environ.get("COUCHDB_URL"):
        pytest.skip("COUCHDB_URL not set")
    if not Path(os.environ.get("BATTERY_MODEL_WEIGHTS_DIR", ".pretrained")).exists():
        pytest.skip("Pretrained weights not available")
```

**`test_preprocessing.py`** (~20 lines):

```python
def test_preprocess_cycle_shape():
    raw = json.load(open(".data/5. Battery Data Set_json/1. BatteryAgingARC-FY08Q4/B0018.json"))
    dis0 = next(c for c in raw['B0018']['cycle'] if c['type']=='discharge')
    tensor = preprocess_cycle(dis0['data'])
    assert tensor.shape == (4, 500)
    assert not np.any(np.isnan(tensor))

def test_short_cycles_filtered():
    fake = {"Time": list(range(50)), "Voltage_measured": [3.0]*50,
            "Current_measured": [-2.0]*50, "Temperature_measured": [25.0]*50}
    assert preprocess_cycle(fake) is None
```

**`test_tools.py`** (~40 lines): one happy-path assertion per tool.

```python
def test_predict_rul_returns_rul_result():
    result = predict_rul("B0018", from_cycle=100)
    assert isinstance(result, RULResult)
    assert result.rul_cycles > 0
    assert result.inference_ms < 100

def test_voltage_curve_has_500_points():
    result = predict_voltage_curve("B0018", cycle_index=50)
    assert len(result.voltage) == 500

def test_list_batteries_returns_10():
    result = list_batteries()
    assert len(result.cells) == 10
    assert "B0018" in result.cells
```

**Error-handling branches** in existing files:

`model_wrapper.py` already has `_MODEL_AVAILABLE` flag from Hour 1–2. `main.py` tools check it. `preprocessing.py` raises `ValueError` on insufficient cycles.

Time: **~1 hour**.

### Hour 11–12 — README + commit (~1 hour)

**`src/servers/battery/README.md`**

```markdown
# Battery MCP Server

Battery lithium-ion analytics: RUL, voltage curves, impedance, fleet outliers, LLM-narrated diagnosis.
Uses pretrained acctouhou weights (CPU-friendly, ~ms inference).

## Quick start

1. Download weights into `.pretrained/`:
   - `feature_selector_ch.h5`, `feature_selector_dis.h5`, `predictor.h5`, `predictor2.h5`
   - `charge_norm.npy`, `discharge_norm.npy`, `summary_norm.npy`, `predict_renorm.npy`
   Source: https://drive.google.com/... (from acctouhou README)
2. Download raw NASA data into `.data/5. Battery Data Set_json/`
3. Run CouchDB setup: `./src/couchdb/couchdb_setup.sh`
4. Install deps: `uv sync --group battery`
5. Start server: `uv run battery-mcp-server`

## Env vars

See `.env.public` for `BATTERY_DBNAME`, `BATTERY_MODEL_WEIGHTS_DIR`, `BATTERY_DATA_DIR`.

## Tools (8)

| Tool | Purpose |
|------|---------|
| `list_batteries` | List loaded cells |
| `get_battery_cycle_summary` | Per-cycle metrics |
| `predict_rul` | Remaining useful life (cycles) |
| `predict_voltage_curve` | 500-point V-SOC curve |
| `predict_voltage_milestones` | When V crosses thresholds |
| `analyze_impedance_growth` | Rct/Re trend + alarm |
| `detect_capacity_outliers` | Fleet z-score outliers |
| `diagnose_battery` | LLM-narrated one-shot diagnosis |

## Validation

B0018 RUL MAE: {FILL_IN_FROM_TEST} cycles (pass threshold: ≤30).

## Troubleshooting

- **Server starts but RUL queries error**: weights not downloaded. See step 1.
- **`list_batteries` returns empty**: CouchDB not initialized. Run `src/couchdb/init_battery.py`.
- **Planner routes to TSFM instead of battery**: tune docstring keywords in `main.py`.
```

`git commit` + `git tag battery-server-v1`.

Time: **~1 hour**.

---

## File structure at end of 12 hours

```
src/servers/battery/
    __init__.py
    main.py                  # FastMCP, 8 tools, Pydantic models (~300 lines)
    preprocessing.py         # inp_500 + preprocess_cycle + preprocess_cell (~80 lines)
    model_wrapper.py         # _CACHE + _load_once + precompute_cell (~120 lines)
    couchdb_client.py        # mirror vibration (~50 lines)
    diagnosis.py             # diagnose + _PROMPT + LLM init (~80 lines)
    chemistries.yaml         # 15 lines
    README.md                # setup + tools + validation + troubleshooting
    tests/
        conftest.py
        test_preprocessing.py
        test_tools.py
        test_validation.py

src/couchdb/
    init_battery.py          # NASA JSON → CouchDB (~80 lines)

Modified:
    pyproject.toml           # +1 script, +1 dep group
    src/agent/plan_execute/executor.py   # +1 line
    src/couchdb/couchdb_setup.sh         # +1 block
    .env.public                           # +1 section
```

Total new code: **~750 lines**, of which ~400 are direct copy-adapt from existing files.

---

## Acceptance at end of 12 hours

- [ ] `uv run battery-mcp-server` starts, logs "10 cells ready"
- [ ] `test_validation.py` reports MAE, passes OR documents fallback
- [ ] All 15 `battery_scenarios.json` requests produce non-error responses via `plan-execute`
- [ ] `uv run pytest src/servers/battery/tests/` green
- [ ] `inference_ms_per_cycle` < 100ms on CPU (scenario 5 constraint)
- [ ] Missing weights / missing CouchDB → graceful error messages (not crashes)
- [ ] `src/servers/battery/README.md` documents setup with validation number filled in
- [ ] Git tag `battery-server-v1` exists
- [ ] No regressions: `uv run pytest src/servers/{vibration,fmsr,iot,wo,tsfm,utilities}/tests/` still green
- [ ] Core framework untouched: `src/agent/` has only the 1-line `DEFAULT_SERVER_PATHS` addition

---

## Risk register

| Risk | Mitigation |
|------|------------|
| B0018 validation MAE > 30 cycles | Statistical curve-fit fallback inside `predict_rul`; documented in README |
| Planner misroutes battery queries | Hour 8–10 budgets routing fixes via docstring tuning |
| TF 2.15 rejects TF 2.2-era `.h5` weights | Pin older TF in `[dependency-groups] battery` (`tensorflow>=2.10,<2.16`) |
| Summary feature schema differs from expected | Hour 1–2 inspects `summary_norm.npy`; mismatch surfaces immediately |
| Google Drive `.npy` download fails | Fall back to ad-hoc `StandardScaler`; predictions approximate but server functional |

---

## Why 12 hours is realistic

- `init_battery.py` is 95% `init_asset_data.py` — **30 min**
- `couchdb_client.py` is 95% `vibration/couchdb_client.py` — **10 min**
- `model_wrapper.py` is 90% `predict.py` from the cloned repo — **60 min**
- `diagnose_battery` LLM pattern is 100% `fmsr/main.py` — **1 hour**
- 8 tools averaging 20 lines each backed by a precomputed cache — **2 hours**
- Validation test is 20 lines — **30 min**
- 15 scenarios are mostly a loop with planner-routing tuning — **2 hours**
- pytest suite mirrors vibration/tests — **1 hour**
- README is a template fill-in — **30 min**

Key insight: **the weights, raw data, preprocessing code, CouchDB pattern, FastMCP pattern, and LLM pattern all already exist**. The only genuinely new code is glue — maybe 300 lines out of 750 total.
