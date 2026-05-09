# HPML Final Project: AgentOpsBench: High-Throughput Agentic AI for Battery Analytics

> Please read this first. Our project, after discussion with our IBM mentor, ended up straying away from the typical project format for this class. It has two aspects.
>
> 1. An agentic MCP server implementation for battery analytics (the agent / application side).
> 2. Performance optimizations on top of that server (the HPC side).
>
> Because of this hybrid scope, we follow the HPML template as closely as possible, but there are deviations. Some things that matter to our work do not exist in the template, and some template sections do not apply to us.
>
> A second reason for the deviations is that we wanted to keep the standards and structure of AssetOpsBench intact. We did not want to restructure the upstream repository simply to fit the HPML submission format. We adopted template conventions where they were cheap to adopt, but we deliberately did not fundamentally change the AssetOpsBench repository layout.

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui
> **IBM Mentor:** Dhval Patel, Senior Technical Staff Member (STSM), IBM

---

## Team Information

- **Team Name:** Siuuu (AssetOpsBench Team 7)
- **Members:**
  - Siddharth Gowda (scg2178) - MCP server architecture, tool composition, batching optimization, CouchDB integration
  - Rushin Bhatt (rsb2213) - CouchDB telemetry layer, data pipeline, parallel fetch optimization, profiling harness
  - Aryaman Agrawal (aa5775) - DNN model integration, tf_keras compatibility layer, disk cache implementation
  - Winston Li (wl3062) - Scenario design (15 scenarios), benchmark evaluation, results analysis, documentation

## Submission

- **GitHub repository:** [https://github.com/siddharthgowda/AssetOpsBench](https://github.com/siddharthgowda/AssetOpsBench)
- **Original GitHub repository:** [https://github.com/IBM/AssetOpsBench](https://github.com/IBM/AssetOpsBench)
- **Final report:** `[deliverables/report.pdf](deliverables/report.pdf)`
- **Final presentation:** `[deliverables/presentation.pdf](deliverables/presentation.pdf)` (`[.pptx](deliverables/presentation.pptx)`)
- **Experiment-tracking dashboard:** none. JSON results from our custom wall-clock profiling are committed in `[results/](results/)`.

The final report PDF and the presentation file are checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

AssetOpsBench, which is IBM's MCP-based agent benchmark for industrial asset operations, currently only covers HVAC equipment such as chillers. This project extends it to battery systems (scoped only to lithium-ion batteries) by adding a new MCP server with 10 battery analytics tools, a pretrained DNN prognostics pipeline for RUL (remaining useful life) prediction. We used the NASA Prognostics Center cycling dataset and created a 15-scenario for wall-clock evaluation for high performance computing. The constraint is that introducing a heavier asset class must not bottle the overall system latency, so a meaningful portion of this work is targeted performance optimization at the tool, batching, and caching layers.

---

## 2. Model/Application Description

- **Model architecture:** acctouhou DNN battery prognostics pipeline (4 frozen pretrained Keras models, 2 feature selectors, RUL predictor, voltage predictor). The agentic planner / executor uses `cerebras/llama3.1-8b` (originally `watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8`).
- **Framework:** TensorFlow 2.21, `tf_keras`, FastMCP, LiteLLM, AssetOpsBench `plan-execute`.
- **Dataset:** NASA Prognostics Center B0xx Li-ion cycling dataset (NCA 18650, US Public Domain). Upstream: [https://data.nasa.gov/dataset/li-ion-battery-aging-datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets). Pre-staged copy (Columbia LionMail only): [https://drive.google.com/drive/folders/1nV3nh7WHR0k2WfFvfyzT_K24OOYAbpOP?usp=sharing](https://drive.google.com/drive/folders/1nV3nh7WHR0k2WfFvfyzT_K24OOYAbpOP?usp=sharing). Upstream DNN model: [https://github.com/acctouhou/Prediction_of_battery](https://github.com/acctouhou/Prediction_of_battery).
- **Custom layers or modifications:** battery MCP server (`src/servers/battery/`) is the primary contribution, plus light edits to the AssetOpsBench planner / executor system prompts.
- **Hardware target:** Apple M4 Max CPU, 32 GB RAM, no GPU.

---

## 3. Final Results Summary

The template's accuracy and training-time rows do not apply (see Section 5.D). We optimized wall-clock latency. The LLM we used has a 32K context window, which long scenarios overflow, so we benchmark only what we built (the battery tools). Both tables below exclude any LLM round-trip and map to committed JSONs in `[results/submission_profile_benchmark_results/](results/submission_profile_benchmark_results/)`.

Profiling showed that with a small LLM through Cerebras, wall-clock time is dominated by the tool calls (especially RUL prediction), not the planning step. We optimized the tool layer for that reason and ran the ablations below to find the fastest configuration.

#### Optimization ablation, RUL pipeline

Each row adds one optimization on top of the previous configuration.


| Configuration      | Wall (s) | Fetch (s) | Predict (s) |
| ------------------ | -------- | --------- | ----------- |
| Naive baseline     | 12.64    | 5.655     | 6.985       |
| + Parallel fetch   | 11.18    | 3.852     | 7.325       |
| + Graph precompile | 10.88    | 3.859     | 7.025       |
| + Batched predict  | 10.93    | 3.833     | 7.099       |
| + Disk cache       | 3.83     | 3.831     | 0.002       |


All optimizations on gives a 3.30× wall-time speedup over the naive baseline. Reproducible via `benchmark_optimizations.py` (Section 5.E).

#### MCP-level batching

Per-cell vs batched MCP tool calls (one call covers all 10 cells). Includes FastMCP subprocess spawn, excludes the LLM.


| Strategy                          | Wall (s) | Subprocess spawns | Speedup  |
| --------------------------------- | -------- | ----------------- | -------- |
| Per-cell (10 separate tool calls) | 44.64    | 10                | baseline |
| Batched (1 tool call)             | 7.37     | 1                 | 6.06×    |


Reproducible via `mcp_batch_demo.py` (Section 5.D).

**Hardware:** Apple M4 Max CPU, 32 GB RAM (no GPU). LLM used for benchmarking is `cerebras/llama3.1-8b` (see Section 5.D). Numbers are hardware-dependent and will vary on different machines.

**Headline result (one sentence):** *Layering parallel CouchDB fetch, compiled flexible-shape TF graphs, batched predict, and a disk cache reduces a 10-cell RUL pipeline from 12.64 s to 3.83 s, a 3.30× wall-time speedup end-to-end versus the naive baseline.*

---

## 4. Repository Structure

We did not follow the template's exact layout because we built on top of AssetOpsBench, which has its own established structure (see Section 1). AssetOpsBench is also large, so the tree below highlights the parts most relevant to this project rather than every file.

```
.
├── README.md
├── LICENSE                                # Apache 2.0
├── pyproject.toml                         # uv-managed dependencies (no requirements.txt)
├── .env.public                            # template env vars (copy to .env)
├── docker-compose.couchdb.yml             # CouchDB container
├── deliverables/                          # final report and presentation
├── results/
│   └── submission_profile_benchmark_results/   # wall-clock profiling JSONs
├── src/
│   ├── servers/
│   │   ├── battery/                       # primary contribution: 11-tool battery MCP server
│   │   │   ├── main.py                    # tool registration
│   │   │   ├── model_wrapper.py           # TF loader, compiled graphs, disk cache
│   │   │   ├── preprocessing.py           # NASA JSON to tensor
│   │   │   ├── couchdb_client.py
│   │   │   ├── diagnosis.py               # LLM-narrated failure-mode classifier
│   │   │   ├── profiling/                 # benchmark_optimizations, mcp_batch_demo, profile_scenario
│   │   │   ├── tests/
│   │   │   └── artifacts/                 # gitignored: model weights, norms, .npz cache
│   │   └── fmsr/                          # taxonomy extended with 5 battery failure modes
│   ├── agent/plan_execute/                # planner, executor, runner
│   ├── couchdb/init_battery.py            # NASA JSON to CouchDB loader
│   └── scenarios/local/battery_utterances.json   # 15 evaluation scenarios
└── external/battery/nasa/                 # gitignored: NASA cycling data
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

AssetOpsBench is uv-managed we run it CPU-only on Apple Silicon, and it depends on Docker-hosted CouchDB plus manually-downloaded model artifacts and NASA data.

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker, and a Columbia LionMail (`@columbia.edu`) account for the Step 5 download.

#### 1. Install uv

```bash
brew install uv
```

#### 2. Install dependencies

```bash
uv sync --group battery
```

`--group battery` is required. Plain `uv sync` omits `tensorflow` and `tf_keras`, so the battery server fails on import.

#### 3. Configure `.env`

```bash
cp .env.public .env
```

Fill in LLM credentials. Pick one of the two options below.

**Option A.** Any LiteLLM-supported provider (Cerebras, OpenAI, Anthropic, Cohere, Groq, etc.).

```dotenv
LITELLM_API_KEY=<your LiteLLM proxy or provider key>
LITELLM_BASE_URL=<your LiteLLM proxy URL>
```

Invoke with `--model-id "cerebras/llama3.1-8b"` (swap the prefix for other providers).

**Option B.** IBM watsonx.

```dotenv
WATSONX_APIKEY=<your watsonx API key>
WATSONX_PROJECT_ID=<your watsonx project ID>
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

Invoke with e.g. `--model-id "watsonx/meta-llama/llama-3-3-70b-instruct"`.

#### 4. Create the artifact / data directories

```bash
mkdir -p src/servers/battery/artifacts/weights \
         src/servers/battery/artifacts/norms \
         src/servers/battery/artifacts/cache \
         external/battery/nasa
```

#### 5. Download the model + NASA cycling data

> Access restricted to Columbia LionMail (`@columbia.edu`).

[https://drive.google.com/drive/folders/1nV3nh7WHR0k2WfFvfyzT_K24OOYAbpOP?usp=sharing](https://drive.google.com/drive/folders/1nV3nh7WHR0k2WfFvfyzT_K24OOYAbpOP?usp=sharing)

The Drive has two folders.

- NASA cycling data (`B*.json` files), move all into `external/battery/nasa/`.
- Model artifacts (4 `.h5` weights and 4 `.npy` norms). The `.h5` files go in `src/servers/battery/artifacts/weights/`, and the `.npy` files go in `src/servers/battery/artifacts/norms/`.

#### 6. Start CouchDB

```bash
docker compose -f docker-compose.couchdb.yml up -d
```

#### 7. Load the battery dataset

```bash
uv run python -m couchdb.init_battery --drop
```

#### 8. Smoke test, boot the server (optional)

```bash
uv run battery-mcp-server
```

> If the command runs without an error (warnings are fine), it is working. The server uses stdio and waits idle for a client. `Ctrl-C` to exit.

#### 9. Run a sample scenario

```bash
uv run plan-execute \
  "Predict the remaining useful life in cycles for cells B0005, B0006, B0007, B0018, B0033, B0034, B0036, B0054, B0055, and B0056. Rank them from worst to best by remaining cycles, and tell me which 3 cells are closest to end-of-life." \
  --model-id "cerebras/llama3.1-8b"
```

### B. Experiment Tracking Dashboard

There is no public experiment-tracking dashboard. We did not use Wandb, MLflow, Comet, Neptune, or TensorBoard. The acctouhou prognostics model runs CPU-only and we do not train an LLM, so the metrics we care about are wall-clock latency and end-to-end correctness. The measured numbers live as JSON under `[results/submission_profile_benchmark_results/](results/submission_profile_benchmark_results/)`, and the three profiling commands in Section 5.D regenerate equivalent runs.

### C. Dataset

The dataset is the NASA Prognostics Center B0xx Li-ion battery cycling dataset (NCA 18650 chemistry, US Public Domain). The upstream source is [https://data.nasa.gov/dataset/li-ion-battery-aging-datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets), which distributes the data in MATLAB `.mat` format. Our Google Drive copy holds the same data converted to per-cell `.json` files, which is what the loader expects. The dataset is not committed to this repository. The setup steps below repeat Section 5.A Steps 4 and 5 so the template's Dataset section stands on its own.

#### 1. Create the data directory

```bash
mkdir -p external/battery/nasa
```

This matches the `BATTERY_DATA_DIR` default in `.env.public`.

#### 2. Download from Google Drive

> Access restricted to Columbia LionMail (`@columbia.edu`) accounts.

[https://drive.google.com/drive/folders/1nV3nh7WHR0k2WfFvfyzT_K24OOYAbpOP?usp=sharing](https://drive.google.com/drive/folders/1nV3nh7WHR0k2WfFvfyzT_K24OOYAbpOP?usp=sharing)

The Drive contains two folders. The NASA cycling data folder holds a flat collection of `B*.json` files. Move all of them into `external/battery/nasa/`. (The other folder holds the pretrained model artifacts. See Section 5.A Step 5 for those instructions.)

#### 3. Load into CouchDB

```bash
uv run python -m couchdb.init_battery --drop
```

Reads `external/battery/nasa/` and writes the cells listed in `BATTERY_CELL_SUBSET` (the default is 14 cells).

### D. Evaluation, Profiling, and Limitations

We do not train a model and we do not report a task-accuracy metric. What we performed is system-level performance benchmarking of the optimizations layered on top of the frozen pretrained model.

#### No training

The model is the frozen pretrained acctouhou Keras pipeline (4 `.h5` weights and 4 `.npy` norms). We do not train or fine-tune it. Everything in this project is on the inference side.

#### No model-accuracy metric

We do not report MAE or RMSE on RUL, voltage curves, or impedance. The acctouhou model was trained on the Severson LFP dataset, while NASA B0xx cells are NCA 18650. Both are lithium-ion, but the cathode chemistries differ, so predicted absolute values are off relative to the NASA ground truth. Fine-tuning on NASA cycling data would close that gap. Fine-tuning was out of scope, and we recommend it as future work.

The system-performance benchmarks below are unaffected. They measure wall-clock latency, not output accuracy.

#### Performance benchmarks (three profiling commands)

There are three distinct profiling entry points. Each writes JSON to `src/servers/battery/profiles/`. The submission-run outputs are committed under `results/submission_profile_benchmark_results/`.

**1. Optimization ablation.** 5-rung ladder, in-process (no MCP, no LLM). Measures the cumulative effect of parallel CouchDB fetch, compiled TF graphs, batched predict, and disk cache.

```bash
uv run python -m servers.battery.profiling.benchmark_optimizations --repeats 3
```

**2. MCP-level batch demo.** Compares N per-cell `predict_rul` MCP calls against one `predict_rul_batch` MCP call. No planner, no LLM, just a direct `_call_tool` invocation. Isolates the subprocess-spawn savings of collapsing tool calls.

```bash
uv run python -m servers.battery.profiling.mcp_batch_demo
```

**3. Scenario timing (optimizations disabled).** Runs all 11 in-context-window scenarios end-to-end with every optimization toggle off. We used this to identify the bottlenecks that the other two commands target.

```bash
BATTERY_PARALLEL_FETCH=0 BATTERY_GRAPH_PRECOMPILE=0 BATTERY_BATCHED_PREDICT=0 BATTERY_DISK_CACHE=0 \
  uv run python -m servers.battery.profiling.profile_scenario \
    --scenarios 1,2,4,5,6,7,8,9,10,12,13 \
    --model-id "cerebras/llama3.1-8b"
```

#### Reproduction model

Use `cerebras/llama3.1-8b` for all three benchmarks. That is the model every number in `results/submission_profile_benchmark_results/` was measured against. (We originally used `watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8` and switched to Cerebras after the watsonx API key was invalidated.)

#### End-to-end correctness check

```bash
uv run pytest src/servers/battery/tests/
```

Validates the preprocessing math without needing TF, CouchDB, or an LLM. The sample `plan-execute` query in Section 5.A serves as the live integration test.

### E. Quickstart: Reproduce the Optimization Ablation

After Section 5.A is complete, run the command below to regenerate the 5-rung ablation in Section 3 (≈ 5 minutes on Apple M4 Max CPU).

```bash
uv run python -m servers.battery.profiling.benchmark_optimizations --repeats 3
```

Writes one JSON per rung under `src/servers/battery/profiles/benchmark_<ts>/`. The aggregated reference is [`results/submission_profile_benchmark_results/benchmark_2026-05-06T030016Z/summary.json`](results/submission_profile_benchmark_results/benchmark_2026-05-06T030016Z/summary.json). Numbers should match the ablation table to within run-to-run variance.

---

## 6. Results and Observations

No embedded figure. We did not use Wandb, MLflow, or TensorBoard. The raw timing JSONs in `[results/submission_profile_benchmark_results/](results/submission_profile_benchmark_results/)` and the tables in Section 3 carry the same information.

- Even a small LLM was able to plan and execute battery operations end-to-end, much like one would plan operations for a data center. Context-window limits constrain the longer queries that touch many cells at once.
- MCP-level batching was the largest single speedup we observed (6.06×, from roughly 45 s to 7 s on a 10-cell query). Collapsing many per-cell tool calls into one batched call beats fanning out, because each call spawns a fresh subprocess that pays the framework setup and model-load cost. On CPU with a small LLM, this overhead matters more than the LLM call itself.
- A disk cache pays off whenever you have an expensive recurring computation, such as running a neural network on CPU. On a warm cache the model inference disappears, which gave us a 2.85× speedup on that rung.
- A bigger and more diverse benchmark suite would have made the cache analysis more useful. Battery is a new domain inside AssetOpsBench, so we authored only 15 scenarios ourselves, of which 11 completed cleanly under the small LLM (the failures came from limited context length). The HVAC side of AssetOpsBench has been filled in by many contributors over time and now holds hundreds of scenarios, which gives a realistic weekly-to-monthly workflow signal and lets you measure the amortized cache-reuse rate. We did not get to that scale.
- Pre-compiling the model graph at boot did not help much (about 3%), most likely because the underlying model is small enough that graph re-compile is not a large fraction of total inference time on CPU.
- Batching the model call across cells inside the MCP server did not help on CPU (within run-to-run noise). The model is small enough that the work scales linearly with batch size on CPU, so there is nothing left to amortize.

---

## 7. Notes

- Source code lives under `src/`. The battery server (`src/servers/battery/`) is the primary new addition. We extended `fmsr` with a five-mode battery failure-mode taxonomy and left the other upstream servers (`iot`, `tsfm`, `wo`, `vibration`, `utilities`) unchanged.
- We do not train. The pretrained acctouhou model artifacts (four `.h5` weights, four `.npy` norms) are gitignored under `src/servers/battery/artifacts/`. Download them per Section 5.A Step 5.
- API keys and CouchDB credentials load from environment variables. Copy `.env.public` to `.env` and fill in values per Section 5.A Step 3.

### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*

**Did your team use any AI tool in completing this project?**

- No, we did not use any AI tool.
- Yes, we used AI assistance as described below.

**Tool(s) used:** Claude, Cursor.

**Specific purpose:** We used Claude to edit and compress prose in the report (abstract, methodology, discussion) and to generate the LaTeX system architecture diagram. We used Cursor to generate preprocessing code (`src/servers/battery/preprocessing.py`) using the original model repository ([https://github.com/acctouhou/Prediction_of_battery/tree/main](https://github.com/acctouhou/Prediction_of_battery/tree/main)) as context, and to write docstrings for the Battery MCP server and profiling modules (`src/servers/battery/main.py`, `benchmark_optimizations.py`, `mcp_batch_demo.py`, `profile_scenario.py`). We also used Cursor to help condense our writing in the README and help create the repository structure in the README since AssetOpsBench is quite large.

**Sections affected:** Report abstract, methodology, discussion, appendix architecture diagram, README. Code files listed above.

**How we verified correctness:** We re-read all edited prose and rewrote anything that did not match our intent. We reviewed all generated code against our own understanding of the system.

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

Released under the Apache License 2.0. See `[LICENSE](LICENSE)`.

### Citation

If you build on this work, please cite:

```bibtex
@misc{SiuuuAssetOpsBenchTeam7,
  title  = {AgentOpsBench: High-Throughput Agentic AI for Battery Analytics},
  author = {Gowda, Siddharth and Bhatt, Rushin and Agrawal, Aryaman and Li, Winston},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/siddharthgowda/AssetOpsBench}
}
```

### Contact

Open a GitHub Issue or email *[[scg2178@colubmia.edu](mailto:scg2178@colubmia.edu)]*.

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*