# Uniq-Cluster Memory

Uniq-Cluster Memory (UCM) is a research codebase for **patient-level unique memory construction** from long medical dialogues.

The current project focuses on Task 1:

- extract structured medical facts from multi-turn dialogue
- merge repeated mentions into information bundles
- ground relative time expressions
- maintain unique canonical memories with explicit conflict history

This repository is currently best understood as a **research prototype with quantitative evidence**, not as a production system.

## Current Status

The repository already supports:

- `Med-LongMem` quantitative evaluation for Task 1
- `M1` extraction-only evaluation
- module ablations
- real-world validation on official `MedDialog`

Current supporting docs:

- task-oriented plan: `docs/task1_research_plan.md`
- publication-oriented summary: `docs/task1_publication_summary.md`

## Pipeline

The main UCM pipeline is:

1. `M1` event extraction
2. `M2` attribute clustering
3. `M2.5` information bundle construction
4. `M3` time grounding + uniqueness/conflict management
5. `M4` memory compression
6. `M5` retrieval

Core entry:

- `src/uniq_cluster_memory/pipeline.py`

## Repository Layout

```text
uniq_cluster_memory/
├── baselines/          baseline systems
├── benchmarks/         dataset loaders
├── configs/            retrieval and policy configs
├── docs/               design notes and research summaries
├── evaluation/         metric implementations
├── experiments/        quantitative experiment runners
├── scripts/            utility and validation scripts
├── src/uniq_cluster_memory/
│   ├── m1_event_extraction/
│   ├── m2_clustering/
│   ├── m3_uniqueness/
│   ├── m4_compression/
│   ├── m5_retrieval/
│   └── pipeline.py
└── tests/              unit and regression tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set one LLM key before running experiments:

```bash
export QWEN_API_KEY="your-api-key-here"
# or
export DASHSCOPE_API_KEY="your-api-key-here"
# or
export OPENAI_API_KEY="your-api-key-here"
```

Optional runtime knobs:

```bash
export LLM_TIMEOUT_SECONDS=60
export LLM_MAX_RETRIES=2
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## Datasets

The codebase currently uses:

- `Med-LongMem`: main quantitative benchmark for unique-memory evaluation
- `MedDialog`: real-world external validation
- `LongMemEval`: auxiliary long-memory benchmark support

Expected local data locations:

- `data/raw/med_longmem`
- `data/raw/meddialog`
- `data/raw/meddialog_official`
- `data/raw/longmemeval`

## Main Commands

### 1. Run Task 1 Main Evaluation

```bash
export PYTHONPATH=.
export QWEN_API_KEY="your-api-key-here"

.venv/bin/python experiments/eval_our_method.py \
  --data_path data/raw/med_longmem \
  --output_path results/main_results/our_method_eval.json
```

This evaluates the full UCM pipeline on `Med-LongMem` and reports:

- `Unique-F1(strict)`
- `Unique-F1(relaxed)`
- `Conflict-F1`
- `Attribute Coverage`
- `Redundancy`

### 2. Evaluate M1 Extraction

```bash
export PYTHONPATH=.
export QWEN_API_KEY="your-api-key-here"

.venv/bin/python experiments/eval_extraction.py \
  --data_path data/raw/med_longmem \
  --output_path results/main_results/extraction_eval_med_longmem.json
```

This reports:

- strict event F1: `attribute + value + unit + time_scope`
- relaxed event F1: `attribute + value + unit`
- field-level F1 for `attribute/value/unit/time_scope/speaker`

### 3. Run Ablations

```bash
export PYTHONPATH=.
export QWEN_API_KEY="your-api-key-here"

.venv/bin/python experiments/run_ablation.py \
  --ablation all \
  --max_samples 20
```

Output:

- `results/ablation/ablation_summary.json`

### 4. Build Real-World Validation Package

```bash
export PYTHONPATH=.
export QWEN_API_KEY="your-api-key-here"

.venv/bin/python scripts/build_realworld_validation.py \
  --data_path data/raw/meddialog_official/processed_zh_test.json \
  --n_samples 20 \
  --min_turns 10 \
  --audit_ratio 0.2 \
  --case_count 5 \
  --reference_date 2024-01-01 \
  --output_dir results/real_world_validation/meddialog_official_zh_test_long_r20_s42_ref2024
```

Notes:

- `MedDialog` does not provide gold labels for unique memory
- this script should be used for `silver GT + audit + case study`
- do not use the same-version silver output as headline self-proof

### 5. Run Tests

```bash
.venv/bin/python -m pytest -q
```

## Useful Scripts

- `scripts/run_pipeline.py`: run UCM on a chosen dataset
- `scripts/run_manual_eval_meddialog.py`: build a manual review package for `MedDialog`
- `scripts/generate_med_longmem.py`: generate or extend the synthetic benchmark
- `scripts/preview_bundles.py`: inspect bundle graph outputs on a few samples

## Current Evidence Chain

The current Task 1 evidence is organized as:

1. `M1` extraction quality on `Med-LongMem`
2. full UCM Task 1 metrics on `Med-LongMem`
3. ablations for `time`, `conflict`, `M2`, and `M4`
4. real-world validation on official `MedDialog`

See:

- `docs/task1_publication_summary.md`

## Limitations

- real-world `MedDialog` validation is currently `silver + audit`, not real-world gold evaluation
- `M1` still has long-tail latency on some noisy dialogues
- `M4` is currently a supporting layer, not the strongest contribution of the system

## Recommendation

For reporting or paper writing, the strongest current framing is:

- use `Med-LongMem` as the main quantitative benchmark
- use `MedDialog` as real-world qualitative validation
- emphasize `information bundling + time grounding + conflict history`
