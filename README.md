# Uniq-Cluster Memory

Uniq-Cluster Memory (UCM) is a research system for **conflict-aware temporal memory construction** from long medical dialogues.

Given a multi-turn patient-doctor dialogue, UCM extracts structured medical facts, groups them into information bundles, grounds relative time expressions, detects value conflicts with bi-temporal tracking, and outputs a set of unique canonical memories with full provenance and conflict history.

## Key Results (Med-LongMem v0.1, n=20)

| System | Unique-F1(S) | Unique-F1(R) | Conflict-F1 |
|--------|-------------|-------------|-------------|
| **UCM (ours)** | **0.8508** | **0.8585** | **0.9762** |
| Long-Context LLM | 0.3848 | 0.5273 | 0.8867 |
| Graphiti (simulated) | 0.0627 | 0.5622 | 0.3450 |
| No Memory | 0.0000 | 0.5938 | 0.1292 |

## Algorithmic Contributions

1. **Bi-Temporal Conflict Graph**: Each memory carries four timestamps (t_event, t_ingest, t_valid_start, t_valid_end) with confidence-weighted multi-candidate conflict resolution instead of binary winner selection. Ref: Zep/Graphiti (arXiv 2025.01), EvoKG (arXiv 2025.09).

2. **Causal Deconfounded Information Bundling**: Structural causal model with backdoor adjustment to prevent spurious event coreference caused by lexical overlap, temporal proximity, and speaker identity confounders. Ref: Causal Graph ECR (Scientific Reports 2025).

3. **Medical Domain Formal Constraints**: Rule-based temporal logic constraints (medication start/stop, diagnosis-test ordering, dose monotonicity) that adjust candidate confidence without LLM calls. Ref: ALICE (ASE 2024).

## Pipeline

```
Dialogue → M1 (Event Extraction) → M2 (Attribute Clustering)
         → M2.5 (Information Bundle Construction + Causal Deconfounding)
         → M3 (Time Grounding + Bi-Temporal Conflict Resolution + Formal Constraints)
         → M4 (Compression) → M5 (Hybrid Retrieval)
```

Core entry: `src/uniq_cluster_memory/pipeline.py`

## Repository Layout

```text
uniq_cluster_memory/
├── baselines/              baseline systems (long_context_llm, graphiti, hybrid_rag, recursive_summary)
├── benchmarks/             dataset loaders (med_longmem, meddialog, meditod, longmemeval)
├── evaluation/             metrics (uniqueness, conflict, temporal, extraction, llm_judge, error_analysis)
├── experiments/            experiment runners (eval_our_method, run_ablation, eval_extraction)
├── scripts/                utilities (generate_med_longmem, build_realworld_validation, etc.)
├── src/uniq_cluster_memory/
│   ├── m1_event_extraction/    LLM-based sliding window extractor
│   ├── m2_clustering/          attribute clustering + causal_scorer
│   ├── m3_uniqueness/          time_grounder + conflict_detector + formal_constraints + manager
│   ├── m4_compression/         lightweight compression
│   ├── m5_retrieval/           hybrid struct+semantic+recency retrieval
│   ├── schema.py               CanonicalMemory, CandidateValue, ConflictRecord
│   └── pipeline.py             orchestrator
└── tests/                  88 unit and regression tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set one LLM key:

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
# or QWEN_API_KEY / OPENAI_API_KEY
```

## Datasets

- `Med-LongMem v1.0`: 200 samples (40 Easy / 80 Medium / 80 Hard), GT-first synthetic benchmark
- `Med-LongMem v0.1`: 20 Hard samples (validation release)
- `MedDialog`: real-world external validation (silver GT + audit)

Generate Med-LongMem v1.0:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_med_longmem.py \
  --n_samples 200 --difficulty mix --seed 100 \
  --output_dir data/raw/med_longmem_v1
```

## Main Commands

### Run Full Evaluation

```bash
PYTHONPATH=. .venv/bin/python experiments/eval_our_method.py \
  --data_path data/raw/med_longmem \
  --output_path results/main_results/our_method_eval.json
```

### Run Ablation Experiments (8 variants)

```bash
PYTHONPATH=. .venv/bin/python experiments/run_ablation.py \
  --ablation all --max_samples 20
```

Ablation variants: `full`, `w/o_time`, `w/o_conflict`, `w/o_m4`, `w/o_m2`, `w/o_bitemporal`, `w/o_formal_constraints`, `w/o_causal_deconfound`

### Run Baselines

```bash
# Long-Context LLM
PYTHONPATH=. .venv/bin/python baselines/long_context_llm.py \
  --data_path data/raw/med_longmem

# Graphiti (simulated)
PYTHONPATH=. .venv/bin/python baselines/graphiti_baseline.py \
  --data_path data/raw/med_longmem
```

### Run LLM-as-Judge Evaluation

```bash
PYTHONPATH=. .venv/bin/python evaluation/llm_judge_eval.py \
  --data_path data/raw/med_longmem --max_samples 20
```

### Run Error Analysis

```bash
PYTHONPATH=. .venv/bin/python evaluation/error_analysis.py \
  --data_path data/raw/med_longmem --max_samples 20
```

### Run Tests

```bash
.venv/bin/python -m pytest -q
```

## Ablation Results (Med-LongMem v0.1)

| Variant | U-F1(S) | Δ | C-F1 |
|---------|---------|---|------|
| full | 0.8508 | — | 0.9762 |
| w/o_time | 0.1233 | -85.5% | 0.2583 |
| w/o_m2 | 0.5680 | -33.2% | 0.0000 |
| w/o_causal_deconfound | 0.7700 | -9.5% | 0.8533 |
| w/o_formal_constraints | 0.7900 | -7.1% | 0.9233 |
| w/o_m4 | 0.8349 | -1.9% | 0.9662 |
