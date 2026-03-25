# Uniq-Cluster Memory (UCM)

> A Conflict-Aware Temporal Memory Management System for Medical AI Agents

Uniq-Cluster Memory (UCM) is a research-oriented system designed for **conflict-aware temporal memory construction** from long, multi-turn medical dialogues. 

In real-world medical consultations, patient states (e.g., medications, symptoms, test results) are frequently updated, refined, or contradicted over time. Pure context-window extension or standard Retrieval-Augmented Generation (RAG) approaches struggle to maintain a consistent, authoritative view of the patient's state. UCM addresses this by extracting structured medical facts, grouping them into causal information bundles, grounding relative time expressions, and explicitly detecting value conflicts using bi-temporal tracking. The system ultimately outputs a set of unique, canonical memories complete with full provenance and conflict histories.

---

## 🌟 Key Algorithmic Contributions

1. **Bi-Temporal Conflict Graph**: 
   Unlike traditional binary winner-takes-all memory updates, each memory in UCM carries four distinct timestamps (`t_event`, `t_ingest`, `t_valid_start`, `t_valid_end`). We introduce a confidence-weighted, multi-candidate conflict resolution mechanism that preserves historical context and resolves temporal inconsistencies.
   
2. **Causal Deconfounded Information Bundling**: 
   We employ a structural causal model with backdoor adjustment to mitigate spurious event coreferences. This effectively handles confounders common in medical texts, such as lexical overlap, temporal proximity, and speaker identity bias.
   
3. **Medical Domain Formal Constraints**: 
   UCM integrates rule-based temporal logic constraints (e.g., medication start/stop logic, diagnosis-test ordering, dose monotonicity). These constraints adjust candidate confidence scores deterministically, enhancing accuracy without incurring additional LLM inference costs.

---

## 📊 Experimental Results

UCM demonstrates state-of-the-art performance in maintaining unique, consistent memories compared to strong baselines. 

**Main Results on Med-LongMem v0.1 (n=20, Hard Difficulty)**

| System | Unique-F1 (Strict) | Unique-F1 (Relaxed) | Conflict-F1 |
|:---|:---:|:---:|:---:|
| **UCM (Ours)** | **0.8508** | **0.8585** | **0.9762** |
| Long-Context LLM | 0.3848 | 0.5273 | 0.8867 |
| Graphiti (Simulated) | 0.0627 | 0.5622 | 0.3450 |
| No Memory | 0.0000 | 0.5938 | 0.1292 |

**Ablation Study**

| Variant | Unique-F1 (Strict) | Performance Drop (Δ) | Conflict-F1 |
|:---|:---:|:---:|:---:|
| **Full UCM Pipeline** | **0.8508** | — | **0.9762** |
| w/o Time Grounding | 0.1233 | -85.5% | 0.2583 |
| w/o M2 Clustering | 0.5680 | -33.2% | 0.0000 |
| w/o Causal Deconfounding | 0.7700 | -9.5% | 0.8533 |
| w/o Formal Constraints | 0.7900 | -7.1% | 0.9233 |
| w/o M4 Compression | 0.8349 | -1.9% | 0.9662 |

---

## ⚙️ System Architecture Pipeline

![UCM System Architecture](assets/ucm_pipeline.png)

UCM operates through a 5-stage pipeline:

```text
Raw Dialogue 
  ↓
[M1] Event Extraction (LLM-based sliding window)
  ↓
[M2] Attribute Clustering (Semantic grouping)
  ↓
[M2.5] Information Bundle Construction & Causal Deconfounding
  ↓
[M3] Uniqueness Management (Time Grounding + Bi-Temporal Conflict Resolution + Formal Constraints)
  ↓
[M4] Memory Compression (Lightweight redundancy removal)
  ↓
[M5] Hybrid Retrieval (Structured + Semantic + Recency)
```
*Core entry point:* `src/uniq_cluster_memory/pipeline.py`

---

## 📂 Repository Structure

```text
uniq_cluster_memory/
├── baselines/              # Baseline implementations (Long-Context LLM, Graphiti, Hybrid RAG)
├── benchmarks/             # Dataset loaders (Med-LongMem, MedDialog, etc.)
├── evaluation/             # Evaluation metrics (Uniqueness, Conflict, Extraction, LLM-as-Judge)
├── experiments/            # Experiment execution scripts (Eval, Ablation)
├── scripts/                # Utility scripts (Dataset generation, Validation tools)
├── src/uniq_cluster_memory/# Core source code (M1 to M5 modules, schema, pipeline)
└── tests/                  # Unit and regression test suite (88 tests)
```

---

## 🚀 Quick Start

### 1. Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

Set your preferred LLM API key. UCM supports DashScope (Qwen) and OpenAI-compatible endpoints:

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
# Alternatively: export OPENAI_API_KEY="your-api-key-here"
```

### 3. Datasets Preparation

UCM utilizes the `Med-LongMem` benchmark suite. To generate the full synthetic benchmark (v1.0):

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_med_longmem.py \
  --n_samples 200 \
  --difficulty mix \
  --seed 100 \
  --output_dir data/raw/med_longmem_v1
```

---

## 💻 Running Experiments

### Full Pipeline Evaluation
Run the standard evaluation of the UCM method:

```bash
PYTHONPATH=. .venv/bin/python experiments/eval_our_method.py \
  --data_path data/raw/med_longmem \
  --output_path results/main_results/our_method_eval.json
```

### Ablation Studies
Execute all ablation variants (`full`, `w/o_time`, `w/o_conflict`, `w/o_m4`, `w/o_m2`, `w/o_bitemporal`, `w/o_formal_constraints`, `w/o_causal_deconfound`):

```bash
PYTHONPATH=. .venv/bin/python experiments/run_ablation.py \
  --ablation all \
  --max_samples 20
```

### Baseline Comparisons
Evaluate Long-Context LLM and Graphiti baselines:

```bash
# Long-Context LLM Baseline
PYTHONPATH=. .venv/bin/python baselines/long_context_llm.py \
  --data_path data/raw/med_longmem

# Graphiti (Simulated) Baseline
PYTHONPATH=. .venv/bin/python baselines/graphiti_baseline.py \
  --data_path data/raw/med_longmem
```

### Automated Testing
Run the comprehensive test suite to ensure system integrity:

```bash
.venv/bin/python -m pytest -q
```
