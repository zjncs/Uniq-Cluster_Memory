# Uniq-Cluster Memory: A Research Framework

This repository contains the code and experimental framework for the **Uniq-Cluster Memory** research project. The goal of this project is to develop and evaluate a novel memory management system for large language models, focusing on reliability, consistency, and conflict detection in long-term medical dialogues.

This framework was bootstrapped and implemented by **Manus AI** based on the detailed research plan provided by the user.

## Project Structure

The project is organized into a modular structure to facilitate research, development, and experimentation.

```
uniq_cluster_memory/
├── configs/              # Core strategy and weight configurations
├── data/
│   ├── raw/              # Raw datasets (MedDialog, LongMemEval)
│   ├── gt/               # Ground truth data (to be generated)
│   └── processed/        # Processed data (e.g., pre-built FAISS indexes)
├── benchmarks/           # Unified data loaders for all datasets
├── baselines/            # Implementations of all baseline models
├── evaluation/           # Scripts for all evaluation metrics (Recall, QA, F1)
├── experiments/          # Main experiment, ablation, and sensitivity analysis runners
├── scripts/              # Utility scripts (e.g., data generation, index building)
├── src/
│   └── uniq_cluster_memory/ # Source code for our proposed model (M1-M5)
├── tests/                # Unit and integration tests
├── results/              # Directory to save all experimental results
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Setup

1.  **Clone the repository** (or unpack the provided zip file).

2.  **Create a Python virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Optional: Persistent Vector/Graph Stores

You can enable persistent memory backends for production-style retrieval and temporal querying:

- **Qdrant** (vector store)
- **Neo4j** (graph store)

Set environment variables before running:

```bash
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY=""
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-neo4j-password"
```

And initialize pipeline with persistence enabled:

```python
from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline

pipeline = UniqueClusterMemoryPipeline(
    enable_qdrant=True,
    enable_neo4j=True,
    persist_to_stores=True,
)
```

4.  **Set up OpenAI API Key**:
    The framework uses OpenAI models for LLM-as-a-Judge and for some baseline models. Ensure your API key is set as an environment variable:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```

5.  **Download Datasets**:
    The framework is configured to use `LongMemEval` and `MedDialog`. The necessary sample files have already been downloaded and placed in the `data/raw/` directory. To run on the full datasets, you would need to download them and place them in the corresponding folders.

## Running the Baseline Experiments (Step 1 Validation)

We have successfully completed **Step 1** of the research plan: running the baseline models on existing data to establish a complete pipeline from data loading to evaluation.

To reproduce the baseline validation run, you can use the main experiment script `experiments/run_main_exp.py`.

**Example: Run the `hybrid_rag` baseline on 3 samples from `LongMemEval`**

```bash
# Ensure you are in the project root directory
cd /path/to/uniq_cluster_memory

# Set the PYTHONPATH
export PYTHONPATH=.

# Run the experiment
python3 experiments/run_main_exp.py \
  --dataset longmemeval \
  --baseline hybrid_rag \
  --max_samples 3 \
  --output_dir results/main_results
```

This command will:
1.  Load 3 samples from the `LongMemEval` dataset using the `LongMemEvalTask` loader.
2.  Run the `HybridRAGBaseline` on each sample.
3.  Evaluate the generated answers using `LLM-as-a-Judge` (`qa_judge_eval.py`).
4.  Evaluate the retrieval quality using `Recall@K` (`retrieval_eval.py`).
5.  Save the detailed outputs and a summary report to `results/main_results/`.

## Next Steps

With the framework and baseline pipeline now firmly established and validated, the project is ready to proceed to the next steps as outlined in your plan:

*   **Step 2: Solidify `Med-LongMem` Ground Truth**: Based on the now-fixed data interfaces and evaluation logic, we can finalize the exact format for the ground truth data (`event_table_gt`, `unique_key_gt`, `conflict_log_gt`) that our custom dataset will contain.

*   **Step 3: Implement `Med-LongMem` Generation Script**: With the GT format defined, we can now confidently implement the `scripts/generate_med_longmem.py` script to create our novel adversarial benchmark.

*   **Implement Our Model (M1-M5)**: Begin implementing the core modules of the Uniq-Cluster Memory system within the `src/uniq_cluster_memory/` directory.
