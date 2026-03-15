from experiments.eval_our_method import select_comparison_baselines


def test_select_comparison_baselines_excludes_oracle_rows():
    baselines = [
        {"system": "GT_Upper_Bound", "unique_f1": 1.0},
        {"system": "Raw_RAG", "unique_f1": 0.2},
        {"system": "Oracle_v2", "unique_f1": 0.9},
        {"system": "No_Memory", "unique_f1": 0.0},
    ]

    filtered = select_comparison_baselines(baselines)

    assert [item["system"] for item in filtered] == ["Raw_RAG", "No_Memory"]
