import json
from pathlib import Path

from benchmarks.med_longmem_task import MedLongMemTask


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_med_longmem_infers_question_date_from_canonical_gt(tmp_path: Path):
    root = tmp_path / "med_longmem"
    sample_dir = root / "medlm_test"
    sample_dir.mkdir(parents=True)

    _write_json(
        root / "dataset_summary.json",
        {"samples": [{"dialogue_id": "medlm_test", "difficulty": "hard"}]},
    )

    (sample_dir / "dialogue.jsonl").write_text(
        json.dumps({"turn_id": 0, "speaker": "patient", "text": "I feel dizzy."}) + "\n",
        encoding="utf-8",
    )
    (sample_dir / "canonical_gt.jsonl").write_text(
        json.dumps(
            {
                "patient_id": "medlm_test",
                "attribute": "symptom",
                "value": "dizziness",
                "time_scope": "2023-07-15",
                "update_policy": "append",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    task = MedLongMemTask(data_path=str(root), max_samples=1)
    sample = task.get_samples()[0]

    assert sample.question_date == "2023-07-15"
    assert sample.metadata.get("dialogue_date") == "2023-07-15"


def test_med_longmem_prefers_metadata_dialogue_date(tmp_path: Path):
    root = tmp_path / "med_longmem"
    sample_dir = root / "medlm_test2"
    sample_dir.mkdir(parents=True)

    _write_json(
        root / "dataset_summary.json",
        {"samples": [{"dialogue_id": "medlm_test2", "difficulty": "hard"}]},
    )

    (sample_dir / "dialogue.jsonl").write_text(
        json.dumps({"turn_id": 0, "speaker": "doctor", "text": "Take medicine daily."}) + "\n",
        encoding="utf-8",
    )
    (sample_dir / "canonical_gt.jsonl").write_text(
        json.dumps(
            {
                "patient_id": "medlm_test2",
                "attribute": "medication",
                "value": "metformin 500mg bid",
                "time_scope": "2023-07-15",
                "update_policy": "latest",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        sample_dir / "metadata.json",
        {"dialogue_date": "2023-08-20", "difficulty": "hard"},
    )

    task = MedLongMemTask(data_path=str(root), max_samples=1)
    sample = task.get_samples()[0]

    assert sample.question_date == "2023-08-20"
    assert sample.metadata.get("dialogue_date") == "2023-08-20"
