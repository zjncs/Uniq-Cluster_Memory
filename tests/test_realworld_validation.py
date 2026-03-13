import json

from scripts.build_realworld_validation import (
    build_audit_subset,
    load_existing_silver_records,
    parse_utterance,
    record_to_silver_payload,
    select_long_dialogues,
    unresolved_error_records,
    utterances_to_dialogue,
    validate_resume_selection,
)
from src.uniq_cluster_memory.schema import CanonicalMemory


def test_parse_utterance_detects_chinese_prefixes():
    role, text = parse_utterance("病人：最近一直头痛。", "doctor")
    assert role == "patient"
    assert text == "最近一直头痛。"

    role, text = parse_utterance("医生：建议继续复诊。", "patient")
    assert role == "doctor"
    assert text == "建议继续复诊。"


def test_utterances_to_dialogue_falls_back_to_turn_order():
    dialogue = utterances_to_dialogue(
        [
            "最近一直头晕。",
            "建议先测血压。",
        ]
    )
    assert dialogue[0]["speaker"] == "patient"
    assert dialogue[0]["text"] == "最近一直头晕。"
    assert dialogue[1]["speaker"] == "doctor"
    assert dialogue[1]["text"] == "建议先测血压。"


def test_select_long_dialogues_filters_by_turn_count():
    selected = select_long_dialogues(
        dialogues=[
            ["a"] * 2,
            ["b"] * 10,
            ["c"] * 12,
        ],
        min_turns=10,
        n_samples=5,
        seed=7,
    )
    assert len(selected) == 2
    assert {item["raw_index"] for item in selected} == {1, 2}
    assert all(item["n_turns"] >= 10 for item in selected)


def test_build_audit_subset_uses_ratio_and_is_non_empty():
    records = [
        {"sample_id": f"s{i:02d}"}
        for i in range(10)
    ]
    audit = build_audit_subset(records, audit_ratio=0.2, seed=42)
    assert len(audit) == 2
    assert audit[0]["sample_id"] < audit[1]["sample_id"]


def test_record_to_silver_payload_roundtrip(tmp_path):
    record = {
        "sample_id": "meddialog_real_0000001",
        "raw_index": 1,
        "source_split": "processed.zh.test",
        "n_turns": 10,
        "utterances": ["病人：头痛", "医生：复诊"],
        "dialogue": [
            {"turn_id": 0, "speaker": "patient", "text": "头痛"},
            {"turn_id": 1, "speaker": "doctor", "text": "复诊"},
        ],
        "memories": [
            CanonicalMemory(
                patient_id="meddialog_real_0000001",
                attribute="symptom",
                value="头痛",
                time_scope="global",
            )
        ],
        "n_predicted": 1,
        "n_conflicts": 0,
        "latency_sec": 1.23,
    }
    silver_path = tmp_path / "silver_gt.jsonl"
    silver_path.write_text(json.dumps(record_to_silver_payload(record), ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = load_existing_silver_records(silver_path)
    restored = loaded["meddialog_real_0000001"]
    assert restored["utterances"] == record["utterances"]
    assert restored["dialogue"] == record["dialogue"]
    assert restored["n_predicted"] == 1
    assert restored["memories"][0].value == "头痛"


def test_validate_resume_selection_rejects_mismatched_sample_ids(tmp_path):
    selected_path = tmp_path / "selected_dialogues.jsonl"
    selected_path.write_text(
        json.dumps({"sample_id": "meddialog_real_0000001"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        validate_resume_selection(
            selected_path,
            [{"sample_id": "meddialog_real_0000002"}],
        )
    except RuntimeError as exc:
        assert "Resume target does not match" in str(exc)
    else:
        raise AssertionError("Expected validate_resume_selection to reject mismatched sample ids.")


def test_unresolved_error_records_ignores_completed_samples():
    errors = [
        {"sample_id": "s1", "message": "timeout"},
        {"sample_id": "s2", "message": "timeout"},
        {"sample_id": "s2", "message": "retry timeout"},
    ]
    unresolved = unresolved_error_records(errors, completed_ids={"s2"})
    assert unresolved == [{"sample_id": "s1", "message": "timeout"}]
