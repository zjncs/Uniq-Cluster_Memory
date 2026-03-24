"""
tests/test_bitemporal_schema.py
================================
测试双时态字段的填充、序列化和向后兼容性。
"""

from src.uniq_cluster_memory.schema import CanonicalMemory, CandidateValue


def test_bitemporal_fields_auto_derived():
    """双时态字段应从 start_time 和 provenance 自动派生。"""
    mem = CanonicalMemory(
        patient_id="p1",
        attribute="blood_glucose",
        value="7.2",
        unit="mmol/L",
        time_scope="2024-01-15",
        provenance=[3, 5, 7],
    )
    assert mem.t_event == "2024-01-15"
    assert mem.t_ingest == 7
    assert mem.t_valid_start == "2024-01-15"
    assert mem.t_valid_end == "2024-01-15"


def test_bitemporal_fields_none_when_global():
    """global scope 下双时态字段保持 None。"""
    mem = CanonicalMemory(
        patient_id="p1",
        attribute="medication",
        value="Metformin 500mg",
        time_scope="global",
        update_policy="latest",
        provenance=[1, 2],
    )
    assert mem.t_event is None
    assert mem.t_ingest == 2
    assert mem.t_valid_start is None
    assert mem.t_valid_end is None


def test_bitemporal_fields_preserved_in_serialization():
    """双时态字段应正确序列化和反序列化。"""
    mem = CanonicalMemory(
        patient_id="p1",
        attribute="blood_glucose",
        value="7.2",
        time_scope="2024-01-15",
        provenance=[3],
        t_event="2024-01-15",
        t_ingest=3,
        t_valid_start="2024-01-15",
        t_valid_end="2024-01-20",
    )
    d = mem.to_dict()
    assert d["t_event"] == "2024-01-15"
    assert d["t_ingest"] == 3
    assert d["t_valid_start"] == "2024-01-15"
    assert d["t_valid_end"] == "2024-01-20"

    restored = CanonicalMemory.from_dict(d)
    assert restored.t_event == "2024-01-15"
    assert restored.t_ingest == 3
    assert restored.t_valid_start == "2024-01-15"
    assert restored.t_valid_end == "2024-01-20"


def test_backward_compat_without_bitemporal_fields():
    """旧格式数据（无双时态字段）反序列化不应报错。"""
    old_dict = {
        "patient_id": "p1",
        "attribute": "blood_glucose",
        "value": "7.2",
        "unit": "mmol/L",
        "time_scope": "2024-01-15",
    }
    mem = CanonicalMemory.from_dict(old_dict)
    assert mem.patient_id == "p1"
    # 双时态字段应由 __post_init__ 从现有字段派生
    assert mem.t_event == "2024-01-15"


def test_candidate_value_serialization():
    """CandidateValue 序列化和反序列化。"""
    cv = CandidateValue(
        value="7.2",
        unit="mmol/L",
        confidence=0.85,
        provenance=[3, 5],
        speaker="doctor",
        t_event="2024-01-15",
        t_ingest=5,
        source_authority=1.0,
        temporal_recency=0.7,
        evidence_count=2,
    )
    d = cv.to_dict()
    assert d["value"] == "7.2"
    assert d["confidence"] == 0.85
    assert d["evidence_count"] == 2

    restored = CandidateValue.from_dict(d)
    assert restored.value == "7.2"
    assert restored.source_authority == 1.0
