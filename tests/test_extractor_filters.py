from src.uniq_cluster_memory.m1_event_extraction.extractor import MedicalEventExtractor


def test_filter_rejects_placeholder_values():
    assert not MedicalEventExtractor._is_valid_record("heart_rate", "that")
    assert not MedicalEventExtractor._is_valid_record("blood_glucose", "")
    assert not MedicalEventExtractor._is_valid_record("blood_glucose", "not_mentioned")


def test_filter_rejects_non_numeric_measurements():
    assert not MedicalEventExtractor._is_valid_record("blood_glucose", "higher than that reading")
    assert not MedicalEventExtractor._is_valid_record("hemoglobin", "change")


def test_filter_keeps_valid_records():
    assert MedicalEventExtractor._is_valid_record("blood_glucose", "7.2")
    assert MedicalEventExtractor._is_valid_record("medication", "Metformin 500mg bid")
    assert MedicalEventExtractor._is_valid_record("symptom", "fatigue")


def test_normalize_record_converts_measurement_units():
    rec = MedicalEventExtractor._normalize_record(
        "blood_glucose",
        "180 mg/dL",
        "",
        1.0,
    )
    assert rec is not None
    attr, value, unit, _ = rec
    assert attr == "blood_glucose"
    assert unit == "mmol/L"
    assert value == "10"


def test_normalize_record_rejects_inconsistent_units():
    rec = MedicalEventExtractor._normalize_record(
        "heart_rate",
        "80",
        "mmHg",
        1.0,
    )
    assert rec is None


def test_normalize_record_maps_symptom_synonyms():
    rec = MedicalEventExtractor._normalize_record(
        "symptom",
        "frequent urination",
        "",
        1.0,
    )
    assert rec is not None
    attr, value, unit, _ = rec
    assert attr == "symptom"
    assert value == "polyuria"
    assert unit == ""


def test_normalize_record_canonicalizes_medication_frequency():
    rec = MedicalEventExtractor._normalize_record(
        "medication",
        "Metformin 500 mg twice daily",
        "",
        1.0,
    )
    assert rec is not None
    attr, value, unit, _ = rec
    assert attr == "medication"
    assert value == "Metformin 500mg bid"
    assert unit == ""


def test_normalize_record_canonicalizes_medication_night_frequency():
    rec = MedicalEventExtractor._normalize_record(
        "medication",
        "Atorvastatin 40 mg at night",
        "",
        1.0,
    )
    assert rec is not None
    attr, value, unit, _ = rec
    assert attr == "medication"
    assert value == "Atorvastatin 40mg qn"
    assert unit == ""


def test_normalize_record_canonicalizes_diagnosis():
    rec = MedicalEventExtractor._normalize_record(
        "primary_diagnosis",
        "hypertension",
        "",
        1.0,
    )
    assert rec is not None
    attr, value, unit, _ = rec
    assert attr == "primary_diagnosis"
    assert value == "Essential Hypertension"
    assert unit == ""


def test_filter_rejects_speculative_patient_question_diagnosis():
    rec = MedicalEventExtractor._normalize_record(
        "primary_diagnosis",
        "anemia",
        "",
        1.0,
        speaker="patient",
        snippet="Could that be anemia?",
    )
    assert rec is None
