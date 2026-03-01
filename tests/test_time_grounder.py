from src.uniq_cluster_memory.m3_uniqueness.manager import TimeGrounder


def _ground(expr: str, ref_date: str = "2026-02-28") -> str:
    return TimeGrounder(dialogue_date=ref_date).ground(expr)


def test_ground_english_relative_day():
    assert _ground("yesterday") == "2026-02-27"
    assert _ground("3 days ago") == "2026-02-25"


def test_ground_chinese_weekday():
    assert _ground("上周三") == "2026-02-18"
    assert _ground("这周一") == "2026-02-23"


def test_ground_week_month_year_scope():
    assert _ground("last week") == "2026-W08"
    assert _ground("2个月前") == "2025-12"
    assert _ground("去年") == "2025"


def test_ground_absolute_formats():
    assert _ground("2025/12/03") == "2025-12-03"
    assert _ground("2025-W09") == "2025-W09"
    assert _ground("2025-11") == "2025-11"


def test_ground_range():
    assert _ground("2025-01-03 to 2025-01-20") == "2025-01-03..2025-01-20"


def test_ground_missing_time_can_use_reference_day():
    tg = TimeGrounder(dialogue_date="2024-06-18", missing_time_scope="reference_day")
    assert tg.ground("global") == "2024-06-18"
    assert tg.ground("") == "2024-06-18"


def test_ground_unresolved_expression_falls_back_to_reference_day():
    tg = TimeGrounder(dialogue_date="2024-06-18", missing_time_scope="reference_day")
    tg._llm_ground = lambda _: "global"  # type: ignore[method-assign]
    assert tg.ground("a vague temporal mention") == "2024-06-18"
