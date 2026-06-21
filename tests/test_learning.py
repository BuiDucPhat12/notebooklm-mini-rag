"""Unit test cho parse JSON, validate + dedup, schema, export (không gọi LLM)."""
import pytest

from src.export import export
from src.learning import _parse_json, _validate_items
from src.schemas import Flashcard, QuizItem, QuizSet, Summary


def test_parse_json_strips_code_fence():
    raw = '```json\n{"a": 1}\n```'
    assert _parse_json(raw) == {"a": 1}


def test_quizitem_rejects_bad_index():
    with pytest.raises(Exception):
        QuizItem(question="q", options=["a", "b", "c", "d"], correct_index=9, explanation="x")


def test_validate_items_dedup_and_marker_filter():
    payload = {
        "items": [
            {"question": "Trùng", "options": ["a", "b", "c", "d"], "correct_index": 0,
             "explanation": "e", "source_markers": ["S1", "S99"]},
            {"question": "trùng", "options": ["a", "b", "c", "d"], "correct_index": 1,
             "explanation": "e", "source_markers": ["S2"]},  # trùng (case-insensitive) → loại
            {"question": "Lỗi", "options": ["a", "b"], "correct_index": 0, "explanation": "e"},  # thiếu option → loại
        ]
    }
    items = _validate_items(payload, "items", QuizItem, "question", valid_markers={"S1", "S2"})
    assert len(items) == 1
    assert items[0].source_markers == ["S1"]  # S99 bị loại (không hợp lệ)


def test_validate_items_raises_when_empty():
    with pytest.raises(RuntimeError):
        _validate_items({"cards": []}, "cards", Flashcard, "front", valid_markers=set())


def test_export_quiz_markdown_and_json():
    qs = QuizSet(scope="corpus", items=[
        QuizItem(question="1+1?", options=["1", "2", "3", "4"], correct_index=1, explanation="hai"),
    ])
    md = export(qs, fmt="md")
    assert "✅" in md and "1+1?" in md
    js = export(qs, fmt="json")
    assert '"correct_index": 1' in js


def test_export_summary_markdown():
    s = Summary(scope="corpus", summary="tóm tắt", key_points=["ý 1", "ý 2"])
    md = export(s, fmt="md")
    assert "tóm tắt" in md and "ý 1" in md
