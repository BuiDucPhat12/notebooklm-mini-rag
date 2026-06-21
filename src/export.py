"""Xuất kết quả học tập (Summary/QuizSet/FlashcardSet) ra text/markdown/json."""
from pathlib import Path
from typing import Literal

from src.schemas import FlashcardSet, QuizSet, Summary

ExportFormat = Literal["text", "md", "json"]


def _summary_md(m: Summary) -> str:
    lines = [f"# Tóm tắt ({m.scope})", "", m.summary, ""]
    if m.key_points:
        lines.append("## Ý chính")
        lines += [f"- {k}" for k in m.key_points]
    return "\n".join(lines) + "\n"


def _quiz_md(m: QuizSet) -> str:
    lines = [f"# Quiz ({m.scope}) — {len(m.items)} câu", ""]
    for i, q in enumerate(m.items, 1):
        lines.append(f"**Câu {i}. {q.question}**")
        for j, opt in enumerate(q.options):
            mark = " ✅" if j == q.correct_index else ""
            lines.append(f"- {chr(65 + j)}. {opt}{mark}")
        lines.append(f"  > {q.explanation}  `{' '.join(q.source_markers)}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _flashcards_md(m: FlashcardSet) -> str:
    lines = [f"# Flashcards ({m.scope}) — {len(m.cards)} thẻ", ""]
    for i, c in enumerate(m.cards, 1):
        lines.append(f"**{i}. {c.front}**")
        lines.append(f"  - {c.back}")
        if c.hint:
            lines.append(f"  - 💡 {c.hint}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _to_markdown(model) -> str:
    if isinstance(model, Summary):
        return _summary_md(model)
    if isinstance(model, QuizSet):
        return _quiz_md(model)
    if isinstance(model, FlashcardSet):
        return _flashcards_md(model)
    raise TypeError(f"Không hỗ trợ export: {type(model)}")


def export(model, fmt: ExportFormat = "text", output: Path | None = None):
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"
    elif fmt in ("text", "md"):
        text = _to_markdown(model)
    else:
        raise ValueError(f"fmt không hợp lệ '{fmt}' (text|md|json)")

    if output is None:
        return text
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
