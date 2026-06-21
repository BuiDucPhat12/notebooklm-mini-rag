"""Chức năng học tập: tóm tắt (map-reduce), quiz, flashcards.

Nguyên tắc: luôn bám ngữ cảnh đã index, LLM trả JSON → parse → validate Pydantic → dedup.
"""
import json

from pydantic import ValidationError

from src.config import settings
from src.rag import fetch_all_chunks, format_citations, render_prompt, retrieve
from src.llm import invoke_llm
from src.schemas import Flashcard, FlashcardSet, QuizItem, QuizSet, Summary

SUMMARY_SINGLE = "summary_single.jinja2"
SUMMARY_MAP = "summary_map.jinja2"
SUMMARY_REDUCE = "summary_reduce.jinja2"
QUIZ = "quiz.jinja2"
FLASHCARDS = "flashcards.jinja2"


def _resolve_target(document, query, filters, retrieval_k):
    """Xác định phạm vi nội dung: query → semantic search; document/filter → toàn bộ chunk; else corpus."""
    eff = dict(filters or {})
    if document:
        eff["filename"] = document

    if query:
        return retrieve(query, k=retrieval_k, filters=eff or None), "query", query
    if eff:
        scope = "document" if document else "filter"
        target = ", ".join(f"{k}={v}" for k, v in eff.items())
        return fetch_all_chunks(filters=eff), scope, target
    return fetch_all_chunks(filters=None), "corpus", None


def _parse_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    obj = json.loads(cleaned)
    if not isinstance(obj, (dict, list)):
        raise RuntimeError("Mong đợi JSON object hoặc array.")
    return obj


def _validate_summary(payload) -> tuple[str, list[str]]:
    summary = str(payload.get("summary", "")).strip()
    key_points = [str(k).strip() for k in payload.get("key_points", []) if str(k).strip()]
    if not summary:
        raise RuntimeError("Tóm tắt rỗng.")
    return summary, key_points


def _validate_items(payload, key, model_cls, dedup_field, valid_markers):
    items, seen = [], set()
    for raw in payload.get(key, []):
        try:
            item = model_cls.model_validate(raw)
        except ValidationError:
            continue
        norm = str(getattr(item, dedup_field, "")).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        markers = [m for m in item.source_markers if m in valid_markers]
        items.append(item.model_copy(update={"source_markers": markers}))
    if not items:
        raise RuntimeError(f"Không tạo được {key} hợp lệ.")
    return items


def summarize(document=None, query=None, filters=None) -> Summary:
    chunks, scope, target = _resolve_target(document, query, filters, settings.summarize_retrieval_k)
    if not chunks:
        return Summary(scope=scope, target=target, summary="Không có nội dung trong phạm vi đã chọn.")

    if len(chunks) <= settings.summarize_batch_size:
        payload = _parse_json(invoke_llm(render_prompt(SUMMARY_SINGLE, chunks=chunks)))
        summary, key_points = _validate_summary(payload)
    else:
        partials = []
        for start in range(0, len(chunks), settings.summarize_batch_size):
            batch = chunks[start : start + settings.summarize_batch_size]
            p = _parse_json(invoke_llm(render_prompt(SUMMARY_MAP, chunks=batch)))
            s, _ = _validate_summary(p)
            partials.append({"summary": s})
        payload = _parse_json(invoke_llm(render_prompt(SUMMARY_REDUCE, partials=partials)))
        summary, key_points = _validate_summary(payload)

    return Summary(
        scope=scope, target=target, summary=summary, key_points=key_points,
        citations=format_citations(chunks), chunks=chunks,
    )


def _markers(chunks):
    return {f"S{i}" for i in range(1, len(chunks) + 1)}


def generate_quiz(document=None, query=None, filters=None, count=None) -> QuizSet:
    chunks, scope, target = _resolve_target(document, query, filters, settings.generation_retrieval_k)
    n = count or settings.quiz_default_count
    payload = _parse_json(invoke_llm(render_prompt(QUIZ, chunks=chunks, count=n)))
    items = _validate_items(payload, "items", QuizItem, "question", _markers(chunks))
    return QuizSet(scope=scope, target=target, items=items,
                   citations=format_citations(chunks), chunks=chunks)


def generate_flashcards(document=None, query=None, filters=None, count=None) -> FlashcardSet:
    chunks, scope, target = _resolve_target(document, query, filters, settings.generation_retrieval_k)
    n = count or settings.flashcards_default_count
    payload = _parse_json(invoke_llm(render_prompt(FLASHCARDS, chunks=chunks, count=n)))
    cards = _validate_items(payload, "cards", Flashcard, "front", _markers(chunks))
    return FlashcardSet(scope=scope, target=target, cards=cards,
                        citations=format_citations(chunks), chunks=chunks)
