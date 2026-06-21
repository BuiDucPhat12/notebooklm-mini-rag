"""Unit test cho filters, citations và fallback (không gọi LLM/embedding)."""
from qdrant_client import models as qmodels

import src.rag as rag
from src.filters import MetadataFilter, filters_to_qdrant
from src.rag import NO_CONTEXT, answer, format_citations
from src.schemas import ChunkMetadata, RetrievedChunk


def _chunk(i):
    return RetrievedChunk(
        text=f"đoạn {i}",
        score=0.9,
        metadata=ChunkMetadata(
            document_id="doc1", filename="a.pdf", source="/a.pdf", page=i, chunk_id=f"doc1:{i}:0"
        ),
    )


def test_filters_to_qdrant_single_filename():
    q = filters_to_qdrant(MetadataFilter(filename="a.pdf"))
    assert isinstance(q, qmodels.Filter)
    cond = q.must[0]
    assert cond.key == "metadata.filename"
    assert cond.match.value == "a.pdf"


def test_filters_to_qdrant_multi_filenames_uses_matchany():
    q = filters_to_qdrant(MetadataFilter(filenames=["a.pdf", "b.pdf"]))
    cond = q.must[0]
    assert cond.key == "metadata.filename"
    assert set(cond.match.any) == {"a.pdf", "b.pdf"}


def test_filters_none_returns_none():
    assert filters_to_qdrant(None) is None
    assert filters_to_qdrant(MetadataFilter()) is None


def test_format_citations_markers():
    cits = format_citations([_chunk(1), _chunk(2)])
    assert [c.source_marker for c in cits] == ["S1", "S2"]
    assert cits[0].filename == "a.pdf"


def test_answer_fallback_when_no_chunks(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: [])
    res = answer("câu hỏi bất kỳ")
    assert res.answer == NO_CONTEXT
    assert res.citations == []
    assert res.chunks == []
