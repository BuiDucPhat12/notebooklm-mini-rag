"""Unit test cho chunking (không cần embedding model)."""
from pathlib import Path

import pytest

from src.indexing import _chunk_id, _document_id, build_chunks

PDF = Path("data/notebooklm.pdf")
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="thiếu data/notebooklm.pdf")


def test_document_id_stable():
    assert _document_id(PDF) == _document_id(PDF)
    assert len(_document_id(PDF)) == 16


def test_chunk_id_format():
    assert _chunk_id("abc123", 4, 7) == "abc123:4:7"


def test_build_chunks_metadata_and_stability():
    a = build_chunks([PDF])
    b = build_chunks([PDF])

    assert len(a) > 0
    # chunk_id ổn định giữa 2 lần build (deterministic)
    assert [c.metadata["chunk_id"] for c in a] == [c.metadata["chunk_id"] for c in b]

    first = a[0].metadata
    assert first["filename"] == "notebooklm.pdf"
    assert first["page"] >= 1
    assert first["chunk_id"].startswith(first["document_id"])
