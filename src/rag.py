"""Pipeline RAG: retrieve → render prompt → gọi LLM → đóng gói câu trả lời + trích dẫn."""
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import settings
from src.filters import filters_to_qdrant
from src.llm import invoke_llm
from src.schemas import Citation, ChunkMetadata, RagAnswer, RetrievedChunk
from src.store import get_vector_store

PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_TEMPLATE = "answer.jinja2"
NO_CONTEXT = "Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."


@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(template_name: str, **context) -> str:
    return _jinja_env().get_template(template_name).render(**context)


def retrieve(query, k=None, filters=None, collection_name=None) -> list[RetrievedChunk]:
    hits = get_vector_store(collection_name).similarity_search_with_score(
        query=query,
        k=k or settings.top_k,
        filter=filters_to_qdrant(filters),
    )
    return [
        RetrievedChunk(text=doc.page_content, score=float(score), metadata=ChunkMetadata(**doc.metadata))
        for doc, score in hits
    ]


def fetch_all_chunks(filters=None, collection_name=None) -> list[RetrievedChunk]:
    """Lấy toàn bộ chunk thỏa filter (scroll). Dùng cho tóm tắt/tạo học liệu theo phạm vi."""
    from src.filters import filters_to_qdrant
    from src.store import get_client

    name = collection_name or settings.qdrant_collection
    client = get_client()
    qfilter = filters_to_qdrant(filters)
    results, offset = [], None
    while True:
        points, offset = client.scroll(
            name, scroll_filter=qfilter, limit=256, offset=offset, with_payload=True, with_vectors=False
        )
        for p in points:
            payload = p.payload or {}
            meta, text = payload.get("metadata") or {}, payload.get("page_content") or ""
            if meta and text:
                results.append(RetrievedChunk(text=text, score=0.0, metadata=ChunkMetadata(**meta)))
        if offset is None:
            break
    return sorted(results, key=lambda r: (r.metadata.filename, r.metadata.page, r.metadata.chunk_id))


def format_citations(chunks) -> list[Citation]:
    return [
        Citation(
            source_index=i,
            source_marker=f"S{i}",
            filename=c.metadata.filename,
            page=c.metadata.page,
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
        )
        for i, c in enumerate(chunks, start=1)
    ]


def answer(question, k=None, filters=None, collection_name=None) -> RagAnswer:
    chunks = retrieve(question, k=k, filters=filters, collection_name=collection_name)
    if not chunks:
        return RagAnswer(question=question, answer=NO_CONTEXT)

    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=chunks)
    text = invoke_llm(prompt).strip()
    return RagAnswer(
        question=question,
        answer=text,
        citations=format_citations(chunks),
        chunks=chunks,
    )
