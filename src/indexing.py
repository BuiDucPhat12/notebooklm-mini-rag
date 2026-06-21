"""Nạp PDF → chia chunk → embed → lưu Qdrant. ID chunk deterministic để re-ingest không nhân bản."""
import hashlib
import uuid
from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.schemas import ChunkMetadata
from src.store import ensure_collection, get_vector_store


def _document_id(path: Path) -> str:
    raw = f"{path.name}:{path.stat().st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _chunk_id(doc_id: str, page: int, index: int) -> str:
    return f"{doc_id}:{page}:{index}"


def _load_pdf(path: Path):
    pages = PyPDFLoader(str(path)).load()
    doc_id = _document_id(path)
    for doc in pages:
        page_number = int(doc.metadata.get("page", 0)) + 1
        doc.metadata = {
            "document_id": doc_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": doc.metadata.get("section"),
        }
    return pages


def _splitter(chunk_size: int | None = None, chunk_overlap: int | None = None):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )


def build_chunks(pdf_paths, chunk_size=None, chunk_overlap=None, chunker=None):
    page_docs = []
    for path in pdf_paths:
        page_docs.extend(_load_pdf(Path(path)))

    splitter = chunker or _splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(page_docs)
    per_doc = defaultdict(int)

    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = per_doc[doc_id]
        per_doc[doc_id] += 1
        meta = ChunkMetadata(
            document_id=doc_id,
            filename=chunk.metadata["filename"],
            source=chunk.metadata["source"],
            page=chunk.metadata["page"],
            chunk_id=_chunk_id(doc_id, chunk.metadata["page"], idx),
            section=chunk.metadata.get("section"),
        )
        chunk.metadata = meta.model_dump()
    return chunks


def index_chunks(chunks, collection_name: str | None = None) -> int:
    if not chunks:
        return 0
    # uuid5 từ chunk_id → cùng chunk luôn ra cùng point id ⇒ idempotent.
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    get_vector_store(collection_name).add_documents(chunks, ids=ids)
    return len(chunks)


def discover_pdfs(data_dir: Path | None = None):
    base = data_dir or settings.data_dir
    return sorted(base.glob("*.pdf"))


def list_documents(collection_name: str | None = None) -> list[dict]:
    """Liệt kê tài liệu đã index: filename, số chunk, số trang (scroll toàn collection)."""
    from src.store import get_client

    name = collection_name or settings.qdrant_collection
    client = get_client()
    if not client.collection_exists(name):
        return []

    stats: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(name, limit=256, offset=offset, with_payload=True, with_vectors=False)
        for p in points:
            meta = (p.payload or {}).get("metadata") or {}
            fn = meta.get("filename")
            if not fn:
                continue
            s = stats.setdefault(fn, {"filename": fn, "chunks": 0, "pages": set()})
            s["chunks"] += 1
            s["pages"].add(meta.get("page", 0))
        if offset is None:
            break
    return [
        {"filename": s["filename"], "chunks": s["chunks"], "pages": len(s["pages"])}
        for s in sorted(stats.values(), key=lambda x: x["filename"])
    ]


def ingest(recreate: bool = False, collection_name: str | None = None) -> int:
    pdfs = discover_pdfs()
    if not pdfs:
        return 0
    name = ensure_collection(recreate=recreate, collection_name=collection_name)
    chunks = build_chunks(pdfs)
    return index_chunks(chunks, collection_name=name)


def save_and_ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    safe_name = Path(filename).name
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / safe_name
    dest.write_bytes(file_bytes)
    ensure_collection(recreate=False)
    chunks = build_chunks([dest])
    return {"filename": safe_name, "chunks_indexed": index_chunks(chunks)}
