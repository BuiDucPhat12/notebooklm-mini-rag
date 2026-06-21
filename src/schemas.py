"""Schema dữ liệu dùng chung giữa các module (indexing, retrieval, interfaces)."""
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    source: str
    page: int
    chunk_id: str
    section: str | None = None


class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: ChunkMetadata


class Citation(BaseModel):
    source_index: int
    source_marker: str  # "S1", "S2", ...
    filename: str
    page: int
    section: str | None = None
    chunk_id: str | None = None


class RagAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
