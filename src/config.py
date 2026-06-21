"""Cấu hình tập trung. Đọc từ .env (prefix RAG_), override được bằng biến môi trường."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # Đường dẫn
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage/qdrant")
    qdrant_collection: str = "rag_chunks"

    # Chunking / retrieval
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=64)

    # Embedding tiếng Việt
    embedding_model: str = "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"
    embedding_device: str = "cpu"

    # LLM
    llm_provider: Literal["gemini", "hf_local"] = "gemini"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    gemini_model: str = "gemini-2.5-flash"
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    hf_model: str = "Qwen/Qwen2.5-3B-Instruct"

    # Tham số chức năng học tập
    summarize_batch_size: int = Field(default=10, ge=1)
    summarize_retrieval_k: int = Field(default=12, ge=1, le=128)
    generation_retrieval_k: int = Field(default=16, ge=1, le=128)
    quiz_default_count: int = Field(default=8, ge=1, le=50)
    flashcards_default_count: int = Field(default=15, ge=1, le=100)

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
