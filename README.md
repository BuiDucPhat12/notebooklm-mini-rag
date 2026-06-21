# 📒 NotebookLM mini — RAG for Learning System

Hệ thống hỏi-đáp tài liệu học tập theo kiến trúc **Retrieval-Augmented Generation (RAG)**:
nạp PDF của bạn → hỏi-đáp **có trích dẫn nguồn**, tóm tắt, tạo quiz & flashcards — câu trả lời
bám sát tài liệu, hạn chế "ảo giác" (hallucination).

Xây **from scratch** để hiểu sâu từng tầng của pipeline RAG (không dùng framework đóng gói sẵn),
trên dữ liệu & embedding **tiếng Việt**.

> Bài tập lớn AI VIET NAM (AIO2025) — tự cài đặt lại theo kiến trúc đề.

---

## ✨ Tính năng

| | Tính năng | Trạng thái |
|---|---|---|
| ✅ | Nạp PDF → chunk → embed → index vào vector store (idempotent) | **MVP** |
| ✅ | Hỏi-đáp ngữ nghĩa **kèm trích dẫn nguồn** `[S1] (file, trang)` | **MVP** |
| ✅ | Lọc phạm vi truy xuất theo từng tài liệu | **MVP** |
| ✅ | Hai giao diện: **CLI** (Typer) & **Web UI** (Streamlit) | **MVP** |
| 🔜 | Tóm tắt theo chiến lược **map-reduce** | Roadmap |
| 🔜 | Sinh **Quiz** & **Flashcards** tự động | Roadmap |
| 🔜 | Đánh giá chất lượng bằng **Ragas** + thực nghiệm chunking/reranking | Roadmap |

---

## 🏗️ Kiến trúc

```
                    ┌──────────── Ingestion ────────────┐
   PDF  ──►  pypdf  ──►  Recursive   ──►  GreenNode VN  ──►  Qdrant
            (theo trang)   chunker        embedding         (vector store)
                                                                │
                    ┌──────────── Query (RAG) ──────────────────┘
   Câu hỏi ──►  embed ──►  similarity search (top_k, + filter)
                                  │
                          chunks (S1, S2…)
                                  │
                       prompt (Jinja2, ép chỉ dùng ngữ cảnh)
                                  │
                              Gemini LLM
                                  │
                     RagAnswer = câu trả lời + citations
```

**Nguyên tắc thiết kế:** schema-first (Pydantic), tách tầng rõ ràng
(`config · schemas · indexing · store · filters · rag · llm · interfaces`),
LLM gọi qua một lớp trung gian nên đổi backend (Gemini ↔ local) chỉ bằng `.env`.

---

## 🧰 Tech stack

| Thành phần | Lựa chọn |
|---|---|
| Ngôn ngữ / môi trường | Python 3.12 · [`uv`](https://docs.astral.sh/uv/) |
| Orchestration | LangChain |
| Vector store | Qdrant (local mode) |
| Embedding | `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1` (tiếng Việt, 1024-dim) |
| LLM | Google **Gemini** (`gemini-2.5-flash`); hỗ trợ HF-local qua config |
| Giao diện | Streamlit (web) · Typer (CLI) |
| Test | pytest |

---

## 🚀 Chạy thử

```bash
# 1. Cài môi trường (uv tự lo Python 3.12 + deps)
uv sync

# 2. Cấu hình
cp .env.example .env
#   điền GOOGLE_API_KEY (free: https://aistudio.google.com/apikey)

# 3. Bỏ vài file PDF vào data/ rồi index
uv run python -m src.interfaces.cli ingest

# 4a. Hỏi qua CLI
uv run python -m src.interfaces.cli ask "RAG giúp giảm hallucination bằng cách nào?"

# 4b. hoặc mở Web UI
uv run streamlit run src/interfaces/ui.py
```

**Ví dụ output (CLI):**
```
RAG giảm hallucination bằng cách không để LLM trả lời từ tri thức nội tại [S2],
mà truy xuất các đoạn liên quan từ tài liệu rồi đưa vào prompt [S1].

Nguồn:
  [S1] notebooklm.pdf · trang 1
  [S2] notebooklm.pdf · trang 4
```

---

## 📂 Cấu trúc

```
src/
├── config.py        # Settings (pydantic-settings, đọc .env)
├── schemas.py       # ChunkMetadata, RetrievedChunk, Citation, RagAnswer
├── indexing.py      # load PDF → chunk → index (id deterministic, idempotent)
├── store.py         # embedding + Qdrant client / collection
├── filters.py       # MetadataFilter → Qdrant filter
├── rag.py           # retrieve → render prompt → LLM → answer + citations
├── llm.py           # lớp trung gian gọi LLM (Gemini / HF-local)
├── prompts/         # prompt template (Jinja2)
└── interfaces/      # cli.py (Typer) · ui.py (Streamlit)
tests/               # unit test: chunking, filters, citations, fallback
```

---

## ✅ Kiểm thử & chất lượng

```bash
uv run pytest        # 8 unit test: chunk_id ổn định, idempotent ingest,
                     # filter → Qdrant, citations, fallback khi thiếu ngữ cảnh
```

**Retrieval đa tài liệu** (corpus 3 PDF khác chủ đề) — hệ thống định tuyến đúng nguồn:

| Câu hỏi | Nguồn top-1 |
|---|---|
| "LangGraph vs Deep Agents?" | `agentic_rag.pdf` (0.72) |
| "Tạo flashcards và quiz?" | `notebooklm.pdf` (0.65) |
| "Reciprocal Rank Fusion?" | `agentic_rag.pdf` (0.56) |

---

## 🗺️ Roadmap

- [x] **MVP** — ingest → Q&A có trích dẫn (CLI + Web)
- [ ] Tóm tắt map-reduce, Quiz, Flashcards
- [ ] Đánh giá Ragas (context recall/precision, faithfulness, answer relevancy)
- [ ] Thực nghiệm chunking (recursive vs semantic) + reranking (cross-encoder)

---

*Author: Bùi Đức Phát — Data Analyst, học AI engineering theo hướng build-from-scratch.*
