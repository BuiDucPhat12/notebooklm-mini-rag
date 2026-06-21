# Plan — Project 1: NotebookLM mini (RAG for Learning System)

Build-from-scratch, bám kiến trúc đề AIVN. Runtime `uv` + Python 3.12.
Provider dev = Gemini (gemini-2.5-flash). Vector store = Qdrant local.
Embedding = GreenNode VN. **MVP = Phase 0–3.**

Quyết định nguồn: xem memory `aivn-rag-projects-decisions`.

---

## Phase 0 — Scaffold & runtime ✅ gate: env import được + embedding ra vector
- `uv` tạo venv pin Python 3.12; `pyproject.toml` **pin version** (tránh API đổi):
  `langchain>=0.3,<0.4`, `langchain-qdrant>=0.2`, `langchain-google-genai>=2.0`,
  `langchain-huggingface`, `qdrant-client>=1.9`, `sentence-transformers`, `pypdf`,
  `pydantic>=2`, `pydantic-settings`, `jinja2`, `typer`, `streamlit`, `python-dotenv`.
- Embedding **chốt tên model rõ:** `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1`
  (HF Hub, load local qua `langchain-huggingface` `HuggingFaceEmbeddings`, ~model lớn).
- Cấu trúc `src/`: `config.py schemas.py indexing.py store.py rag.py filters.py llm.py prompts/ interfaces/`. `.env.example`, `README.md`, `data/`, `.gitignore`, `git init`.
- **Verify:** `uv run python -c "import langchain, qdrant_client, langchain_google_genai"` OK
  **+ [fix CRITICAL] `embed_query("xin chào")` trả về vector đúng dim** (không để lỗi embedding lòi ra ở Phase 1).

## Phase 1 — Ingestion → Qdrant ✅ gate: index PDF mẫu, chunks > 0
- `config.py`: `Settings(BaseSettings)` (data_dir, storage_dir, qdrant_collection, chunk_size/overlap, top_k, embedding_model, llm_provider, gemini_model, GOOGLE_API_KEY) + validator.
- `schemas.py`: ChunkMetadata, RetrievedChunk, Citation, RagAnswer.
- `indexing.py`: `_load_pdf` (pypdf, gắn metadata/trang), recursive splitter, `build_chunks`.
  **[fix CRITICAL] chunk_id deterministic:** `doc_id = sha1(filename + filesize)[:16]`,
  `chunk_id = f"{doc_id}:{page}:{index}"`, point id = `uuid5(NAMESPACE_DNS, chunk_id)` → re-ingest KHÔNG nhân bản.
- `store.py`: `get_embeddings` (GreenNode), `get_client` (Qdrant local path), `ensure_collection` (payload index), `get_vector_store`.
- **Verify + test:** ingest PDF mẫu → số chunk > 0; chunk_id ổn định; metadata đúng (filename/page);
  **[fix CRITICAL] ingest 2 lần cùng PDF → collection size KHÔNG tăng (idempotent).**

## Phase 2 — RAG Q&A core (MVP end-to-end) ✅ gate: hỏi → trả lời + citation
- `filters.py`: MetadataFilter + chuẩn hóa + `filters_to_qdrant`.
- `llm.py`: backend Gemini (`langchain-google-genai`), `invoke_llm`; HF-local để stub/config.
- `rag.py`: `retrieve`, `render_prompt` (Jinja2), `format_citations`, `answer`.
- `prompts/answer.jinja2`: ép model chỉ dùng ngữ cảnh, đánh dấu S1/S2.
- **Verify + test:** retrieve trả RetrievedChunk; answer trả RagAnswer có citations khi có chunk;
  fallback "không đủ thông tin" khi rỗng; **[fix MINOR] câu hỏi ngoài chủ đề (có chunk nhưng không liên quan) → trả fallback, KHÔNG bịa**;
  **[fix MINOR] unit test `filters_to_qdrant` với 1 filter filename → khớp Qdrant filter object expected.**

## Phase 3 — Interfaces: CLI + Streamlit (đóng MVP) ✅ gate: demo trên browser
- Typer CLI: `ingest`, `ask`, `debug-retrieval`.
- Streamlit UI khớp mockup: sidebar (upload PDF + danh sách doc + chọn phạm vi + top_k), tab "Hỏi đáp" (câu trả lời + nguồn S1/S2 + snippet + score). Các tab khác hiện disabled.
- **[fix MAJOR] Qdrant local path chỉ 1 process giữ:** Streamlit dùng `@st.cache_resource`
  giữ 1 client/vector-store xuyên suốt session (tránh tự lock khi hot-reload). Lưu ý trong README: đừng chạy CLI + Streamlit đồng thời ở MVP.
- **Verify:** `streamlit run`, upload PDF, hỏi, thấy citation (không lock-error). → **MVP DONE.**

## Phase 4 — Chức năng học tập
- `learning.py`: summarize (single + map-reduce), generate_quiz, generate_flashcards; schemas Summary/QuizItem/QuizSet/Flashcard/FlashcardSet + validate/dedup; `export.py` (text/md/json). Bật 3 tab còn lại trong UI.
- **[fix MINOR] summarize map-reduce dùng splitter riêng** (chunk lớn, overlap=0) để tránh tóm tắt nội dung trùng — không tái dùng splitter RAG có overlap.
- **Verify + test:** quiz/flashcard parse JSON + validate pydantic; loại item lỗi/trùng.

## Phase 5 — Đánh giá & thực nghiệm (scope gọn lại cho timeline portfolio)
- Ragas: **chỉ chạy trên 5–10 câu hỏi mẫu** (không full benchmark) — ưu tiên context_precision/recall (rẻ); faithfulness/answer_relevancy tốn API → chạy giới hạn.
- Chunking: so recursive vs semantic (giữ).
- **[fix MAJOR] Reranking (bge-reranker-v2-m3) là OPTIONAL/stretch** — ~560MB, CPU Mac 2–5s/query; chỉ làm nếu còn thời gian, không block portfolio.
- **Verify:** xuất bảng metrics cho ≥1 cấu hình chunking trên tập câu hỏi mẫu.

## Phase 6 — Polish & ship
- README (kiến trúc, cách chạy, screenshot), dọn code, push GitHub (repo mới, author Bui Duc Phat — KHÔNG gắn Co-Authored-By Claude), thêm card vào portfolio.
- **Verify:** clone sạch → `uv sync` → chạy lại được theo README.

---

## Rủi ro / phụ thuộc
- **GOOGLE_API_KEY** cần có trước Phase 2 (Gemini). Nếu chưa có → Phase 2 chỉ test được tới bước retrieve (không gọi LLM).
- **GreenNode embedding** (~model lớn) tải lần đầu chậm trên Mac CPU; sau đó cache. Cân nhắc model VN nhẹ hơn nếu quá chậm.
- **Qdrant local mode** (path-based) chỉ 1 process truy cập — Streamlit + CLI chạy đồng thời có thể khóa file; MVP chấp nhận, sau cân nhắc Qdrant server (docker).
- `langchain-google-genai` đổi API theo version — pin version cụ thể.
