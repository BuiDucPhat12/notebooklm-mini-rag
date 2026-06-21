# NotebookLM mini — RAG for Learning System

Build-from-scratch một "NotebookLM mini": nạp PDF học tập → hỏi-đáp có trích dẫn nguồn,
tóm tắt, quiz, flashcards. Mục tiêu: học sâu cơ chế RAG + portfolio.

Bám kiến trúc đề AIVN (AIO2025). Vector store **Qdrant** local · embedding tiếng Việt
**GreenNode** · LLM **Gemini** (dev) · LangChain · Streamlit/CLI.

## Yêu cầu
- [`uv`](https://docs.astral.sh/uv/) (quản lý môi trường)
- Python 3.12 (uv tự cài)
- `GOOGLE_API_KEY` cho Gemini — free tại https://aistudio.google.com/apikey (cần từ Phase 2)

## Chạy
```bash
uv sync                       # cài deps
cp .env.example .env          # rồi điền GOOGLE_API_KEY
# (các lệnh ingest / ask / streamlit thêm ở Phase sau)
```

## Trạng thái
- [x] Phase 0 — scaffold, runtime uv+3.12, deps, embedding verify (dim 1024)
- [x] Phase 1 — ingestion → Qdrant (90 chunks, idempotent, 3 unit test pass)
- [x] Phase 2 — RAG Q&A core: retrieve→prompt→Gemini→answer+citations (8 test pass, fallback OK)
- [x] Phase 3 — CLI (ingest/ask/docs/debug) + Streamlit UI → **MVP DONE**
- [ ] Phase 4 — tóm tắt / quiz / flashcards
- [ ] Phase 5 — đánh giá Ragas + thực nghiệm
- [ ] Phase 6 — polish + ship

Chi tiết kế hoạch: [`PLAN.md`](PLAN.md). Mockup UI: [`mockup_ui.html`](mockup_ui.html).
