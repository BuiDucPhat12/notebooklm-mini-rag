"""Giao diện Streamlit. Chạy: uv run streamlit run src/interfaces/ui.py

Qdrant local path mode chỉ 1 process giữ → dùng @st.cache_resource để khởi tạo
client/embedding một lần duy nhất, tránh tự lock khi Streamlit hot-reload.
"""
import os
import sys

# Cho phép `import src.*` khi Streamlit chạy file trực tiếp.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from src.config import settings
from src.filters import MetadataFilter
from src.indexing import list_documents, save_and_ingest_pdf
from src.rag import answer


@st.cache_resource(show_spinner="Đang nạp embedding + Qdrant…")
def _warmup():
    from src.store import get_client, get_embeddings

    get_embeddings()
    get_client()
    return True


def _sidebar():
    st.sidebar.header("📒 Tài liệu")
    up = st.sidebar.file_uploader("Tải PDF lên", type=["pdf"])
    if up is not None:
        if st.sidebar.button(f"Index '{up.name}'"):
            with st.spinner("Đang index…"):
                info = save_and_ingest_pdf(up.getvalue(), up.name)
            st.sidebar.success(f"{info['filename']}: {info['chunks_indexed']} chunks")
            list_documents.clear() if hasattr(list_documents, "clear") else None
            st.rerun()

    docs = list_documents()
    st.sidebar.caption("Đã index")
    for d in docs:
        st.sidebar.write(f"🟢 **{d['filename']}** — {d['pages']} trang · {d['chunks']} chunks")

    names = [d["filename"] for d in docs]
    chosen = st.sidebar.multiselect("Phạm vi truy xuất (trống = tất cả)", names, default=[])
    top_k = st.sidebar.slider("top_k", 1, 20, settings.top_k)
    return chosen, top_k


def _tab_chat(chosen, top_k):
    q = st.text_input("Câu hỏi", placeholder="VD: RAG giúp giảm hallucination bằng cách nào?")
    if st.button("Hỏi", type="primary") and q.strip():
        flt = MetadataFilter(filenames=chosen) if chosen else None
        with st.spinner("Đang truy xuất + sinh câu trả lời…"):
            res = answer(q, k=top_k, filters=flt)
        st.markdown(f"**Hỏi:** {res.question}")
        st.markdown(res.answer)
        if res.chunks:
            st.divider()
            st.caption("Nguồn trích dẫn")
            for c, cit in zip(res.chunks, res.citations):
                with st.container(border=True):
                    st.caption(f"**[{cit.source_marker}]** {cit.filename} · trang {cit.page} · score {c.score:.3f}")
                    st.write(c.text[:400] + ("…" if len(c.text) > 400 else ""))


def run():
    st.set_page_config(page_title="NotebookLM mini", layout="wide")
    st.title("📒 NotebookLM mini")
    _warmup()
    chosen, top_k = _sidebar()

    tab_chat, tab_sum, tab_quiz, tab_fc = st.tabs(["Hỏi đáp", "Tóm tắt", "Quiz", "Flashcards"])
    with tab_chat:
        _tab_chat(chosen, top_k)
    for tab, name in [(tab_sum, "Tóm tắt"), (tab_quiz, "Quiz"), (tab_fc, "Flashcards")]:
        with tab:
            st.info(f"'{name}' sẽ có ở Phase 4.")


run()
