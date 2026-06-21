"""CLI bằng Typer — kiểm thử nhanh trong terminal.

Dùng:  uv run python -m src.interfaces.cli ingest
        uv run python -m src.interfaces.cli ask "RAG là gì?"
        uv run python -m src.interfaces.cli debug-retrieval "RAG là gì?"
"""
import json

import typer

from src.filters import MetadataFilter
from src.indexing import ingest as ingest_data
from src.indexing import list_documents
from src.rag import answer, retrieve

app = typer.Typer(add_completion=False, help="NotebookLM mini — CLI")


def _filter(filename: str | None):
    return MetadataFilter(filename=filename) if filename else None


@app.command()
def ingest(recreate: bool = typer.Option(False, help="Xóa collection rồi index lại")):
    """Index toàn bộ PDF trong data/."""
    n = ingest_data(recreate=recreate)
    typer.echo(f"Done. {n} chunks indexed.")


@app.command()
def docs():
    """Liệt kê tài liệu đã index."""
    for d in list_documents():
        typer.echo(f"- {d['filename']}: {d['chunks']} chunks, {d['pages']} trang")


@app.command()
def ask(
    question: str,
    k: int = typer.Option(None, help="Số chunk truy xuất"),
    filename: str = typer.Option(None, help="Giới hạn theo 1 file"),
):
    """Hỏi-đáp RAG có trích dẫn."""
    res = answer(question, k=k, filters=_filter(filename))
    typer.echo("\n" + res.answer + "\n")
    if res.citations:
        typer.echo("Nguồn:")
        for c in res.citations:
            typer.echo(f"  [{c.source_marker}] {c.filename} · trang {c.page}")


@app.command("debug-retrieval")
def debug_retrieval(question: str, k: int = typer.Option(None), filename: str = typer.Option(None)):
    """Xem chunk truy xuất (không gọi LLM)."""
    chunks = retrieve(question, k=k, filters=_filter(filename))
    typer.echo(json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
