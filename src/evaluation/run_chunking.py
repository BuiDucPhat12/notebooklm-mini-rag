"""Thực nghiệm: so sánh các cấu hình Recursive Chunking bằng retrieval-metric nhẹ.

Mỗi cấu hình (chunk_size/overlap) được index vào một collection riêng (chỉ trên
notebooklm.pdf cho nhanh), rồi đo trên bộ gold set:
  - keyword_recall@k: trung bình tỉ lệ keyword kỳ vọng xuất hiện trong top-k chunk.
  - avg_top1_score: điểm tương đồng trung bình của chunk hạng 1.

Chạy: uv run python -m src.evaluation.run_chunking
"""
from pathlib import Path

from src.config import settings
from src.evaluation.gold_set import GOLD
from src.indexing import build_chunks, index_chunks
from src.rag import retrieve
from src.store import ensure_collection

CONFIGS = [(500, 50), (1000, 150), (1500, 200)]
PDF = settings.data_dir / "notebooklm.pdf"
K = 5


def _keyword_recall(text: str, keywords: list[str]) -> float:
    low = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in low)
    return hits / len(keywords) if keywords else 0.0


def _evaluate(collection: str) -> dict:
    recalls, top1 = [], []
    for case in GOLD:
        chunks = retrieve(case["question"], k=K, collection_name=collection)
        joined = " ".join(c.text for c in chunks)
        recalls.append(_keyword_recall(joined, case["keywords"]))
        top1.append(chunks[0].score if chunks else 0.0)
    n = len(GOLD)
    return {
        "keyword_recall@%d" % K: round(sum(recalls) / n, 3),
        "avg_top1_score": round(sum(top1) / n, 3),
    }


def run():
    if not PDF.exists():
        raise SystemExit(f"Thiếu {PDF} — bỏ PDF vào data/ trước.")

    rows = []
    for cs, co in CONFIGS:
        name = f"eval_rc_{cs}_{co}"
        ensure_collection(recreate=True, collection_name=name)
        chunks = build_chunks([PDF], chunk_size=cs, chunk_overlap=co)
        index_chunks(chunks, collection_name=name)
        metrics = _evaluate(name)
        rows.append({"config": f"{cs}/{co}", "chunks": len(chunks), **metrics})

    # In bảng
    headers = ["config", "chunks", f"keyword_recall@{K}", "avg_top1_score"]
    print(" | ".join(h.ljust(16) for h in headers))
    print("-" * 72)
    for r in rows:
        print(" | ".join(str(r[h]).ljust(16) for h in headers))
    return rows


if __name__ == "__main__":
    run()
