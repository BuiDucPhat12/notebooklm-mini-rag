"""Bộ câu hỏi vàng nhỏ để đo chất lượng retrieval (nội dung trong notebooklm.pdf).

Mỗi câu kèm expected_keywords: các từ/cụm phải xuất hiện trong chunk truy xuất nếu
retrieval đúng. Đây là proxy nhẹ cho context recall (không cần LLM-judge).
"""

GOLD = [
    {
        "question": "RAG giúp giảm hallucination bằng cách nào?",
        "keywords": ["ngữ cảnh", "truy xuất", "hallucination"],
    },
    {
        "question": "Hệ thống lưu vector vào cơ sở dữ liệu nào?",
        "keywords": ["qdrant"],
    },
    {
        "question": "Các chiến lược chunking nào được thực nghiệm?",
        "keywords": ["recursive", "semantic"],
    },
    {
        "question": "Bước reranking dùng mô hình gì?",
        "keywords": ["cross-encoder"],
    },
    {
        "question": "Hệ thống cung cấp những chức năng học tập nào?",
        "keywords": ["quiz", "flashcard", "tóm tắt"],
    },
    {
        "question": "Ragas đánh giá theo những chỉ số nào?",
        "keywords": ["faithfulness", "context", "relevanc"],
    },
]
