"""Small local TF-IDF retriever; Chroma can replace this behind the same API."""

from __future__ import annotations

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from AI_Product_Manager.config import settings
except ImportError:
    from config import settings


class LocalKnowledgeRetriever:
    def __init__(self, knowledge_dir: Path | None = None):
        self.knowledge_dir = knowledge_dir or settings.knowledge_dir

    def retrieve(self, query: str, limit: int = 6) -> list[dict]:
        try:
            from AI_Product_Manager.rag.vector_store import ChromaKnowledgeStore

            return ChromaKnowledgeStore().retrieve(query, limit)
        except (ImportError, RuntimeError, ValueError):
            pass
        files = sorted(self.knowledge_dir.glob("**/*.md")) + sorted(
            self.knowledge_dir.glob("**/*.txt")
        )
        documents = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
            documents.extend({"source": str(path), "text": part} for part in paragraphs)
        if not documents:
            return []
        matrix = TfidfVectorizer(stop_words="english").fit_transform(
            [query] + [item["text"] for item in documents]
        )
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        ranked = scores.argsort()[::-1][:limit]
        return [
            {**documents[index], "score": round(float(scores[index]), 4)}
            for index in ranked
            if scores[index] > 0
        ]
