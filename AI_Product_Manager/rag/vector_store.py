"""Persistent Chroma vector store using local deterministic embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from AI_Product_Manager.config import settings


class HashingEmbeddingFunction:
    """No-download 384-dimensional embedding accepted by Chroma."""

    def __init__(self, dimensions=384):
        self.vectorizer = HashingVectorizer(
            n_features=dimensions, alternate_sign=False, norm="l2", ngram_range=(1, 2)
        )

    def __call__(self, input):
        return [
            row for row in self.vectorizer.transform(input).toarray().astype(np.float32)
        ]

    def embed_query(self, input):
        return self(input)

    @staticmethod
    def name():
        return "apm-local-hashing-v1"

    def is_legacy(self):
        return False

    def default_space(self):
        return "cosine"

    def supported_spaces(self):
        return ["cosine", "l2", "ip"]

    def get_config(self):
        return {"dimensions": self.vectorizer.n_features}

    @staticmethod
    def build_from_config(config):
        return HashingEmbeddingFunction(config.get("dimensions", 384))

    @staticmethod
    def validate_config(config):
        if int(config.get("dimensions", 0)) <= 0:
            raise ValueError("Embedding dimensions must be positive.")


class ChromaKnowledgeStore:
    def __init__(self, persist_dir=None):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb to use persistent RAG.") from exc
        self.client = chromadb.PersistentClient(path=str(persist_dir or settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            "market_knowledge_v2",
            embedding_function=HashingEmbeddingFunction(),
            metadata={"description": "Verified competitor and release-note evidence"},
            configuration={"hnsw": {"space": "cosine"}},
        )

    def ingest_directory(self, directory: Path | None = None):
        root = directory or settings.knowledge_dir
        records = []
        for path in sorted(root.glob("**/*")):
            if path.suffix.lower() not in {".md", ".txt"} or path.name == "README.md":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for index, paragraph in enumerate(x.strip() for x in text.split("\n\n")):
                if len(paragraph) < 20:
                    continue
                identity = hashlib.sha256(f"{path}:{index}".encode()).hexdigest()
                records.append((identity, paragraph, {"source": str(path), "chunk": index}))
        if records:
            self.collection.upsert(
                ids=[x[0] for x in records],
                documents=[x[1] for x in records],
                metadatas=[x[2] for x in records],
            )
        return len(records)

    def retrieve(self, query, limit=6):
        if self.collection.count() == 0:
            self.ingest_directory()
        if self.collection.count() == 0:
            return []
        result = self.collection.query(query_texts=[query], n_results=min(limit, self.collection.count()))
        return [
            {
                "text": document,
                "source": metadata["source"],
                "score": round(max(0.0, min(1.0, 1.0 - float(distance))), 4),
            }
            for document, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
