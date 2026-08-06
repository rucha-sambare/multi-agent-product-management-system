"""CLI for ingesting verified knowledge documents into Chroma."""

from AI_Product_Manager.rag.vector_store import ChromaKnowledgeStore


if __name__ == "__main__":
    count = ChromaKnowledgeStore().ingest_directory()
    print(f"Ingested {count} knowledge chunks.")

