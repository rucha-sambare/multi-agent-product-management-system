from AI_Product_Manager.rag.vector_store import ChromaKnowledgeStore


def test_chroma_ingest_and_retrieve(tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "competitor.md").write_text(
        "Source: https://example.com\n\nSpotify added offline playlist controls.",
        encoding="utf-8",
    )
    store = ChromaKnowledgeStore(tmp_path / "chroma")
    assert store.ingest_directory(knowledge) == 2
    results = store.retrieve("Spotify offline playlist", limit=2)
    assert results
    assert "Spotify" in results[0]["text"]
    assert results[0]["source"].endswith("competitor.md")

