"""Integration tests for the ingestion pipeline."""
import pytest
from ingestion.pipeline import run_ingestion


class TestRunIngestion:
    def test_full_pipeline(self, sample_pdf_path, tmp_path, fake_embeddings):
        stats = run_ingestion(
            pdf_paths={"test_doc": sample_pdf_path},
            embedding_function=fake_embeddings,
            persist_directory=tmp_path / "pipeline_test",
        )
        assert stats["documents_loaded"] == 2  # 2 pages in sample PDF
        assert stats["chunks_created"] >= 2
        assert (tmp_path / "pipeline_test").exists()

    def test_pipeline_with_missing_pdf_raises(self, tmp_path, fake_embeddings):
        with pytest.raises(FileNotFoundError):
            run_ingestion(
                pdf_paths={"missing": tmp_path / "nope.pdf"},
                embedding_function=fake_embeddings,
                persist_directory=tmp_path / "fail_test",
            )

    def test_pipeline_metadata_end_to_end(self, sample_pdf_path, tmp_path, fake_embeddings):
        """Verify citation metadata flows through the entire pipeline."""
        from ingestion.store import load_vector_store

        persist_dir = tmp_path / "e2e_test"
        run_ingestion(
            pdf_paths={"test_doc": sample_pdf_path},
            embedding_function=fake_embeddings,
            persist_directory=persist_dir,
        )

        store = load_vector_store(fake_embeddings, persist_dir)
        results = store.similarity_search("test", k=1)
        meta = results[0].metadata

        assert "source_file" in meta
        assert "page" in meta
        assert "document_name" in meta
        assert "document_type" in meta
        assert "chunk_index" in meta
