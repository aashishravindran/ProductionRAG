"""Tests for PDF loading."""
import pytest
from ingestion.loader import load_pdf, load_all_pdfs


class TestLoadPdf:
    def test_load_valid_pdf(self, sample_pdf_path):
        docs = load_pdf(sample_pdf_path)
        assert len(docs) == 2
        assert docs[0].metadata["page"] == 0
        assert docs[1].metadata["page"] == 1
        assert docs[0].metadata["source_file"] == "test_doc.pdf"
        assert "Test content" in docs[0].page_content

    def test_load_nonexistent_pdf_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path / "nonexistent.pdf")


class TestLoadAllPdfs:
    def test_load_multiple_pdfs(self, sample_pdf_path, tmp_path):
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(text="Second doc content.")
        second_path = tmp_path / "second.pdf"
        pdf.output(str(second_path))

        paths = {"doc_a": sample_pdf_path, "doc_b": second_path}
        docs = load_all_pdfs(paths)

        assert len(docs) == 3  # 2 pages + 1 page
        doc_names = {d.metadata["document_name"] for d in docs}
        assert doc_names == {"doc_a", "doc_b"}

    def test_each_doc_has_source_file(self, sample_pdf_path):
        docs = load_all_pdfs({"test": sample_pdf_path})
        for doc in docs:
            assert "source_file" in doc.metadata
            assert "document_name" in doc.metadata
