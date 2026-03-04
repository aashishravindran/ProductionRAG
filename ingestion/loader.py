"""Load PDF documents with page-level metadata."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: Path) -> list[Document]:
    """Load a single PDF, returning one Document per page.

    Each Document has metadata: source, page, source_file.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        ValueError: If the PDF yields zero pages.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    if not documents:
        raise ValueError(f"PDF yielded zero pages: {path}")

    for doc in documents:
        doc.metadata["source_file"] = path.name

    return documents


def load_all_pdfs(pdf_paths: dict[str, Path]) -> list[Document]:
    """Load multiple PDFs, tagging each with its document_name.

    Args:
        pdf_paths: Dict mapping document_name -> Path.

    Returns:
        Combined list of all Document objects.
    """
    all_docs = []
    for doc_name, path in pdf_paths.items():
        docs = load_pdf(path)
        for doc in docs:
            doc.metadata["document_name"] = doc_name
        all_docs.extend(docs)
    return all_docs
