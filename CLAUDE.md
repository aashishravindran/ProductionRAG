# ProductionRAG -- Development Log

## Project Overview
A production-ready RAG application that turns a personal resume into an interactive chat interface, grounded in real data sources (GitHub profile, LinkedIn profile, supplementary PDFs).

## Project Structure
- `data_sourcing/` -- Module for gathering and validating source documents
  - `github_to_pdf.py` -- Fetches GitHub API data and generates a structured PDF
  - `validate_sources.py` -- Confirms all required PDFs exist and are valid
  - `data/` -- Contains source PDF files (github_profile, linkedin_profile, Gen_ai_divide)
- `ingestion/` -- Data ingestion pipeline (PDF loading, chunking, metadata, vector store)
  - `config.py` -- Pipeline configuration constants (paths, chunk sizes, document types)
  - `loader.py` -- PDF loading with page-level metadata via PyPDFLoader
  - `chunker.py` -- RecursiveCharacterTextSplitter-based document chunking
  - `metadata.py` -- Metadata enrichment (document types, chunk indexing, citations)
  - `store.py` -- ChromaDB vector store creation and loading
  - `pipeline.py` -- Orchestrator that chains load -> chunk -> enrich -> store
- `tests/` -- Test suite for the ingestion pipeline
  - `conftest.py` -- Shared fixtures (sample documents, fake embeddings, temp PDFs)
  - `test_loader.py`, `test_chunker.py`, `test_metadata.py`, `test_store.py`, `test_pipeline.py`
- `requirements.txt` -- Python dependencies
- `.env.example` -- Template for environment variables (GITHUB_TOKEN, GITHUB_USERNAME)

## Tech Stack
- Python 3
- requests (GitHub API)
- fpdf2 (PDF generation)
- python-dotenv (environment config)
- LangChain (document loading, text splitting, embeddings interface)
- ChromaDB (vector store)
- HuggingFace sentence-transformers (embeddings -- all-MiniLM-L6-v2)
- pypdf (PDF parsing backend)
- pytest (testing)

## Conventions
- Commit messages follow conventional commit style: `type(scope): description`
- Co-authored commits with Claude include the Co-Authored-By trailer
- Source PDFs are tracked in git (binary files in data_sourcing/data/)

---

## Session: 2026-03-03

### Summary
Built the data sourcing module for the ProductionRAG pipeline. This included a GitHub-to-PDF generator that fetches profile, repository, language, and contribution data from the GitHub API and renders it as a structured PDF. A source validation script was created to verify all required PDFs are present and valid. Three source documents were added: GitHub profile, LinkedIn profile, and a Gen AI supplementary PDF.

### Changes Made
- `data_sourcing/__init__.py` -- Created module init
- `data_sourcing/github_to_pdf.py` -- 530-line GitHub API fetcher and PDF generator
- `data_sourcing/validate_sources.py` -- PDF validation utility with magic byte checking and size validation
- `data_sourcing/data/github_profile.pdf` -- Generated GitHub profile PDF
- `data_sourcing/data/linkedin_profile.pdf` -- LinkedIn profile export (manually sourced)
- `data_sourcing/data/Gen_ai_divide.pdf` -- Supplementary Gen AI PDF document
- `requirements.txt` -- Added requests, fpdf2, python-dotenv
- `.env.example` -- Added GITHUB_TOKEN and GITHUB_USERNAME placeholders
- `.gitignore` -- Configured to exclude .env, __pycache__, venv, etc.

### Key Decisions
- GitHub profile data is fetched via API and rendered to PDF rather than scraped, ensuring reliability and respecting rate limits
- PDF validation uses magic byte checking (PDF header) plus minimum size threshold to catch corrupt or empty files
- Source PDFs are committed directly to the repo for simplicity at this stage

### Known Issues / Next Steps
- Resume data ingestion pipeline (chunking and parsing PDFs) not yet implemented
- Vector store integration needed
- Retrieval and reranking pipeline needed
- No tests written yet for the data sourcing module
- LinkedIn profile PDF must be manually exported and placed in the data directory

## Session: 2026-03-03 (continued)

### Summary
Built the complete data ingestion pipeline that loads source PDFs, chunks them with LangChain's RecursiveCharacterTextSplitter, enriches metadata (document type, chunk indexing, citations), and stores embeddings in a ChromaDB vector store. Also added a comprehensive test suite covering all pipeline components with unit and integration tests.

### Changes Made
- `ingestion/__init__.py` -- Created module init with public API exports
- `ingestion/config.py` -- Pipeline configuration: source PDF paths, ChromaDB settings, chunk size (500) and overlap (100)
- `ingestion/loader.py` -- PDF loading via PyPDFLoader with page-level and source metadata
- `ingestion/chunker.py` -- Document chunking with RecursiveCharacterTextSplitter, metadata preservation
- `ingestion/metadata.py` -- Metadata enrichment (document_type, chunk_index) and citation formatting
- `ingestion/store.py` -- ChromaDB vector store creation and loading with persistence
- `ingestion/pipeline.py` -- End-to-end orchestrator: load -> chunk -> enrich -> store
- `tests/__init__.py` -- Test package init
- `tests/conftest.py` -- Shared fixtures: sample documents, fake embeddings, temporary PDF generation
- `tests/test_loader.py` -- Tests for PDF loading (valid, missing, multi-file)
- `tests/test_chunker.py` -- Tests for chunking (short docs, long docs, empty list, metadata preservation)
- `tests/test_metadata.py` -- Tests for enrichment (chunk indexing, document types, citation formatting)
- `tests/test_store.py` -- Tests for vector store (create, persist, load, metadata round-trip)
- `tests/test_pipeline.py` -- Integration tests for full pipeline (end-to-end, error handling, metadata flow)
- `requirements.txt` -- Added langchain, langchain-community, langchain-chroma, langchain-huggingface, sentence-transformers, langchain-text-splitters, chromadb, pypdf, pytest
- `.gitignore` -- Added chroma_db/ to ignore list

### Key Decisions
- Used LangChain as the framework for document loading, splitting, and vector store integration for consistency and extensibility
- Chose ChromaDB as the vector store for its simplicity and local persistence capability
- Selected sentence-transformers/all-MiniLM-L6-v2 as the default embedding model (lightweight, good general-purpose performance)
- Chunk size of 500 characters with 100 overlap balances context preservation with retrieval granularity
- Pipeline is fully parameterized -- all defaults can be overridden for testing and flexibility
- Tests use fake embeddings and temp directories to avoid external dependencies

### Known Issues / Next Steps
- Retrieval and reranking pipeline not yet implemented
- LLM integration (generation step) needed
- Chat interface / API layer not yet built
- No tests for the data_sourcing module
- LinkedIn profile PDF must still be manually exported
