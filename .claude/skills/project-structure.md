# ProductionRAG -- Project Structure

> Single source of truth for project layout, tech stack, and conventions.
> Last updated: 2026-03-11 (hybrid retrieval session)

## Overview
A production-ready RAG application that turns a personal resume into an interactive chat interface, grounded in real data sources (GitHub profile, LinkedIn profile, supplementary PDFs).

## Directory Layout

```
ProductionRAG/
  .claude/                  -- Claude Code config and skills (not tracked in git)
    skills/
      project-structure.md  -- This file
    settings.local.json     -- Local Claude settings
  data_sourcing/            -- Data gathering module
    __init__.py
    github_to_pdf.py        -- Fetches GitHub API data, renders structured PDF
    validate_sources.py     -- Validates required PDFs (magic bytes, size checks)
    data/                   -- Source PDF files (tracked in git)
      github_profile.pdf
      linkedin_profile.pdf
      Resume.pdf
      Gen_ai_divide.pdf
  retrieval/                -- Retrieval + generation module (hybrid search + Ollama)
    __init__.py             -- Public API exports (ask, retrieve, bm25_search, vector_search, rrf, rerank)
    config.py               -- Ollama model config, retrieval top_k, BM25/vector top_k, reranker model, prompts
    retriever.py            -- Hybrid retrieval: BM25 + vector search, RRF fusion, cross-encoder reranking
    generator.py            -- Ollama generation via ChatOllama (build_llm, format_context, generate)
    rag_chain.py            -- End-to-end orchestrator (ask -> {answer, sources})
  ingestion/                -- Data ingestion pipeline
    __init__.py             -- Public API exports
    config.py               -- Pipeline config (paths, chunk sizes per doc type, doc types)
    loader.py               -- PDF loading via PyPDFLoader with page metadata
    chunker.py              -- RecursiveCharacterTextSplitter-based chunking
    metadata.py             -- Metadata enrichment (doc types, chunk index, citations)
    questions.py            -- Hypothetical question generation via Ollama (profile chunks only)
    store.py                -- ChromaDB vector store creation and loading
    pipeline.py             -- Orchestrator: load -> chunk (per-type sizes) -> enrich -> questions -> store
  tests/                    -- pytest test suite (ingestion + retrieval)
    __init__.py
    conftest.py             -- Shared fixtures (sample docs, fake embeddings, populated_store)
    test_loader.py
    test_chunker.py
    test_metadata.py
    test_store.py
    test_pipeline.py
    test_retriever.py       -- Hybrid retriever + get_retriever tests
    test_hybrid_retrieval.py -- BM25, RRF fusion, reranking, tokenizer tests
    test_generator.py       -- Generator tests
    test_rag_chain.py       -- End-to-end RAG chain tests
  .env.example              -- Env var template (GITHUB_TOKEN, GITHUB_USERNAME)
  .gitignore
  CLAUDE.md                 -- Project log (lean format)
  README.md
  requirements.txt          -- Python dependencies
  chroma_db/                -- ChromaDB persistence (gitignored)
  venv/                     -- Virtual environment (gitignored)
```

## Tech Stack
- Python 3
- requests -- GitHub API access
- fpdf2 -- PDF generation
- python-dotenv -- Environment config
- LangChain ecosystem -- Document loading, text splitting, embeddings interface
- ChromaDB -- Vector store (local persistence)
- HuggingFace sentence-transformers -- Embeddings (all-MiniLM-L6-v2, 384 dims) and cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- rank-bm25 -- BM25Okapi keyword retrieval
- pypdf -- PDF parsing backend
- langchain-ollama -- Ollama LLM integration (ChatOllama)
- pytest -- Testing framework (63 tests)

## Key Configuration
- Chunk sizes: per-document-type (profile: 1000/200, resume: 500/100, research: 500/100)
- Embedding model: all-MiniLM-L6-v2 (384 dimensions)
- Vector store: ChromaDB with local file persistence
- Retrieval: hybrid (BM25 top-10 + vector top-10, RRF fusion, cross-encoder rerank to top-5)
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (cached via @lru_cache)
- BM25 tokenizer: lowercase + stopword removal
- Hypothetical question enrichment: 3 questions per profile chunk (Ollama-generated at ingestion)

## Conventions
- Commit style: conventional commits (`type(scope): description`)
- Co-authored commits include `Co-Authored-By` trailer
- Source PDFs tracked in git; generated artifacts (chroma_db, venv) gitignored
- Tests use FakeEmbeddings (384-dim) and tmp_path fixtures -- no external deps needed
- Remote: `origin` at `https://github.com/aashishravindran/ProductionRAG.git`
- Branch: `master`

## Pipeline Flow
```
Source PDFs --> loader.py --> chunker.py --> metadata.py --> questions.py --> store.py
                (PyPDF)   (per-type split) (enrich meta) (HyDE questions)  (ChromaDB)

Query --> BM25 keyword search ──┐
          Vector semantic search ┘──> RRF Fusion ──> Cross-Encoder Rerank ──> generator.py ──> Answer
```
Ingestion orchestrated by `pipeline.py`; retrieval orchestrated by `rag_chain.py`.

## Status / What Exists
- [x] Data sourcing (GitHub API to PDF, validation)
- [x] Ingestion pipeline (load, chunk, enrich, store)
- [x] Test suite for ingestion
- [x] Hybrid retrieval pipeline (BM25 + vector search, RRF fusion, cross-encoder reranking)
- [x] LLM integration (Ollama via ChatOllama)
- [x] RAG chain orchestrator (ask -> answer + sources)
- [x] Test suite for retrieval + generation (63 tests total)
- [ ] Chat interface / API layer
- [ ] Data sourcing tests
