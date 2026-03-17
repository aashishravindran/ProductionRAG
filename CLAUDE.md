# ProductionRAG -- Development Log

## Project Overview
A production-ready RAG application that turns a personal resume into an interactive chat interface, grounded in real data sources (detailed GitHub project descriptions, resume, and supplementary PDFs).

For detailed project structure, tech stack, and conventions see: `.claude/skills/project-structure.md`

## Conventions
- Commit style: conventional commits (`type(scope): description`)
- Source PDFs tracked in git; generated artifacts gitignored
- Tests use fake embeddings and tmp_path -- no external deps

---

## Session: 2026-03-03

Built the data sourcing module (GitHub API to PDF generator, source validation) and the complete ingestion pipeline (PDF loading, chunking, metadata enrichment, ChromaDB vector store). Added a comprehensive test suite covering all ingestion components.

## Session: 2026-03-03 (wrap-up)

Restructured project documentation: moved detailed project structure to `.claude/skills/project-structure.md` as single source of truth, trimmed CLAUDE.md to lean format with brief session entries.

## Session: 2026-03-09

Added retrieval and generation module using Ollama (ChatOllama) for the RAG pipeline. Includes ChromaDB retrieval, LLM generation, and an end-to-end orchestrator. Added 12 new tests (37 total passing).

## Session: 2026-03-09 (retrieval quality)

Improved retrieval quality by adding per-document-type chunk sizes, hypothetical question generation for profile chunks at ingestion time, and using original content in LLM context. "Featured Repositories" query went from not-in-top-10 to rank 1; store reduced from 636 duplicate chunks to 153 unique.

## Session: 2026-03-11

Replaced pure vector search with a hybrid retrieval pipeline: BM25 keyword search + vector semantic search, merged via Reciprocal Rank Fusion, then re-scored with a cached cross-encoder reranker. Added stopword removal to BM25 tokenizer to fix irrelevant results on common-word queries. Added Resume.pdf as a new source document. 63 tests passing.

## Session: 2026-03-12

Added LLM-powered query analyzer for pre-retrieval metadata filtering, replaced old profile sources with detailed github_projects_detailed.pdf, and rewrote metadata enrichment with source-specific fields (project details, resume section detection). Tuned chunk sizes for resume and research docs. 89 tests passing.

## Session: 2026-03-16

Added RESEARCH_VALIDATION.md with academic citations (arXiv/ACM papers) backing the RAG architecture decisions: HyDE, hybrid search with RRF, cross-encoder reranking, the drowning effect, and compute/cold-start tradeoffs. Closes GitHub issues #3 and #4.
