# ProductionRAG -- Development Log

## Project Overview
A production-ready RAG application that turns a personal resume into an interactive chat interface, grounded in real data sources (GitHub profile, LinkedIn profile, supplementary PDFs).

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
