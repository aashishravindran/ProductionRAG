# ProductionRAG -- Project Memory

## Project Structure
- Root: `/Users/raashish/Documents/projects/ProductionRAG`
- `data_sourcing/` -- data gathering module (GitHub API to PDF, validation)
- `data_sourcing/data/` -- source PDFs (tracked in git)
- `ingestion/` -- data ingestion pipeline (loader, chunker, metadata, store, pipeline orchestrator)
- `retrieval/` -- retrieval + generation module (ChromaDB retriever, ChatOllama generator, RAG chain)
- `tests/` -- pytest test suite for ingestion + retrieval (37 tests)
- `CLAUDE.md` -- lean project log (brief session entries only)
- `.claude/skills/project-structure.md` -- detailed project structure (single source of truth)

## Tech Stack
- Python 3, LangChain ecosystem, ChromaDB, HuggingFace sentence-transformers
- Embeddings: all-MiniLM-L6-v2 (384 dimensions)
- Chunking: RecursiveCharacterTextSplitter, per-type sizes (profile: 1000/200, research: 500/100)
- Hypothetical question enrichment on profile chunks (Ollama-generated, prepended to page_content)

## Conventions
- Commit style: conventional commits (`type(scope): description`)
- Co-authored commits use `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
- Remote: `origin` at `https://github.com/aashishravindran/ProductionRAG.git`
- Branch: `master` (main branch)

## Notes
- Git credential helper has a minor config typo ("credential-stor" vs "credential-store") -- causes a warning but push works fine
- Source PDFs are binary files committed directly to the repo
- `.claude/` directory IS tracked in git (fixed in commit 78d4a32)
- Tests use FakeEmbeddings (384-dim constant vectors) and tmp_path fixtures to avoid external deps
