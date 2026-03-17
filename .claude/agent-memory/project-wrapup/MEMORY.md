# ProductionRAG -- Project Memory

## Project Structure
- Root: `/Users/raashish/Documents/projects/ProductionRAG`
- `data_sourcing/` -- data gathering module (GitHub API to PDF, validation, github_projects_detailed.py)
- `data_sourcing/data/` -- source PDFs (tracked in git): github_projects_detailed, Resume, Gen_ai_divide
- `ingestion/` -- data ingestion pipeline (loader, chunker, rich metadata, questions, store, pipeline orchestrator)
- `retrieval/` -- hybrid retrieval + generation module (query_analyzer, BM25, vector, RRF, cross-encoder, ChatOllama)
- `tests/` -- pytest test suite for ingestion + retrieval (89 tests)
- `CLAUDE.md` -- lean project log (brief session entries only)
- `.claude/skills/project-structure.md` -- detailed project structure (single source of truth)

## Tech Stack
- Python 3, LangChain ecosystem, ChromaDB, HuggingFace sentence-transformers
- Embeddings: all-MiniLM-L6-v2 (384 dimensions)
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (cached via @lru_cache)
- BM25: rank-bm25 with stopword removal
- Chunking: RecursiveCharacterTextSplitter, per-type sizes (projects: 1000/200, resume: 800/200, research: 1500/200)
- Hypothetical question enrichment on projects chunks (Ollama-generated, prepended to page_content)
- Pre-retrieval: LLM query analyzer (doc_types, resume_section, project_name) -> ChromaDB metadata filter

## Conventions
- Commit style: conventional commits (`type(scope): description`)
- Co-authored commits use `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
- Remote: `origin` at `https://github.com/aashishravindran/ProductionRAG.git`
- Branch: `master` (main branch)

## Notes
- Git credential helper has a minor config typo ("credential-stor" vs "credential-store") -- causes a warning but push works fine
- Source PDFs are binary files committed directly to the repo
- `.claude/` directory IS tracked in git (fixed in commit 78d4a32)
- Tests use FakeEmbeddings (384-dim constant vectors), mocked CrossEncoder, and tmp_path fixtures to avoid external deps
- conftest.py has autouse fixture to clear cross-encoder lru_cache between tests
- retriever.py and rag_chain.py support `skip_analysis=True` param to bypass query analyzer in tests
- metadata.py uses page-number-to-project mapping from github_projects_detailed.py PROJECTS list
- Resume section detection uses regex with `   +` pattern for whitespace-heavy PDF text
