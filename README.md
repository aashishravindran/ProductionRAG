# ProductionRAG — AI-Powered Resume Chat

A production-ready RAG (Retrieval-Augmented Generation) application that turns my resume into an interactive chat interface. Ask it anything about my experience, skills, and projects — and get accurate, context-grounded answers.

## What Makes This Different

Everyone builds RAG apps. This one is **about me** — a personal resume assistant powered by retrieval-augmented generation, built to production standards. It's both a portfolio piece and a practical demonstration of RAG done right.

## Architecture

```
Source PDFs --> Ingestion Pipeline --> ChromaDB Vector Store
                                            |
                    ┌───────────────────────────────────────────┐
User Query ──> BM25 Keyword Search ──┐                         │
               Vector Semantic Search ┘──> RRF Fusion ──> Cross-Encoder Rerank ──> Ollama LLM ──> Answer
```

### Data Ingestion
- **PDF Loading** — PyPDFLoader extracts text from source PDFs (GitHub profile, LinkedIn profile, resume, research papers)
- **Per-type Chunking** — Profile docs use 1000-char chunks to preserve structured sections (e.g. Featured Repos); resume and research docs use 500-char chunks with 100 overlap
- **Metadata Enrichment** — Each chunk is tagged with source file, page number, document type, and chunk index
- **Hypothetical Questions (HyDE)** — For profile documents, Ollama generates 3 hypothetical questions per chunk at ingestion time. These are prepended to the chunk content before embedding, bridging the query-document gap (user questions match better against question-enriched embeddings)
- **Vector Store** — Chunks are embedded using `all-MiniLM-L6-v2` (384 dims) and persisted in ChromaDB

### Search Methodology: Hybrid Retrieval

The retrieval pipeline combines keyword and semantic search for best-of-both-worlds results:

1. **BM25 Keyword Search** — Ranks all chunks by keyword relevance using BM25Okapi with stopword removal. Excels at exact term matches: company names ("Amazon"), technical acronyms ("CRC"), version numbers ("Python 3.10")
2. **Vector Semantic Search** — ChromaDB cosine similarity over 384-dim embeddings. Excels at meaning-based matching: "DevOps automation" finds chunks about "CI/CD pipelines" even without keyword overlap
3. **Reciprocal Rank Fusion (RRF)** — Merges the top-10 results from both BM25 and vector search into a single ranked list. Documents appearing in both lists get boosted scores
4. **Cross-Encoder Reranking** — The fused candidates are re-scored by a cached cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that reads each (query, chunk) pair jointly for precise relevance scoring. The model is loaded once and reused across queries for low latency. Returns the final top-5 results

### Generation
- **Ollama (local LLM)** — Uses `llama3.2` running locally via Ollama for answer generation
- **Grounded Responses** — System prompt enforces that answers are based only on retrieved context, minimizing hallucination
- **Structured Output** — Returns both the generated answer and the source documents used

## Tech Stack

- **Python 3** — Core language
- **LangChain** — Document loading, text splitting, embeddings interface, retriever abstraction
- **ChromaDB** — Vector store with local file persistence
- **HuggingFace sentence-transformers** — Embeddings (all-MiniLM-L6-v2) and cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- **rank-bm25** — BM25Okapi keyword retrieval
- **Ollama + langchain-ollama** — Local LLM inference (llama3.2)
- **pypdf / fpdf2** — PDF parsing and generation
- **pytest** — Test suite (63 tests, all using fake embeddings and mocks — no external deps needed)

## Getting Started

```bash
# Clone the repo
git clone https://github.com/aashishravindran/ProductionRAG.git
cd ProductionRAG

# Set up virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the ingestion pipeline (populates ChromaDB)
python -m ingestion.pipeline

# Ask a question (requires Ollama running locally with llama3.2)
python -m retrieval.rag_chain "What are your key skills?"
```

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed with `llama3.2` model pulled (`ollama pull llama3.2`)

## Roadmap

- [x] Project scaffolding and dependency setup
- [x] Data sourcing (GitHub API to PDF, source validation)
- [x] Resume data ingestion pipeline
- [x] Vector store integration (ChromaDB)
- [x] Retrieval pipeline (hybrid: BM25 + vector search)
- [x] LLM-powered answer generation (Ollama)
- [x] Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- [x] Hypothetical question enrichment (HyDE) for profile docs
- [ ] Chat UI
- [ ] Deployment and observability
- [ ] Evaluation and testing suite

## License

MIT
