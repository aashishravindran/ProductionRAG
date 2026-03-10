# ProductionRAG — AI-Powered Resume Chat

A production-ready RAG (Retrieval-Augmented Generation) application that turns my resume into an interactive chat interface. Ask it anything about my experience, skills, and projects — and get accurate, context-grounded answers.

## What Makes This Different

Everyone builds RAG apps. This one is **about me** — a personal resume assistant powered by retrieval-augmented generation, built to production standards. It's both a portfolio piece and a practical demonstration of RAG done right.

## Architecture

```
Source PDFs --> Ingestion Pipeline --> ChromaDB Vector Store
                                            |
User Query --> Embedding --> Semantic Search (top-k) --> Context Formatting --> Ollama LLM --> Answer
```

### Data Ingestion
- **PDF Loading** — PyPDFLoader extracts text from source PDFs (GitHub profile, LinkedIn profile, research papers)
- **Chunking** — RecursiveCharacterTextSplitter (500 chars, 100 overlap) breaks documents into manageable pieces
- **Metadata Enrichment** — Each chunk is tagged with source file, page number, document type, and chunk index
- **Vector Store** — Chunks are embedded and persisted in ChromaDB for fast retrieval

### Search Methodology
- **Vector-based Semantic Search** — Queries and documents are embedded into 384-dimensional vectors using `all-MiniLM-L6-v2` (sentence-transformers). At query time, ChromaDB performs cosine similarity search to find the top-k most relevant chunks. This means searches are meaning-based, not keyword-based — a query like "programming languages" will match chunks mentioning "Python, JavaScript" even without exact word overlap.
- **Top-k Retrieval** — Returns the 4 most relevant chunks by default (configurable)
- **Source Attribution** — Each retrieved chunk carries its source file and page number for traceability

### Generation
- **Ollama (local LLM)** — Uses `llama3.2` running locally via Ollama for answer generation
- **Grounded Responses** — System prompt enforces that answers are based only on retrieved context, minimizing hallucination
- **Structured Output** — Returns both the generated answer and the source documents used

## Tech Stack

- **Python 3** — Core language
- **LangChain** — Document loading, text splitting, embeddings interface, retriever abstraction
- **ChromaDB** — Vector store with local file persistence
- **HuggingFace sentence-transformers** — Embeddings (all-MiniLM-L6-v2, 384 dims)
- **Ollama + langchain-ollama** — Local LLM inference (llama3.2)
- **pypdf / fpdf2** — PDF parsing and generation
- **pytest** — Test suite (37 tests, all using fake embeddings — no external deps needed)

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
- [x] Retrieval pipeline (semantic search)
- [x] LLM-powered answer generation (Ollama)
- [ ] Reranking (optional, to improve retrieval precision)
- [ ] Chat UI
- [ ] Deployment and observability
- [ ] Evaluation and testing suite

## License

MIT
