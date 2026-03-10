"""Retrieval and generation configuration constants."""

# Ollama LLM settings
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieval settings
RETRIEVAL_TOP_K = 4

# Prompt template
SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a person's resume "
    "and professional background. Use ONLY the provided context to answer. "
    "If the context does not contain enough information, say so honestly. "
    "Do not make up information."
)

CONTEXT_TEMPLATE = (
    "Context:\n"
    "---\n"
    "{context}\n"
    "---\n\n"
    "Question: {question}"
)
