"""Generate answers using Ollama via ChatOllama."""
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from .config import OLLAMA_MODEL, OLLAMA_BASE_URL, SYSTEM_PROMPT, CONTEXT_TEMPLATE


def build_llm(
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> ChatOllama:
    """Create a ChatOllama instance."""
    return ChatOllama(model=model, base_url=base_url)


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a single context string.

    Each chunk is separated by a blank line and prefixed with its
    source metadata for traceability.
    """
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        # Use original_content if available (before hypothetical questions were prepended)
        content = doc.metadata.get("original_content", doc.page_content)
        parts.append(f"[{i}] (source: {source}, page: {page})\n{content}")
    return "\n\n".join(parts)


def generate(
    query: str,
    context_documents: list[Document],
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """Generate an answer given a query and retrieved context documents.

    Returns the LLM response as a plain string.
    """
    llm = build_llm(model=model, base_url=base_url)
    context_str = format_context(context_documents)
    user_message = CONTEXT_TEMPLATE.format(context=context_str, question=query)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    return response.content
