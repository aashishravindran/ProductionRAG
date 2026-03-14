"""Generate hypothetical questions for each chunk using Ollama.

This improves retrieval by bridging the query-document gap: user queries
are questions, but chunk content is statements. By prepending hypothetical
questions to each chunk, embeddings capture question-intent and match
better against user queries.
"""
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

QUESTION_PROMPT = (
    "Given the following text chunk from a document, generate exactly 3 short questions "
    "that this chunk could answer. Return ONLY the questions, one per line, no numbering "
    "or bullets.\n\n"
    "Text:\n{text}\n\n"
    "Questions:"
)


def generate_questions_for_chunk(
    text: str,
    llm: ChatOllama,
) -> list[str]:
    """Generate hypothetical questions for a single text chunk."""
    prompt = QUESTION_PROMPT.format(text=text)
    response = llm.invoke(prompt)
    lines = [line.strip() for line in response.content.strip().splitlines() if line.strip()]
    return lines[:3]


def enrich_with_questions(
    chunks: list[Document],
    doc_types: set[str] | None = None,
    model: str = "llama3.2:latest",
    base_url: str = "http://localhost:11434",
) -> list[Document]:
    """Add hypothetical questions to chunks of specified document types.

    Args:
        chunks: All chunks from the pipeline.
        doc_types: Only enrich chunks whose document_type is in this set.
            Defaults to {"projects"} (GitHub projects detailed doc).

    For matching chunks:
    - Stores original content in metadata["original_content"]
    - Stores generated questions in metadata["hypothetical_questions"]
    - Prepends questions to page_content so embeddings capture question-intent
    """
    target_types = doc_types or {"projects", "resume"}
    target_chunks = [c for c in chunks if c.metadata.get("document_type") in target_types]

    if not target_chunks:
        print("  No chunks matched target document types, skipping.")
        return chunks

    llm = ChatOllama(model=model, base_url=base_url)
    total = len(target_chunks)
    print(f"  Enriching {total} chunks (types: {target_types})...")

    for i, chunk in enumerate(target_chunks):
        original = chunk.page_content
        chunk.metadata["original_content"] = original

        questions = generate_questions_for_chunk(original, llm)
        chunk.metadata["hypothetical_questions"] = "\n".join(questions)

        # Prepend questions to content for better embedding
        questions_block = "\n".join(questions)
        chunk.page_content = f"Questions this answers:\n{questions_block}\n\n{original}"

        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"  Generated questions for {i + 1}/{total} chunks...")

    return chunks
