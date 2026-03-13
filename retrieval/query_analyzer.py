"""Pre-retrieval query analysis: classifies user queries to extract metadata filters.

An LLM examines the query and returns structured JSON indicating which document
types and metadata fields are most relevant, allowing the retriever to narrow
the candidate pool before BM25/vector search.
"""
import json

from langchain_ollama import ChatOllama

from .config import OLLAMA_MODEL, OLLAMA_BASE_URL

ANALYSIS_PROMPT = """\
You are a query analyzer for a personal RAG system containing these document types:

- "resume": Work experience, education, skills, certifications, professional summary
- "projects": GitHub repositories with descriptions, tech stacks, features, architecture
- "research": Academic research papers (GenAI adoption, network analysis, etc.)

Given a user query, return a JSON object with:
- "doc_types": list of relevant document types (1-3 from the list above)
- "resume_section": if resume is relevant, which section? One of: "Professional Summary", "Experience", "Education", "Skills", or null
- "project_name": if asking about a specific project, the project name, or null

Rules:
- Personal questions (skills, experience, education, background) → ["resume"]
- Project questions (repos, tech stack, architecture) → ["projects"]
- Questions that span both (e.g. "what technologies do you use") → ["resume", "projects"]
- Research/paper questions → ["research"]
- Broad questions ("tell me about yourself") → ["resume", "projects"]
- When unsure, include all types

Return ONLY valid JSON, no explanation.

Query: {query}
JSON:"""


def analyze_query(
    query: str,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> dict:
    """Analyze a query and return structured retrieval filters.

    Returns:
        {
            "doc_types": ["resume", "projects"],
            "resume_section": "Experience" | None,
            "project_name": "ProductionRAG" | None,
        }
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=0)
    prompt = ANALYSIS_PROMPT.format(query=query)
    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content.strip())
    except (json.JSONDecodeError, AttributeError):
        # Fallback: no filtering
        return {"doc_types": None, "resume_section": None, "project_name": None}

    # Validate doc_types
    valid_types = {"resume", "projects", "research"}
    doc_types = result.get("doc_types")
    if isinstance(doc_types, list):
        doc_types = [t for t in doc_types if t in valid_types]
        if not doc_types:
            doc_types = None
    else:
        doc_types = None

    return {
        "doc_types": doc_types,
        "resume_section": result.get("resume_section"),
        "project_name": result.get("project_name"),
    }
