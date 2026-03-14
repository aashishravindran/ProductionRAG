"""Pre-retrieval query analysis using embedding-based prototype vector routing.

Instead of calling an LLM to classify queries (slow, ~5-10s on CPU), this module
uses the embedding model already in the pipeline to route queries by cosine
similarity against pre-computed category prototypes (~5ms).

Resume section and project name are extracted via lightweight keyword matching.
"""
from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# Category prototypes -- representative sentences per doc type
# ---------------------------------------------------------------------------

_CATEGORY_EXEMPLARS: dict[str, list[str]] = {
    "resume": [
        "What is your work experience?",
        "Tell me about your education and degree",
        "What are your skills and certifications?",
        "Describe your professional summary",
        "Where have you worked before?",
        "What is your job title and role?",
        "What technologies do you know?",
        "Tell me about your career background",
    ],
    "projects": [
        "Tell me about your GitHub repositories",
        "What projects have you built?",
        "Describe the architecture of your project",
        "What tech stack did you use in your project?",
        "What are the key features of your application?",
        "How does your project work?",
        "What repos do you have on GitHub?",
        "Explain the design of your system",
    ],
    "research": [
        "Tell me about your research papers",
        "What academic research have you done?",
        "Describe your methodology and experiments",
        "What are the findings of your study?",
        "Tell me about GenAI adoption research",
        "What does the research paper discuss?",
        "Summarize the analysis and results",
        "What data did you analyze in your paper?",
    ],
}

# ---------------------------------------------------------------------------
# Resume section keywords
# ---------------------------------------------------------------------------

_SECTION_KEYWORDS: dict[str, list[str]] = {
    "Professional Summary": ["summary", "overview", "introduction", "who are you", "tell me about yourself", "professional background"],
    "Experience": ["experience", "work", "job", "role", "position", "employed", "company", "worked"],
    "Education": ["education", "degree", "university", "college", "school", "graduated", "gpa", "major"],
    "Skills": ["skills", "technologies", "tools", "languages", "frameworks", "proficient", "expertise"],
    "Certifications": ["certification", "certified", "certificate", "aws certified", "credential"],
}

# ---------------------------------------------------------------------------
# Project name matching
# ---------------------------------------------------------------------------

_PROJECT_NAMES: list[str] = [
    "ProductionRAG", "agentic-fitness-app", "SuperSetUI", "steal-my-agents",
    "AiAgents", "aashishravindran.github.io", "YelpCamp",
    "CodingInterviewPractice", "crc_analysis", "PacketLossAnalysis",
    "video_quantification", "PostureRecognition", "humanActivityRecognition",
]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


_cached_prototypes: dict[str, np.ndarray] | None = None


def _get_prototypes(embeddings: Embeddings) -> dict[str, np.ndarray]:
    """Compute and cache average embedding vectors for each category.

    Prototypes are computed once on first call, then reused for all subsequent queries.
    """
    global _cached_prototypes
    if _cached_prototypes is not None:
        return _cached_prototypes

    prototypes = {}
    for category, exemplars in _CATEGORY_EXEMPLARS.items():
        vectors = embeddings.embed_documents(exemplars)
        prototypes[category] = np.mean(vectors, axis=0)
    _cached_prototypes = prototypes
    return prototypes


def _classify_doc_types(
    query: str,
    embeddings: Embeddings,
    threshold: float = 0.4,
) -> list[str] | None:
    """Classify query into doc types by cosine similarity to category prototypes.

    Returns doc types whose similarity exceeds the threshold.
    If no type passes the threshold, returns None (search all).
    """
    prototypes = _get_prototypes(embeddings)
    query_vec = np.array(embeddings.embed_query(query))

    scores = {
        category: _cosine_similarity(query_vec, proto)
        for category, proto in prototypes.items()
    }

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Include top type always, plus any others within threshold gap of top
    top_score = ranked[0][1]
    if top_score < threshold:
        return None

    # Include types that are close to the top score (within 0.1)
    gap = 0.1
    result = [cat for cat, score in ranked if score >= top_score - gap and score >= threshold]
    return result if result else None


def _detect_resume_section(query: str) -> str | None:
    """Match query against resume section keywords."""
    query_lower = query.lower()
    best_section = None
    best_count = 0

    for section, keywords in _SECTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in query_lower)
        if count > best_count:
            best_count = count
            best_section = section

    return best_section if best_count > 0 else None


def _detect_project_name(query: str) -> str | None:
    """Match query against known project names (case-insensitive)."""
    query_lower = query.lower()
    for name in _PROJECT_NAMES:
        if name.lower() in query_lower:
            return name
    return None


def analyze_query(
    query: str,
    embeddings: Embeddings | None = None,
) -> dict:
    """Analyze a query using embedding-based routing and keyword matching.

    Uses cosine similarity against category prototypes for doc_type classification
    (~5ms) instead of an LLM call (~5-10s).

    Returns:
        {
            "doc_types": ["resume", "projects"] | None,
            "resume_section": "Experience" | None,
            "project_name": "ProductionRAG" | None,
        }
    """
    if embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    doc_types = _classify_doc_types(query, embeddings)
    resume_section = _detect_resume_section(query)
    project_name = _detect_project_name(query)

    # If a project name is detected, ensure "projects" is in doc_types
    if project_name:
        if doc_types is None:
            doc_types = ["projects"]
        elif "projects" not in doc_types:
            doc_types.append("projects")

    # If a resume section is detected, ensure "resume" is in doc_types
    if resume_section:
        if doc_types is None:
            doc_types = ["resume"]
        elif "resume" not in doc_types:
            doc_types.append("resume")

    return {
        "doc_types": doc_types,
        "resume_section": resume_section,
        "project_name": project_name,
    }
