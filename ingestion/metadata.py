"""Enrich chunk metadata with document type, chunk indexing, and source-specific fields."""
import re

from langchain_core.documents import Document

from .config import DOCUMENT_TYPES


# ---------------------------------------------------------------------------
# Project metadata (page → project mapping from github_projects_detailed.pdf)
# ---------------------------------------------------------------------------

def _build_project_page_map() -> dict[int, dict]:
    """Build a page-number → project-metadata map from the generator module."""
    from data_sourcing.github_projects_detailed import PROJECTS

    # Page 0 = cover, Page 1 = TOC, Pages 2..N = one project per page
    page_map: dict[int, dict] = {}
    for i, proj in enumerate(PROJECTS):
        page = i + 2  # projects start at page 2
        page_map[page] = {
            "project_name": proj["name"],
            "project_category": proj["category"],
            "project_language": proj["language"],
            "project_status": proj["status"],
            "project_url": proj["url"],
        }
    return page_map


_PROJECT_PAGE_MAP: dict[int, dict] | None = None


def _get_project_page_map() -> dict[int, dict]:
    global _PROJECT_PAGE_MAP
    if _PROJECT_PAGE_MAP is None:
        _PROJECT_PAGE_MAP = _build_project_page_map()
    return _PROJECT_PAGE_MAP


# ---------------------------------------------------------------------------
# Resume section detection
# ---------------------------------------------------------------------------

_RESUME_SECTIONS = [
    "Professional Summary",
    "Experience",
    "Education",
    "Skills",
    "Certifications",
    "Projects",
]

# Regex matches section headers allowing for whitespace variations in PDF text.
# PDF extraction often places headers after long runs of spaces (not newlines),
# so we match after either a newline or 3+ consecutive spaces.
_SECTION_PATTERN = re.compile(
    r"(?:^|\n|   +)\s*(" + "|".join(re.escape(s) for s in _RESUME_SECTIONS) + r")\b",
    re.IGNORECASE,
)


def _detect_resume_section(text: str) -> str:
    """Detect which resume section a chunk belongs to based on content."""
    matches = _SECTION_PATTERN.findall(text)
    if matches:
        # Return the last section header found (most specific to this chunk)
        return matches[-1].strip().title()
    return "General"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_metadata(
    chunks: list[Document],
    document_types: dict[str, str] | None = None,
) -> list[Document]:
    """Add document_type, chunk_index, and source-specific metadata to chunks.

    Enrichments by document type:
        - projects: project_name, project_category, project_language,
                    project_status, project_url (mapped from page number)
        - resume: resume_section (detected from content)
        - all: document_type, chunk_index
    """
    type_map = document_types or DOCUMENT_TYPES

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        doc_name = chunk.metadata.get("document_name", "")
        doc_type = type_map.get(doc_name, "unknown")
        chunk.metadata["document_type"] = doc_type

        # Source-specific enrichment
        if doc_type == "projects":
            page = chunk.metadata.get("page", -1)
            proj_meta = _get_project_page_map().get(page, {})
            chunk.metadata.update(proj_meta)

        elif doc_type == "resume":
            chunk.metadata["resume_section"] = _detect_resume_section(
                chunk.page_content
            )

    # Second pass: propagate resume sections forward through consecutive
    # resume chunks so bullet-point chunks inherit the preceding header.
    last_section = "General"
    for chunk in chunks:
        if chunk.metadata.get("document_type") != "resume":
            continue
        section = chunk.metadata.get("resume_section", "General")
        if section != "General":
            last_section = section
        else:
            chunk.metadata["resume_section"] = last_section

    return chunks


def format_citation(metadata: dict) -> str:
    """Format chunk metadata into a human-readable citation.

    Example: "Resume.pdf, page 1, chunk 5 [Experience]"
    """
    source_file = metadata.get("source_file", "unknown")
    page = metadata.get("page", "?")
    chunk_index = metadata.get("chunk_index", "?")
    display_page = page + 1 if isinstance(page, int) else page

    citation = f"{source_file}, page {display_page}, chunk {chunk_index}"

    # Add context-specific labels
    project_name = metadata.get("project_name")
    resume_section = metadata.get("resume_section")
    if project_name:
        citation += f" [{project_name}]"
    elif resume_section and resume_section != "General":
        citation += f" [{resume_section}]"

    return citation
