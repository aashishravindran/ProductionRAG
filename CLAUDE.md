# ProductionRAG -- Development Log

## Project Overview
A production-ready RAG application that turns a personal resume into an interactive chat interface, grounded in real data sources (GitHub profile, LinkedIn profile, supplementary PDFs).

## Project Structure
- `data_sourcing/` -- Module for gathering and validating source documents
  - `github_to_pdf.py` -- Fetches GitHub API data and generates a structured PDF
  - `validate_sources.py` -- Confirms all required PDFs exist and are valid
  - `data/` -- Contains source PDF files (github_profile, linkedin_profile, Gen_ai_divide)
- `requirements.txt` -- Python dependencies (requests, fpdf2, python-dotenv)
- `.env.example` -- Template for environment variables (GITHUB_TOKEN, GITHUB_USERNAME)

## Tech Stack
- Python 3
- requests (GitHub API)
- fpdf2 (PDF generation)
- python-dotenv (environment config)

## Conventions
- Commit messages follow conventional commit style: `type(scope): description`
- Co-authored commits with Claude include the Co-Authored-By trailer
- Source PDFs are tracked in git (binary files in data_sourcing/data/)

---

## Session: 2026-03-03

### Summary
Built the data sourcing module for the ProductionRAG pipeline. This included a GitHub-to-PDF generator that fetches profile, repository, language, and contribution data from the GitHub API and renders it as a structured PDF. A source validation script was created to verify all required PDFs are present and valid. Three source documents were added: GitHub profile, LinkedIn profile, and a Gen AI supplementary PDF.

### Changes Made
- `data_sourcing/__init__.py` -- Created module init
- `data_sourcing/github_to_pdf.py` -- 530-line GitHub API fetcher and PDF generator
- `data_sourcing/validate_sources.py` -- PDF validation utility with magic byte checking and size validation
- `data_sourcing/data/github_profile.pdf` -- Generated GitHub profile PDF
- `data_sourcing/data/linkedin_profile.pdf` -- LinkedIn profile export (manually sourced)
- `data_sourcing/data/Gen_ai_divide.pdf` -- Supplementary Gen AI PDF document
- `requirements.txt` -- Added requests, fpdf2, python-dotenv
- `.env.example` -- Added GITHUB_TOKEN and GITHUB_USERNAME placeholders
- `.gitignore` -- Configured to exclude .env, __pycache__, venv, etc.

### Key Decisions
- GitHub profile data is fetched via API and rendered to PDF rather than scraped, ensuring reliability and respecting rate limits
- PDF validation uses magic byte checking (PDF header) plus minimum size threshold to catch corrupt or empty files
- Source PDFs are committed directly to the repo for simplicity at this stage

### Known Issues / Next Steps
- Resume data ingestion pipeline (chunking and parsing PDFs) not yet implemented
- Vector store integration needed
- Retrieval and reranking pipeline needed
- No tests written yet for the data sourcing module
- LinkedIn profile PDF must be manually exported and placed in the data directory
