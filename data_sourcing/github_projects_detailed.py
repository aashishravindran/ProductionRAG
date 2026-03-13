"""
Generates a detailed, RAG-optimized PDF of Aashish Ravindran's GitHub repositories.

Each repository gets a dedicated section with:
  - Full description and purpose
  - Tech stack & frameworks
  - Key features / architecture highlights
  - Category tags (for semantic retrieval)

Output: data_sourcing/data/github_projects_detailed.pdf

Usage:
    python -m data_sourcing.github_projects_detailed
"""

from __future__ import annotations

import os
from datetime import datetime

from fpdf import FPDF

OUTPUT_PATH = "data_sourcing/data/github_projects_detailed.pdf"

# ---------------------------------------------------------------------------
# Repository data -- curated for RAG retrieval quality
# ---------------------------------------------------------------------------

OWNER = "Aashish Ravindran"
GITHUB_URL = "github.com/aashishravindran"

PROJECTS: list[dict] = [
    {
        "name": "ProductionRAG",
        "url": "https://github.com/aashishravindran/ProductionRAG",
        "category": "GenAI / RAG / Production AI",
        "language": "Python",
        "last_updated": "March 2026",
        "status": "Active",
        "description": (
            "A production-ready Retrieval-Augmented Generation (RAG) application that turns "
            "a personal resume and related documents into an interactive AI chat interface. "
            "Users can ask natural-language questions about skills, experience, projects, and "
            "background and receive grounded, accurate, citation-backed answers."
        ),
        "what_it_does": (
            "Ingests PDFs (resume, LinkedIn profile, GitHub profile, research papers) through "
            "a multi-stage pipeline: PDF loading with PyPDFLoader, per-document-type adaptive "
            "chunking, Hypothetical Document Embedding (HyDE) enrichment at ingest time, and "
            "ChromaDB vector storage. At query time it runs hybrid retrieval -- BM25 keyword "
            "search combined with semantic vector search -- merged via Reciprocal Rank Fusion "
            "(RRF). A cross-encoder reranker (ms-marco-MiniLM-L-6-v2) rescores the top "
            "candidates before a local Ollama LLM (llama3.2) generates the final grounded answer."
        ),
        "tech_stack": [
            "Python 3.10+",
            "LangChain (orchestration)",
            "ChromaDB (vector store)",
            "HuggingFace sentence-transformers -- all-MiniLM-L6-v2 (embeddings)",
            "HuggingFace cross-encoder -- ms-marco-MiniLM-L-6-v2 (reranker)",
            "rank-bm25 (BM25 keyword retrieval)",
            "Ollama + langchain-ollama (local LLM: llama3.2)",
            "pypdf / PyPDFLoader (PDF ingestion)",
            "fpdf2 (PDF generation)",
            "pytest (63-test suite with fake embeddings)",
        ],
        "key_features": [
            "HyDE (Hypothetical Document Embedding): Ollama generates 3 hypothetical questions "
            "per chunk at ingest time, prepended to chunks before embedding to improve "
            "query-document alignment.",
            "Hybrid retrieval: BM25 keyword search + ChromaDB semantic search merged via "
            "Reciprocal Rank Fusion (RRF) for superior recall.",
            "Cross-encoder reranking: cached ms-marco-MiniLM-L-6-v2 rescores top candidates "
            "for precision at low latency.",
            "Per-document-type chunking: 1000-char chunks for profile docs; 500-char with "
            "100-char overlap for resume and research papers.",
            "Grounded generation: hallucination-minimizing system prompts, local LLM inference.",
            "63-test pytest suite using fake embeddings and tmp_path -- no external deps required.",
        ],
    },
    {
        "name": "agentic-fitness-app",
        "url": "https://github.com/aashishravindran/agentic-fitness-app",
        "category": "GenAI / Multi-Agent System / Full-Stack",
        "language": "Python",
        "last_updated": "March 2026",
        "status": "Active",
        "description": (
            "A full-stack AI fitness coaching platform that demonstrates RAG and multi-agent "
            "system design for personalised workout and nutrition guidance. A supervisor agent "
            "routes user requests to specialist worker agents covering strength training, yoga, "
            "HIIT, and kickboxing."
        ),
        "what_it_does": (
            "Users onboard with biometrics and goals via a REST API. A supervisor LangGraph "
            "agent analyses intent and delegates to the appropriate specialist agent (Iron for "
            "strength, Yoga, HIIT, or Kickboxing). Workouts are streamed to the React frontend "
            "(SuperSetUI) in real time over WebSocket. A RAG layer ingests creator philosophy "
            "markdown files, embeds them via Ollama (mxbai-embed-large), and stores them in "
            "ChromaDB -- keeping agent responses grounded in creator voice and methodology. "
            "Fatigue is tracked across sessions with a time-based decay model (3% per hour), "
            "RPE-based accumulation, and automatic rest-day recovery. All state (workout history, "
            "fatigue scores, user profiles) is persisted in per-user SQLite databases."
        ),
        "tech_stack": [
            "Python (FastAPI backend)",
            "LangGraph (multi-agent supervisor/worker architecture)",
            "ChromaDB (RAG vector store)",
            "Ollama -- mxbai-embed-large (embeddings), local LLM inference",
            "SQLite (per-user state persistence)",
            "WebSocket (real-time workout streaming)",
            "Multi-LLM provider support: Gemini, OpenAI, AWS Bedrock, Ollama, DeepSeek",
            "uv (Python package manager)",
            "React 18 + TypeScript (frontend -- SuperSetUI repo)",
            "Tailwind CSS, Framer Motion (UI)",
        ],
        "key_features": [
            "LangGraph supervisor/worker multi-agent pattern: Supervisor routes to Iron "
            "(strength), Yoga, HIIT, and Kickboxing specialist workers.",
            "RAG-grounded agents: creator philosophy markdown ingested, chunked, embedded, "
            "and retrieved to keep agent tone and methodology consistent.",
            "Real-time workout streaming via WebSocket to React frontend.",
            "Fatigue tracking system: 3%/hr time-based decay, RPE-based accumulation, "
            "automatic rest-day recovery -- prevents overtraining.",
            "Per-user SQLite persistence: workout history, fatigue scores, onboarding data.",
            "Multi-LLM backend: swap between Gemini, OpenAI, AWS Bedrock, Ollama, or DeepSeek.",
            "AWS CLI-style `superset` CLI tool for all API interactions.",
        ],
    },
    {
        "name": "SuperSetUI",
        "url": "https://github.com/aashishravindran/SuperSetUI",
        "category": "Frontend / React / TypeScript",
        "language": "TypeScript",
        "last_updated": "February 2026",
        "status": "Active",
        "description": (
            "The React 18 frontend companion for agentic-fitness-app. Provides a polished "
            "chat-style interface where users onboard with a conversational flow, pick an AI "
            "coach (Iron, Yoga, HIIT, or Kickboxing), and receive real-time workout sessions "
            "streamed over WebSocket."
        ),
        "what_it_does": (
            "Delivers a full UI flow: landing page, username-based login, chat-style onboarding "
            "that calls backend REST endpoints to capture biometrics and goals, a dashboard with "
            "coach persona cards, a live workout view that opens a WebSocket connection and "
            "renders AGENT_RESPONSE / FINISH_WORKOUT messages in real time, and a workout "
            "history page pulling from the REST API. Smooth animations via Framer Motion and "
            "responsive layout via Tailwind CSS."
        ),
        "tech_stack": [
            "Vite (build tool)",
            "React 18",
            "TypeScript",
            "React Router (client-side routing)",
            "Framer Motion (animations)",
            "Tailwind CSS (styling)",
            "Radix UI -- button/slot primitives",
            "Sonner (toast notifications)",
            "WebSocket (live workout streaming)",
        ],
        "key_features": [
            "Coach persona preview cards: Iron, Yoga, HIIT, Kickboxing with distinct branding.",
            "Chat-style onboarding flow that drives REST calls to backend onboarding endpoints.",
            "Real-time workout session rendered via WebSocket (USER_INPUT / AGENT_RESPONSE / "
            "FINISH_WORKOUT message types).",
            "Workout history page fetching from REST API.",
            "Framer Motion page transitions and Tailwind CSS responsive design.",
            "localStorage-based session management.",
        ],
    },
    {
        "name": "steal-my-agents",
        "url": "https://github.com/aashishravindran/steal-my-agents",
        "category": "AI Tooling / Claude Code / Developer Tools",
        "language": "Shell / Markdown",
        "last_updated": "March 2026",
        "status": "Active",
        "description": (
            "A curated, installable collection of Claude Code sub-agents -- reusable AI agent "
            "definitions (Markdown files with YAML frontmatter) that extend Claude Code's "
            "capabilities. Designed to be stolen and dropped into any project."
        ),
        "what_it_does": (
            "Ships an interactive install.sh script that lets users choose global vs. "
            "per-project installation and install individual agents or all at once. "
            "Each agent is a self-contained .md file describing its purpose, trigger conditions, "
            "available tools, and behavior. The flagship agent is project-wrapup: an end-of-session "
            "agent that autonomously updates documentation, writes conventional-commit messages, "
            "commits, pushes to remote, and produces a clean session summary."
        ),
        "tech_stack": [
            "Shell (bash) -- interactive installer",
            "Markdown / YAML frontmatter (Claude Code agent format)",
            "Claude Code sub-agent framework",
        ],
        "key_features": [
            "project-wrapup agent: end-of-session automation -- docs update, commit, push, summary.",
            "Idempotent installer: skips already-installed agents.",
            "Global or per-project installation modes.",
            "Structured contribution guide for adding new agents.",
            "All agents follow Claude Code's YAML frontmatter spec.",
        ],
    },
    {
        "name": "AiAgents",
        "url": "https://github.com/aashishravindran/AiAgents",
        "category": "GenAI / AI Agents / MCP",
        "language": "Python",
        "last_updated": "June 2025",
        "status": "Active",
        "description": (
            "A demonstration project for building AI agents using the Strands framework with "
            "Model Context Protocol (MCP) tool integration. Shows how to wire custom and "
            "built-in tools into an agent that answers multi-part queries through sequential "
            "tool invocations."
        ),
        "what_it_does": (
            "Defines a Strands-based agent equipped with a suite of tools: current time lookup, "
            "arithmetic calculations, Python code execution, and a custom letter-counting tool. "
            "The agent chains tool calls to answer compound questions, demonstrating the "
            "Strands + MCP agent development pattern on AWS."
        ),
        "tech_stack": [
            "Python",
            "Strands framework (AWS agent framework)",
            "Model Context Protocol (MCP)",
            "AWS CLI (bundled for local AWS integration)",
        ],
        "key_features": [
            "Custom tool authoring with Strands + MCP.",
            "Multi-step tool chaining: time, arithmetic, code execution, custom tools.",
            "AWS Bedrock integration via Strands.",
            "Minimal, educational codebase -- ideal as a Strands/MCP starter reference.",
        ],
    },
    {
        "name": "aashishravindran.github.io",
        "url": "https://github.com/aashishravindran/aashishravindran.github.io",
        "category": "Portfolio / Web / Frontend",
        "language": "HTML / CSS / JavaScript",
        "last_updated": "June 2025",
        "status": "Active",
        "description": (
            "A Netflix-inspired dark-themed personal portfolio website deployed via GitHub Pages. "
            "Showcases professional experience, skills, and education as a polished single-page "
            "application with smooth CSS animations and responsive design."
        ),
        "what_it_does": (
            "Presents professional background -- SDE II at AWS, prior experience at Accenture, "
            "Stony Brook University education -- through interactive modal overlays, animated "
            "sections, and a dark cinematic aesthetic. No frameworks; pure HTML5, CSS3, and "
            "vanilla JavaScript."
        ),
        "tech_stack": [
            "HTML5",
            "CSS3 (custom animations, responsive layout)",
            "Vanilla JavaScript (no frameworks)",
            "GitHub Pages (deployment)",
        ],
        "key_features": [
            "Netflix-inspired dark theme with entry animation.",
            "Interactive modal overlays for detailed experience info.",
            "Fully responsive: desktop, tablet, mobile.",
            "Sections: About, Experience (AWS + Accenture), Skills, Education.",
            "Zero external dependencies -- pure web platform APIs.",
        ],
    },
    {
        "name": "YelpCamp",
        "url": "https://github.com/aashishravindran/YelpCamp",
        "category": "Full-Stack Web / Node.js",
        "language": "JavaScript (Node.js / EJS)",
        "last_updated": "June 2025",
        "status": "Active",
        "description": (
            "A full-stack Yelp-style web application for campgrounds built with Node.js, "
            "Express, and MongoDB. Users can register, browse campgrounds, create new listings, "
            "and leave reviews and comments."
        ),
        "what_it_does": (
            "Implements complete CRUD for campgrounds and comments, user authentication with "
            "Passport.js (register, login, logout), a database seeding script for development, "
            "and a server-side rendered UI with EJS templates and Bootstrap. Follows MVC "
            "structure (models / routes / views / middleware)."
        ),
        "tech_stack": [
            "Node.js",
            "Express.js",
            "MongoDB + Mongoose (ODM)",
            "EJS (server-side templating)",
            "Passport.js (authentication)",
            "Bootstrap (responsive UI)",
        ],
        "key_features": [
            "Full CRUD for campgrounds and comments.",
            "User authentication: register, login, logout via Passport.js.",
            "MVC architecture: models / routes / views / middleware.",
            "Database seeder script for development.",
            "Bootstrap responsive UI.",
        ],
    },
    {
        "name": "CodingInterviewPractice",
        "url": "https://github.com/aashishravindran/CodingInterviewPractice",
        "category": "Data Structures & Algorithms / Interview Prep",
        "language": "Python",
        "last_updated": "July 2025",
        "status": "Active",
        "description": (
            "A personal repository of coding interview practice problems, algorithm "
            "implementations, and data structure exercises. Organized into thematic folders "
            "covering important algorithms, general problem-solving, company-specific prep, "
            "and the complete Striver A2Z DSA sheet."
        ),
        "what_it_does": (
            "Serves as a running notebook of solved problems across arrays, graphs, trees, "
            "dynamic programming, and system design patterns. Covers preparation for top "
            "tech companies (Okta, etc.) and structured curriculum (Striver's A2Z DSA sheet)."
        ),
        "tech_stack": ["Python"],
        "key_features": [
            "ImpAlgos: important algorithmic patterns and implementations.",
            "striver_a2z_dsa: solutions following Striver's A2Z DSA structured curriculum.",
            "Company-specific practice folders (ookta, Others).",
            "Python throughout -- consistent and readable implementations.",
        ],
    },
    {
        "name": "crc_analysis",
        "url": "https://github.com/aashishravindran/crc_analysis",
        "category": "Network Analysis / Research",
        "language": "Python",
        "last_updated": "October 2019",
        "status": "Archived (academic research)",
        "description": (
            "A research-grade analysis tool for identifying and quantifying packet corruption "
            "at a byte level within network frames using CRC (Cyclic Redundancy Check) data. "
            "Results are consolidated at a per-run level for statistical analysis."
        ),
        "what_it_does": (
            "Parses raw CRC data from network capture experiments, computes burst length "
            "statistics to characterise error patterns, consolidates findings across experimental "
            "runs, and visualises results. Part of video multicast research at Stony Brook "
            "University."
        ),
        "tech_stack": ["Python", "matplotlib (visualisation)"],
        "key_features": [
            "Byte-level CRC corruption analysis.",
            "Burst length characterisation.",
            "Per-run consolidated statistics.",
            "Visualisation of error distributions.",
        ],
    },
    {
        "name": "PacketLossAnalysis",
        "url": "https://github.com/aashishravindran/PacketLossAnalysis",
        "category": "Network Analysis / Research / Signal Processing",
        "language": "Python",
        "last_updated": "October 2019",
        "status": "Archived (academic research)",
        "description": (
            "Identifies and characterises frame loss patterns across multiple receivers in a "
            "video multicast experimental setup. Combines three complementary analyses: loss "
            "aggregation (which receivers lost which frames), time synchronisation, and "
            "Probability Mass Function (PMF) modelling of loss bursts."
        ),
        "what_it_does": (
            "Assigns unique receiver-combination identifiers (1-16 for 4 receivers, 2^4) to "
            "each video frame to track which subset of receivers received it. Aligns all "
            "received frames to a global timestamp to compute per-receiver delay. Computes "
            "PMFs of loss burst lengths and inter-loss intervals at both run and per-receiver "
            "granularity. Produces visualisations for all three analysis tracks."
        ),
        "tech_stack": [
            "Python",
            "matplotlib (visualisation)",
            "scipy / numpy (statistical analysis)",
        ],
        "key_features": [
            "Loss aggregation with unique receiver-combination encoding.",
            "Global timestamp alignment for multi-receiver time synchronisation.",
            "PMF analysis of burst length and inter-loss intervals.",
            "Per-run and per-receiver granularity.",
            "Multi-module architecture: separate drivers for each analysis track.",
        ],
    },
    {
        "name": "video_quantification",
        "url": "https://github.com/aashishravindran/video_quantification",
        "category": "Video Processing / Signal Processing / Research",
        "language": "Python",
        "last_updated": "October 2019",
        "status": "Archived (academic research)",
        "description": (
            "Quantifies video quality degradation caused by packet loss in a VMAC (Video "
            "Multicast) system. Compares transmitted and received video frames at raw pixel "
            "level using SSIM (Structural Similarity Index Measure)."
        ),
        "what_it_does": (
            "Loads transmitted and received video frame pairs, computes SSIM scores using "
            "scikit-image, and reports frame-level quality degradation metrics. Provides a "
            "quantitative basis for evaluating the perceptual impact of network packet loss "
            "on video quality in multicast distribution scenarios."
        ),
        "tech_stack": [
            "Python",
            "OpenCV (frame loading and processing)",
            "scikit-image / skimage (SSIM computation)",
        ],
        "key_features": [
            "SSIM-based perceptual quality measurement.",
            "Raw pixel-level frame comparison.",
            "Designed for VMAC multicast loss evaluation.",
            "OpenCV integration for video frame handling.",
        ],
    },
    {
        "name": "PostureRecognition",
        "url": "https://github.com/aashishravindran/PostureRecognition",
        "category": "Machine Learning / Gesture Recognition / Research",
        "language": "MATLAB",
        "last_updated": "September 2019",
        "status": "Archived (academic research)",
        "description": (
            "A machine learning pipeline for posture and gesture recognition developed as part "
            "of an academic research project (likely MobiSys / mobile systems context at "
            "Stony Brook University)."
        ),
        "what_it_does": (
            "Implements a full ML pipeline in MATLAB for classifying body postures and gestures "
            "from sensor data. Contains gesture recognition logic, ML model training and "
            "evaluation pipeline components."
        ),
        "tech_stack": ["MATLAB (ML Pipeline, Gesture Recognition)"],
        "key_features": [
            "End-to-end ML pipeline for posture/gesture classification.",
            "Sensor data processing for mobile systems research.",
            "MATLAB implementation of classification models.",
        ],
    },
    {
        "name": "humanActivityRecognition",
        "url": "https://github.com/aashishravindran/humanActivityRecognition",
        "category": "Android / Mobile / Machine Learning / CPS",
        "language": "Kotlin",
        "last_updated": "July 2019",
        "status": "Archived (academic research)",
        "description": (
            "An Android application for human activity recognition developed as part of a "
            "Cyber-Physical Systems (CPS) course project at Stony Brook University. Uses "
            "on-device sensor data to classify human activities in real time."
        ),
        "what_it_does": (
            "Collects accelerometer and/or gyroscope data from the Android device, processes "
            "it through a classification model, and recognises physical activities in real time "
            "on-device."
        ),
        "tech_stack": [
            "Kotlin (Android)",
            "Android SDK",
            "On-device sensor APIs (accelerometer/gyroscope)",
        ],
        "key_features": [
            "Real-time on-device human activity classification.",
            "Android sensor data collection (accelerometer, gyroscope).",
            "CPS (Cyber-Physical Systems) research project.",
            "Kotlin-first Android development.",
        ],
    },
]

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

COLOR_COVER_BG = (15, 23, 42)       # slate-900
COLOR_COVER_FG = (248, 250, 252)    # slate-50
COLOR_ACCENT = (59, 130, 246)       # blue-500
COLOR_SECTION_BG = (30, 58, 138)    # blue-900
COLOR_SECTION_FG = (255, 255, 255)
COLOR_BODY = (30, 30, 30)
COLOR_SECONDARY = (90, 90, 90)
COLOR_MUTED = (150, 150, 150)
COLOR_SEP = (210, 220, 235)
COLOR_TAG_BG = (239, 246, 255)      # blue-50
COLOR_TAG_FG = (29, 78, 216)        # blue-700


def _s(text: str) -> str:
    """Safe latin-1 encode -- replace common Unicode with ASCII equivalents first."""
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "*",    # bullet
        "\u2026": "...",  # ellipsis
        "\u00b7": "-",    # middle dot
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _section_header(pdf: FPDF, title: str) -> None:
    pdf.set_fill_color(*COLOR_SECTION_BG)
    pdf.set_text_color(*COLOR_SECTION_FG)
    pdf.set_font("Helvetica", style="B", size=10)
    pdf.cell(w=0, h=7, text=f"  {_s(title)}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_text_color(*COLOR_BODY)


def _subsection(pdf: FPDF, title: str) -> None:
    pdf.set_font("Helvetica", style="B", size=9)
    pdf.set_text_color(*COLOR_ACCENT)
    pdf.cell(w=0, h=5, text=_s(title.upper()), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*COLOR_BODY)
    pdf.ln(1)


def _body_text(pdf: FPDF, text: str, indent: float = 0) -> None:
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*COLOR_BODY)
    if indent:
        pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(w=pdf.w - pdf.l_margin - pdf.r_margin - indent, h=5, text=_s(text))


def _bullet(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*COLOR_BODY)
    # bullet symbol + indented text
    x0 = pdf.get_x()
    pdf.cell(w=5, h=5, text=chr(149))  # bullet
    pdf.multi_cell(w=pdf.w - pdf.l_margin - pdf.r_margin - 5, h=5, text=_s(text))
    pdf.set_x(x0)


def _separator(pdf: FPDF, heavy: bool = False) -> None:
    pdf.set_draw_color(*COLOR_SEP)
    lw = 0.5 if heavy else 0.2
    pdf.set_line_width(lw)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)
    pdf.set_line_width(0.2)


def _tag_row(pdf: FPDF, tags: list[str], label: str = "") -> None:
    """Render inline tag pills."""
    pdf.set_font("Helvetica", size=8)
    if label:
        pdf.set_text_color(*COLOR_SECONDARY)
        pdf.cell(w=pdf.get_string_width(label) + 2, h=5, text=_s(label))
    pdf.set_fill_color(*COLOR_TAG_BG)
    pdf.set_text_color(*COLOR_TAG_FG)
    for tag in tags:
        w = pdf.get_string_width(tag) + 6
        pdf.cell(w=w, h=5, text=_s(tag), fill=True, border=0)
        pdf.cell(w=2, h=5, text="")
    pdf.ln(6)
    pdf.set_text_color(*COLOR_BODY)


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------

def _render_cover(pdf: FPDF) -> None:
    pdf.add_page()
    # Dark background
    pdf.set_fill_color(*COLOR_COVER_BG)
    pdf.rect(0, 0, pdf.w, pdf.h, style="F")

    pdf.ln(40)
    pdf.set_font("Helvetica", style="B", size=28)
    pdf.set_text_color(*COLOR_COVER_FG)
    pdf.cell(w=0, h=14, text=_s(OWNER), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", size=13)
    pdf.set_text_color(*COLOR_ACCENT)
    pdf.cell(w=0, h=8, text="GitHub Projects -- Detailed Reference", align="C",
             new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(180, 190, 210)
    pdf.cell(w=0, h=6, text=_s(f"github.com/aashishravindran"), align="C",
             new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_draw_color(*COLOR_ACCENT)
    pdf.set_line_width(1)
    mid = pdf.w / 2
    pdf.line(mid - 40, pdf.get_y(), mid + 40, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(10)

    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(130, 145, 170)
    note = (
        "This document is optimised for RAG (Retrieval-Augmented Generation) ingestion.\n"
        "Each repository section contains a rich description, full tech stack, key features,\n"
        "and architectural context to support high-quality semantic retrieval."
    )
    pdf.multi_cell(w=0, h=6, text=_s(note), align="C")

    pdf.ln(10)
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(100, 115, 140)
    pdf.cell(w=0, h=6,
             text=_s(f"Generated: {datetime.now().strftime('%Y-%m-%d')}  |  13 repositories"),
             align="C", new_x="LMARGIN", new_y="NEXT")


# ---------------------------------------------------------------------------
# Table of contents
# ---------------------------------------------------------------------------

def _render_toc(pdf: FPDF, projects: list[dict]) -> None:
    pdf.add_page()
    _section_header(pdf, "TABLE OF CONTENTS")
    pdf.ln(2)

    active = [p for p in projects if "Archived" not in p["status"]]
    archived = [p for p in projects if "Archived" in p["status"]]

    pdf.set_font("Helvetica", style="B", size=9)
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.cell(w=0, h=5, text="ACTIVE PROJECTS", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    for i, p in enumerate(active, 1):
        pdf.set_font("Helvetica", style="B", size=9)
        pdf.set_text_color(*COLOR_BODY)
        pdf.cell(w=8, h=6, text=f"{i}.")
        pdf.cell(w=70, h=6, text=_s(p["name"]))
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(*COLOR_SECONDARY)
        pdf.cell(w=0, h=6, text=_s(p["category"]), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", style="B", size=9)
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.cell(w=0, h=5, text="ARCHIVED / ACADEMIC PROJECTS", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    for i, p in enumerate(archived, len(active) + 1):
        pdf.set_font("Helvetica", style="B", size=9)
        pdf.set_text_color(*COLOR_BODY)
        pdf.cell(w=8, h=6, text=f"{i}.")
        pdf.cell(w=70, h=6, text=_s(p["name"]))
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(*COLOR_SECONDARY)
        pdf.cell(w=0, h=6, text=_s(p["category"]), new_x="LMARGIN", new_y="NEXT")


# ---------------------------------------------------------------------------
# Project page
# ---------------------------------------------------------------------------

def _render_project(pdf: FPDF, p: dict, number: int) -> None:
    pdf.add_page()

    # ── Project header ──
    pdf.set_fill_color(*COLOR_COVER_BG)
    pdf.rect(0, 0, pdf.w, 28, style="F")

    pdf.set_y(8)
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.set_text_color(*COLOR_COVER_FG)
    pdf.cell(w=12, h=9, text=f"{number}.")
    pdf.cell(w=0, h=9, text=_s(p["name"]), new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(pdf.l_margin + 12)
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*COLOR_ACCENT)
    meta = f"{p['language']}  |  {p['category']}  |  Last updated: {p['last_updated']}  |  {p['status']}"
    pdf.cell(w=0, h=5, text=_s(meta), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    pdf.set_text_color(*COLOR_BODY)

    # ── URL ──
    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(*COLOR_MUTED)
    pdf.cell(w=0, h=5, text=_s(p["url"]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── Description ──
    _subsection(pdf, "Description")
    _body_text(pdf, p["description"])
    pdf.ln(4)

    # ── What it does ──
    _subsection(pdf, "What It Does")
    _body_text(pdf, p["what_it_does"])
    pdf.ln(4)

    # ── Tech stack ──
    _subsection(pdf, "Tech Stack")
    for item in p["tech_stack"]:
        _bullet(pdf, item)
    pdf.ln(3)

    # ── Key features ──
    _subsection(pdf, "Key Features & Architecture")
    for feat in p["key_features"]:
        _bullet(pdf, feat)
    pdf.ln(3)

    # ── Category tags ──
    tags = [t.strip() for t in p["category"].split("/")]
    tags.append(p["language"].split()[0])
    tags.append(p["status"].split()[0])
    _tag_row(pdf, tags, label="Tags: ")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate() -> None:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=15, top=12, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)

    _render_cover(pdf)
    _render_toc(pdf, PROJECTS)

    for i, project in enumerate(PROJECTS, 1):
        _render_project(pdf, project, i)

    # Page numbers
    total = len(pdf.pages)
    for page_num in range(3, total + 1):  # skip cover + TOC
        pdf.page = page_num
        pdf.set_y(-12)
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(*COLOR_MUTED)
        pdf.cell(
            w=0, h=5,
            text=_s(f"Aashish Ravindran -- GitHub Projects  |  Page {page_num} of {total}  |  {GITHUB_URL}"),
            align="C",
        )

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    pdf.output(OUTPUT_PATH)
    print(f"PDF saved: {OUTPUT_PATH}")
    print(f"  {len(PROJECTS)} repositories documented")
    print(f"  {total} pages total")


if __name__ == "__main__":
    generate()
