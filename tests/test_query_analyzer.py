"""Tests for retrieval.query_analyzer and metadata filter building."""
import json
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from retrieval.retriever import _build_metadata_filter


class TestBuildMetadataFilter:
    def test_doc_types_only(self):
        analysis = {"doc_types": ["resume"], "resume_section": None, "project_name": None}
        result = _build_metadata_filter(analysis)
        assert result == {"document_type": {"$in": ["resume"]}}

    def test_multiple_doc_types(self):
        analysis = {"doc_types": ["resume", "projects"], "resume_section": None, "project_name": None}
        result = _build_metadata_filter(analysis)
        assert result == {"document_type": {"$in": ["resume", "projects"]}}

    def test_doc_types_with_resume_section(self):
        analysis = {"doc_types": ["resume"], "resume_section": "Skills", "project_name": None}
        result = _build_metadata_filter(analysis)
        assert result == {
            "$and": [
                {"document_type": {"$in": ["resume"]}},
                {"resume_section": "Skills"},
            ]
        }

    def test_doc_types_with_project_name(self):
        analysis = {"doc_types": ["projects"], "resume_section": None, "project_name": "ProductionRAG"}
        result = _build_metadata_filter(analysis)
        assert result == {
            "$and": [
                {"document_type": {"$in": ["projects"]}},
                {"project_name": "ProductionRAG"},
            ]
        }

    def test_no_filters_returns_none(self):
        analysis = {"doc_types": None, "resume_section": None, "project_name": None}
        result = _build_metadata_filter(analysis)
        assert result is None

    def test_empty_analysis_returns_none(self):
        result = _build_metadata_filter({})
        assert result is None


class TestAnalyzeQuery:
    @patch("retrieval.query_analyzer.ChatOllama")
    def test_resume_query(self, mock_ollama_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "doc_types": ["resume"],
            "resume_section": "Skills",
            "project_name": None,
        })
        mock_llm.invoke.return_value = mock_response
        mock_ollama_cls.return_value = mock_llm

        from retrieval.query_analyzer import analyze_query

        result = analyze_query("What are my skills?")
        assert result["doc_types"] == ["resume"]
        assert result["resume_section"] == "Skills"

    @patch("retrieval.query_analyzer.ChatOllama")
    def test_project_query(self, mock_ollama_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "doc_types": ["projects"],
            "resume_section": None,
            "project_name": "ProductionRAG",
        })
        mock_llm.invoke.return_value = mock_response
        mock_ollama_cls.return_value = mock_llm

        from retrieval.query_analyzer import analyze_query

        result = analyze_query("Tell me about ProductionRAG")
        assert result["doc_types"] == ["projects"]
        assert result["project_name"] == "ProductionRAG"

    @patch("retrieval.query_analyzer.ChatOllama")
    def test_invalid_json_returns_no_filters(self, mock_ollama_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "not valid json"
        mock_llm.invoke.return_value = mock_response
        mock_ollama_cls.return_value = mock_llm

        from retrieval.query_analyzer import analyze_query

        result = analyze_query("random query")
        assert result["doc_types"] is None
        assert result["resume_section"] is None
        assert result["project_name"] is None

    @patch("retrieval.query_analyzer.ChatOllama")
    def test_invalid_doc_types_filtered(self, mock_ollama_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "doc_types": ["resume", "invalid_type"],
            "resume_section": None,
            "project_name": None,
        })
        mock_llm.invoke.return_value = mock_response
        mock_ollama_cls.return_value = mock_llm

        from retrieval.query_analyzer import analyze_query

        result = analyze_query("experience")
        assert result["doc_types"] == ["resume"]

    @patch("retrieval.query_analyzer.ChatOllama")
    def test_all_invalid_doc_types_returns_none(self, mock_ollama_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "doc_types": ["bad1", "bad2"],
            "resume_section": None,
            "project_name": None,
        })
        mock_llm.invoke.return_value = mock_response
        mock_ollama_cls.return_value = mock_llm

        from retrieval.query_analyzer import analyze_query

        result = analyze_query("query")
        assert result["doc_types"] is None
