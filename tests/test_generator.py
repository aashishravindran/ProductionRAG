"""Tests for retrieval.generator module."""
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from retrieval.generator import build_llm, format_context, generate


class TestFormatContext:
    def test_formats_single_doc(self):
        docs = [
            Document(
                page_content="Some content here.",
                metadata={"source_file": "resume.pdf", "page": 0},
            )
        ]
        result = format_context(docs)
        assert "[1]" in result
        assert "resume.pdf" in result
        assert "Some content here." in result

    def test_formats_multiple_docs(self):
        docs = [
            Document(page_content="First.", metadata={"source_file": "a.pdf", "page": 0}),
            Document(page_content="Second.", metadata={"source_file": "b.pdf", "page": 1}),
        ]
        result = format_context(docs)
        assert "[1]" in result
        assert "[2]" in result
        assert "a.pdf" in result
        assert "b.pdf" in result

    def test_handles_missing_metadata(self):
        docs = [Document(page_content="No metadata.", metadata={})]
        result = format_context(docs)
        assert "unknown" in result
        assert "?" in result


class TestBuildLlm:
    def test_returns_chat_ollama(self):
        llm = build_llm(model="llama3.2:latest", base_url="http://localhost:11434")
        assert isinstance(llm, ChatOllama)


class TestGenerate:
    @patch("retrieval.generator.ChatOllama")
    def test_generate_calls_llm(self, mock_ollama_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="The answer is 42.")
        mock_ollama_cls.return_value = mock_llm

        docs = [
            Document(
                page_content="some context",
                metadata={"source_file": "a.pdf", "page": 0},
            )
        ]
        result = generate("what is it?", docs)

        assert result == "The answer is 42."
        mock_llm.invoke.assert_called_once()
        messages = mock_llm.invoke.call_args[0][0]
        assert len(messages) == 2  # system + human
