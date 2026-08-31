"""PDF text extraction and the uploaded-document Q&A tool built on top of
the same BM25 index used for web research."""
from unittest.mock import MagicMock

import tools


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class TestParsePdf:
    def test_extracts_and_joins_all_pages(self, monkeypatch):
        fake_reader = MagicMock()
        fake_reader.pages = [FakePage("Page one text."), FakePage("Page two text.")]
        monkeypatch.setattr(tools, "PdfReader", lambda f: fake_reader)

        result = tools.parse_pdf(object())
        assert "Page one text." in result
        assert "Page two text." in result

    def test_pages_with_no_extractable_text_become_empty_strings(self, monkeypatch):
        # A scanned/image-only PDF: PdfReader succeeds but extract_text()
        # returns None per page.
        fake_reader = MagicMock()
        fake_reader.pages = [FakePage(None), FakePage(None)]
        monkeypatch.setattr(tools, "PdfReader", lambda f: fake_reader)

        result = tools.parse_pdf(object())
        assert result == ""

    def test_reader_exception_returns_empty_string_not_raised(self, monkeypatch):
        def boom(f):
            raise Exception("corrupted PDF")
        monkeypatch.setattr(tools, "PdfReader", boom)

        result = tools.parse_pdf(object())
        assert result == ""


class TestQueryUploadedDocumentTool:
    def test_no_document_uploaded_returns_friendly_message(self):
        result = tools.query_uploaded_document.func("What was the revenue?")
        assert result == "No data stored yet."

    def test_answers_from_the_uploaded_document_collection_only(self):
        tools.store_in_vector_db(
            ["Fiscal 2025 revenue was 4.2 billion dollars, up 18 percent."],
            collection_name="uploaded_document",
        )
        # A same-named chunk in the *web research* collection must not leak
        # into a document question - they're deliberately separate indexes.
        tools.store_in_vector_db(
            ["Fiscal 2025 revenue was 9.9 billion dollars from a web article."],
            collection_name="financial_research",
        )

        result = tools.query_uploaded_document.func("What was fiscal 2025 revenue?")
        assert "4.2 billion" in result
        assert "9.9 billion" not in result
