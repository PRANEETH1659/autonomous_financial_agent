"""scrape_website and its two backends: Firecrawl (preferred, handles
JS-rendered pages) and a raw requests+BeautifulSoup fallback."""
from unittest.mock import MagicMock

import pytest
import requests

import tools


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, json_raises=False, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self._json_raises = json_raises
        self.text = text

    def json(self):
        if self._json_raises:
            raise ValueError("not json")
        return self._json_data


class TestScrapeWithFirecrawl:
    def test_success_returns_markdown(self, monkeypatch):
        monkeypatch.setattr(
            tools.requests, "post",
            lambda *a, **k: FakeResponse(200, {"data": {"markdown": "# Hello"}}),
        )
        result = tools._scrape_with_firecrawl("https://x.com", "key", 10)
        assert result == "# Hello"

    def test_non_200_returns_none(self, monkeypatch):
        monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse(500))
        assert tools._scrape_with_firecrawl("https://x.com", "key", 10) is None

    def test_network_error_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(tools.requests, "post", boom)
        assert tools._scrape_with_firecrawl("https://x.com", "key", 10) is None

    def test_non_json_body_returns_none_instead_of_raising(self, monkeypatch):
        # Regression case: a proxy/CDN error page returning HTML with a 200
        # status previously crashed the whole research pipeline here.
        monkeypatch.setattr(
            tools.requests, "post", lambda *a, **k: FakeResponse(200, json_raises=True)
        )
        assert tools._scrape_with_firecrawl("https://x.com", "key", 10) is None

    def test_unexpected_json_shape_returns_none(self, monkeypatch):
        monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse(200, ["not", "a", "dict"]))
        assert tools._scrape_with_firecrawl("https://x.com", "key", 10) is None

    def test_data_not_a_dict_returns_none(self, monkeypatch):
        monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse(200, {"data": "oops"}))
        assert tools._scrape_with_firecrawl("https://x.com", "key", 10) is None

    def test_empty_markdown_returns_none(self, monkeypatch):
        monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse(200, {"data": {"markdown": ""}}))
        assert tools._scrape_with_firecrawl("https://x.com", "key", 10) is None


class TestScrapeWithBs4:
    def test_extracts_paragraph_text(self, monkeypatch):
        html = "<html><body><p>First.</p><p>Second.</p></body></html>"
        monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse(200, text=html))
        result = tools._scrape_with_bs4("https://x.com", retries=1, timeout=5)
        assert "First." in result and "Second." in result

    def test_no_paragraphs_reports_no_content(self, monkeypatch):
        monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse(200, text="<html></html>"))
        result = tools._scrape_with_bs4("https://x.com", retries=1, timeout=5)
        assert result == "No content found on this page."

    def test_non_200_status_reported_without_retrying(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            tools.requests, "get",
            lambda *a, **k: calls.append(1) or FakeResponse(404),
        )
        result = tools._scrape_with_bs4("https://x.com", retries=2, timeout=5)
        assert "Status code 404" in result
        assert len(calls) == 1  # a clean 404 is not retried, only exceptions are

    def test_retries_on_exception_then_succeeds(self, monkeypatch):
        attempts = {"n": 0}

        def flaky_get(*a, **k):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise requests.exceptions.Timeout("slow")
            return FakeResponse(200, text="<p>Recovered</p>")

        monkeypatch.setattr(tools.requests, "get", flaky_get)
        result = tools._scrape_with_bs4("https://x.com", retries=2, timeout=5)
        assert "Recovered" in result
        assert attempts["n"] == 2

    def test_exhausts_retries_and_reports_error(self, monkeypatch):
        attempts = {"n": 0}

        def always_fails(*a, **k):
            attempts["n"] += 1
            raise requests.exceptions.ConnectionError("dead site")

        monkeypatch.setattr(tools.requests, "get", always_fails)
        result = tools._scrape_with_bs4("https://x.com", retries=1, timeout=5)
        assert "Error during scraping" in result
        assert attempts["n"] == 2  # 1 retry == 2 total attempts


class TestScrapeWebsite:
    def test_uses_firecrawl_when_key_present_and_succeeds(self, monkeypatch):
        monkeypatch.setitem(tools.st.secrets, "FIRECRAWL_API_KEY", "fc-key")
        monkeypatch.setattr(tools, "_scrape_with_firecrawl", lambda *a, **k: "firecrawl content")
        bs4_calls = []
        monkeypatch.setattr(tools, "_scrape_with_bs4", lambda *a, **k: bs4_calls.append(1))

        result = tools.scrape_website("https://x.com")
        assert result == "firecrawl content"
        assert bs4_calls == []  # bs4 fallback never invoked

    def test_falls_back_to_bs4_when_firecrawl_key_missing(self, monkeypatch):
        monkeypatch.setitem(tools.st.secrets, "FIRECRAWL_API_KEY", "")
        firecrawl_calls = []
        monkeypatch.setattr(
            tools, "_scrape_with_firecrawl", lambda *a, **k: firecrawl_calls.append(1)
        )
        monkeypatch.setattr(tools, "_scrape_with_bs4", lambda *a, **k: "bs4 content")

        result = tools.scrape_website("https://x.com")
        assert result == "bs4 content"
        assert firecrawl_calls == []

    def test_falls_back_to_bs4_when_firecrawl_keeps_failing(self, monkeypatch):
        monkeypatch.setitem(tools.st.secrets, "FIRECRAWL_API_KEY", "fc-key")
        monkeypatch.setattr(tools, "_scrape_with_firecrawl", lambda *a, **k: None)
        monkeypatch.setattr(tools, "_scrape_with_bs4", lambda *a, **k: "bs4 content")

        result = tools.scrape_website("https://x.com", retries=2)
        assert result == "bs4 content"
