"""perform_web_search and its cached Serper-backed helper. GoogleSerperAPIWrapper
is mocked at the class level so no test ever makes a real HTTP call."""
from unittest.mock import MagicMock

import pytest
import requests

import tools


def _patch_serper(monkeypatch, results=None, side_effect=None):
    wrapper = MagicMock()
    if side_effect is not None:
        wrapper.results.side_effect = side_effect
    else:
        wrapper.results.return_value = results or {}
    monkeypatch.setattr(
        tools, "GoogleSerperAPIWrapper", MagicMock(return_value=wrapper)
    )
    return wrapper


class TestSearchWeb:
    def test_missing_api_key_raises_search_error(self, monkeypatch, st_secrets_without):
        st_secrets_without("SERPER_API_KEY")
        with pytest.raises(tools.SearchError, match="SERPER_API_KEY not found"):
            tools._search_web("Tesla news")

    def test_formats_top_three_organic_results(self, monkeypatch):
        _patch_serper(monkeypatch, results={
            "organic": [
                {"title": f"Title {i}", "link": f"https://example.com/{i}", "snippet": f"Snippet {i}"}
                for i in range(5)
            ]
        })
        result = tools._search_web("unique query one")
        assert "Result 1:" in result and "Result 3:" in result
        assert "Result 4:" not in result  # only top 3, even though 5 came back
        assert "Title 0" in result and "https://example.com/0" in result

    def test_no_organic_results_returns_friendly_message(self, monkeypatch):
        _patch_serper(monkeypatch, results={})
        result = tools._search_web("unique query two")
        assert result == "No search results found for 'unique query two'."

    def test_missing_fields_in_a_result_fall_back_to_placeholders(self, monkeypatch):
        _patch_serper(monkeypatch, results={"organic": [{}]})
        result = tools._search_web("unique query three")
        assert "No Title" in result and "No Link" in result and "No Snippet" in result

    def test_network_error_raises_search_error_with_friendly_message(self, monkeypatch):
        _patch_serper(monkeypatch, side_effect=requests.exceptions.ConnectionError("dns fail"))
        with pytest.raises(tools.SearchError, match="network issue"):
            tools._search_web("unique query four")

    def test_unexpected_exception_is_wrapped_as_search_error(self, monkeypatch):
        _patch_serper(monkeypatch, side_effect=ValueError("weird serper payload"))
        with pytest.raises(tools.SearchError, match="weird serper payload"):
            tools._search_web("unique query five")


class TestPerformWebSearchTool:
    def test_returns_formatted_results_on_success(self, monkeypatch):
        _patch_serper(monkeypatch, results={
            "organic": [{"title": "T", "link": "https://x.com", "snippet": "S"}]
        })
        result = tools.perform_web_search.func("unique query six")
        assert "Result 1:" in result

    def test_search_error_is_returned_as_string_not_raised(self, monkeypatch):
        _patch_serper(monkeypatch, side_effect=requests.exceptions.Timeout("slow"))
        # perform_web_search must never raise - the agent loop only knows
        # how to handle tool observations, not exceptions bubbling out of a
        # tool call.
        result = tools.perform_web_search.func("unique query seven")
        assert isinstance(result, str)
        assert "network issue" in result
