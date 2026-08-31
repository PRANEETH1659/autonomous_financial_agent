"""process_research: the full search -> scrape -> chunk -> index -> retrieve
pipeline behind the "deep research" tool. Every network-touching step
(_search_web, scrape_website) is mocked; only the pure chunking/BM25 stages
run for real, so these tests also double as pipeline-wiring tests."""
import tools


LONG_ARTICLE = (
    "Nvidia continues to expand its AI data center investments. " * 60
)  # comfortably over 2000 chars once chunked back together


class TestProcessResearchTool:
    def test_search_failure_is_reported_and_pipeline_stops(self, monkeypatch):
        def raise_search_error(query):
            raise tools.SearchError("Error performing web search for 'x': network issue, please try again.")
        monkeypatch.setattr(tools, "_search_web", raise_search_error)

        result = tools.process_research.func("Nvidia AI investments")
        assert "network issue" in result

    def test_no_links_in_search_results_reports_that_clearly(self, monkeypatch):
        monkeypatch.setattr(tools, "_search_web", lambda q: "No search results found for 'x'.")
        result = tools.process_research.func("something obscure")
        assert result == "No links found to research."

    def test_all_links_blocked_reports_that_clearly(self, monkeypatch):
        monkeypatch.setattr(
            tools, "_search_web",
            lambda q: "Result 1:\nTitle: A\nLink: https://a.com\nSnippet: s\n",
        )
        monkeypatch.setattr(tools, "scrape_website", lambda url, **k: "Failed to retrieve the webpage. Status code 403")
        result = tools.process_research.func("blocked topic")
        assert result == "All Links are getting blocked. Try New URL..."

    def test_skips_failed_links_and_uses_first_successful_one(self, monkeypatch):
        monkeypatch.setattr(
            tools, "_search_web",
            lambda q: (
                "Result 1:\nTitle: A\nLink: https://bad.com\nSnippet: s\n"
                "Result 2:\nTitle: B\nLink: https://good.com\nSnippet: s\n"
            ),
        )
        scraped_urls = []

        def fake_scrape(url, **k):
            scraped_urls.append(url)
            if url == "https://bad.com":
                return "Error during scraping: timeout"
            return "Good article content about the requested topic. " * 5

        monkeypatch.setattr(tools, "scrape_website", fake_scrape)
        result = tools.process_research.func("some topic")

        assert scraped_urls == ["https://bad.com", "https://good.com"]  # stops at first success
        assert "requested topic" in result

    def test_result_is_capped_at_2000_characters(self, monkeypatch):
        monkeypatch.setattr(
            tools, "_search_web",
            lambda q: "Result 1:\nTitle: A\nLink: https://good.com\nSnippet: s\n",
        )
        monkeypatch.setattr(tools, "scrape_website", lambda url, **k: LONG_ARTICLE)

        result = tools.process_research.func("Nvidia AI investments")
        assert len(result) <= 2000

    def test_progress_callback_receives_pipeline_updates(self, monkeypatch):
        monkeypatch.setattr(
            tools, "_search_web",
            lambda q: "Result 1:\nTitle: A\nLink: https://good.com\nSnippet: s\n",
        )
        monkeypatch.setattr(tools, "scrape_website", lambda url, **k: "Some article content here. " * 5)

        messages = []
        tools.set_progress_callback(messages.append)
        try:
            tools.process_research.func("topic")
        finally:
            tools.set_progress_callback(None)

        joined = " ".join(messages)
        assert "Searching the web" in joined
        assert "Scraping" in joined
        assert "Indexed chunks" in joined
