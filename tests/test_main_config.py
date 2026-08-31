"""Characterization tests for main.py's agent wiring and Groq rate-limit
tracking.

The model name, iteration budget, and memory window have all changed more
than once already (see PROGRESS.md / INTERVIEW_MASTERY.html history: the
model has been swapped after a Groq deprecation, and max_iterations has
moved 5 -> 15 -> 8 across separate tuning passes). These tests pin the
*current* known-good values so a future edit that changes one of them shows
up as a failing test instead of silent drift between the code and the
docs/interview prep material describing it.
"""
import httpx

import main


class TestAgentConfig:
    def test_groq_model_pin(self):
        assert main.GROQ_MODEL == "qwen/qwen3.6-27b"

    def test_max_iterations_budget(self):
        assert main.agent_executor.max_iterations == 8

    def test_memory_window_keeps_last_three_exchanges(self):
        assert main.memory.k == 6  # 3 user + 3 assistant turns
        assert main.memory.memory_key == "chat_history"

    def test_parsing_errors_are_handled_not_raised(self):
        assert main.agent_executor.handle_parsing_errors == "Please provide a Final Answer."

    def test_all_four_tools_are_registered_with_expected_names(self):
        names = {t.name for t in main.tools}
        assert names == {
            "get_stock_info",
            "perform_web_search",
            "process_research",
            "query_uploaded_document",
        }

    def test_prompt_still_contains_the_core_safety_rules(self):
        # Cheap guard against an edit to the ReAct template accidentally
        # dropping one of the rules that keep tool-call spend bounded.
        assert "at most 2 tool calls" in main.template
        assert "NEVER repeat a tool call" in main.template
        assert "Never invent tool names" in main.template


class TestRateLimitTracking:
    def _response(self, headers):
        return httpx.Response(200, headers=headers, request=httpx.Request("GET", "https://x.com"))

    def test_starts_empty(self):
        assert main.get_rate_limits() == {}

    def test_captures_only_ratelimit_headers_case_insensitively(self):
        main._capture_rate_limits(self._response({
            "X-RateLimit-Remaining-Requests": "42",
            "X-RateLimit-Limit-Requests": "100",
            "Content-Type": "application/json",
        }))
        limits = main.get_rate_limits()
        assert limits["x-ratelimit-remaining-requests"] == "42"
        assert limits["x-ratelimit-limit-requests"] == "100"
        assert "content-type" not in limits

    def test_later_calls_update_rather_than_replace_the_stored_values(self):
        main._capture_rate_limits(self._response({"X-RateLimit-Remaining-Requests": "42"}))
        main._capture_rate_limits(self._response({"X-RateLimit-Remaining-Tokens": "9000"}))
        limits = main.get_rate_limits()
        assert limits["x-ratelimit-remaining-requests"] == "42"
        assert limits["x-ratelimit-remaining-tokens"] == "9000"

    def test_same_header_seen_twice_takes_the_latest_value(self):
        main._capture_rate_limits(self._response({"X-RateLimit-Remaining-Requests": "42"}))
        main._capture_rate_limits(self._response({"X-RateLimit-Remaining-Requests": "41"}))
        assert main.get_rate_limits()["x-ratelimit-remaining-requests"] == "41"

    def test_get_rate_limits_returns_a_copy_not_the_live_dict(self):
        main._capture_rate_limits(self._response({"X-RateLimit-Remaining-Requests": "42"}))
        limits = main.get_rate_limits()
        limits["x-ratelimit-remaining-requests"] = "tampered"
        assert main.get_rate_limits()["x-ratelimit-remaining-requests"] == "42"
