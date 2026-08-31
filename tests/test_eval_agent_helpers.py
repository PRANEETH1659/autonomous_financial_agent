"""Unit tests for eval_agent.py's own assertion helpers. These are the
building blocks every check in TEST_CASES relies on (tool_was_called,
count_calls, tool_called_with_all) - if one of them is subtly wrong, the
whole manual eval suite could report false passes without anyone noticing,
since eval_agent.py itself is only ever run manually against live APIs.
"""
import eval_agent as ea


SAMPLE_CALLS = [
    ("get_stock_info", "TSLA, NVDA, AAPL"),
    ("perform_web_search", "Tesla news today"),
    ("get_stock_info", "AAPL"),
]


class TestToolWasCalled:
    def test_true_when_present(self):
        assert ea.tool_was_called(SAMPLE_CALLS, "get_stock_info") is True

    def test_false_when_absent(self):
        assert ea.tool_was_called(SAMPLE_CALLS, "process_research") is False

    def test_false_on_empty_calls(self):
        assert ea.tool_was_called([], "get_stock_info") is False


class TestCountCalls:
    def test_counts_multiple_matches(self):
        assert ea.count_calls(SAMPLE_CALLS, "get_stock_info") == 2

    def test_zero_for_no_matches(self):
        assert ea.count_calls(SAMPLE_CALLS, "query_uploaded_document") == 0


class TestToolCalledWithAll:
    def test_true_when_all_terms_present_case_insensitively(self):
        assert ea.tool_called_with_all(SAMPLE_CALLS, "get_stock_info", ["tsla", "NVDA", "aapl"]) is True

    def test_false_when_one_term_missing(self):
        assert ea.tool_called_with_all(SAMPLE_CALLS, "get_stock_info", ["TSLA", "MSFT"]) is False

    def test_false_when_tool_name_never_called(self):
        assert ea.tool_called_with_all(SAMPLE_CALLS, "process_research", ["TSLA"]) is False

    def test_checks_across_separate_calls_independently_not_merged(self):
        # The second get_stock_info call ("AAPL" alone) does NOT contain
        # "TSLA" - tool_called_with_all must not pass by combining terms
        # found across two different calls.
        assert ea.tool_called_with_all(SAMPLE_CALLS, "get_stock_info", ["AAPL", "TSLA"]) is True
        assert ea.tool_called_with_all(
            [("get_stock_info", "AAPL"), ("get_stock_info", "TSLA")], "get_stock_info", ["AAPL", "TSLA"]
        ) is False


class TestConstants:
    def test_llm_call_budget_matches_documented_rationale(self):
        # 1 tool call costs 2 LLM calls (choose tool, write answer); a
        # two-tool question plus one retry is the documented ceiling.
        assert ea.MAX_LLM_CALLS_PER_QUERY == 4
