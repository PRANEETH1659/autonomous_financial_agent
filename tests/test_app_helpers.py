"""Pure-logic helpers from app.py: ticker sniffing, the zero-LLM bare-ticker
fast path, markdown-$ escaping, chart-intent detection, and Groq token/quota
tracking. app.py runs its Streamlit script body at import time too (sidebar,
chat input, etc.) - in a bare pytest process those calls are harmless no-ops
(Streamlit logs "missing ScriptRunContext" and moves on), which is what
makes importing it directly here safe.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

import app


# ---------------------------------------------------------------------------
# find_ticker_in_text
# ---------------------------------------------------------------------------

class TestFindTickerInText:
    def test_returns_none_when_no_candidates_present(self, monkeypatch):
        calls = []
        monkeypatch.setattr(app, "get_stock_history", lambda *a, **k: calls.append(1))
        assert app.find_ticker_in_text("just some lowercase words") is None
        assert calls == []

    def test_skips_known_stopwords_without_checking_them(self, monkeypatch):
        calls = []
        monkeypatch.setattr(app, "get_stock_history", lambda t, **k: calls.append(t))
        result = app.find_ticker_in_text("The CEO discussed AI and EPS growth with the SEC.")
        assert result is None
        assert calls == []  # CEO, AI, EPS, SEC are all stopwords

    def test_returns_first_candidate_with_real_history(self, monkeypatch):
        def fake_history(ticker, period="1mo"):
            return pd.DataFrame({"Close": [1, 2]}) if ticker == "TSLA" else None

        monkeypatch.setattr(app, "get_stock_history", fake_history)
        result = app.find_ticker_in_text("Compare NOTREAL and TSLA performance.")
        assert result == "TSLA"

    def test_deduplicates_repeated_candidates(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            app, "get_stock_history",
            lambda t, **k: calls.append(t) or None,
        )
        app.find_ticker_in_text("NVDA NVDA NVDA NVDA NVDA NVDA NVDA")
        assert calls == ["NVDA"]  # checked once, not 7 times

    def test_stops_after_max_candidates_checked(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            app, "get_stock_history",
            lambda t, **k: calls.append(t) or None,
        )
        text = "AAAA BBBB CCCC DDDD EEEE FFFF GGGG"  # 7 distinct candidates
        app.find_ticker_in_text(text)
        assert len(calls) <= app.MAX_TICKER_CANDIDATES_CHECKED


# ---------------------------------------------------------------------------
# try_bare_ticker_fast_path
# ---------------------------------------------------------------------------

class TestTryBareTickerFastPath:
    def test_bare_ticker_returns_summary_and_ticker(self, monkeypatch):
        monkeypatch.setattr(app, "get_stock_summary", lambda t: f"Company: {t} Inc.")
        result = app.try_bare_ticker_fast_path("TSLA")
        assert result == ("Company: TSLA Inc.", "TSLA")

    def test_trailing_question_mark_and_whitespace_are_stripped(self, monkeypatch):
        monkeypatch.setattr(app, "get_stock_summary", lambda t: f"Company: {t} Inc.")
        assert app.try_bare_ticker_fast_path("  nvda?  ") == ("Company: NVDA Inc.", "NVDA")

    def test_multi_word_message_is_not_a_fast_path(self, monkeypatch):
        calls = []
        monkeypatch.setattr(app, "get_stock_summary", lambda t: calls.append(t))
        assert app.try_bare_ticker_fast_path("What is TSLA worth") is None
        assert calls == []

    def test_stopword_shaped_like_a_ticker_is_rejected(self, monkeypatch):
        calls = []
        monkeypatch.setattr(app, "get_stock_summary", lambda t: calls.append(t))
        assert app.try_bare_ticker_fast_path("AI") is None
        assert calls == []

    def test_word_too_long_for_a_ticker_is_rejected(self, monkeypatch):
        calls = []
        monkeypatch.setattr(app, "get_stock_summary", lambda t: calls.append(t))
        assert app.try_bare_ticker_fast_path("TOOLONG") is None
        assert calls == []

    def test_share_class_suffix_is_accepted(self, monkeypatch):
        monkeypatch.setattr(app, "get_stock_summary", lambda t: f"Company: {t}")
        assert app.try_bare_ticker_fast_path("brk.b") == ("Company: BRK.B", "BRK.B")

    def test_invalid_ticker_summary_falls_back_to_none(self, monkeypatch):
        monkeypatch.setattr(app, "get_stock_summary", lambda t: "Could not find valid stock data for 'ZZZZ'.")
        assert app.try_bare_ticker_fast_path("ZZZZ") is None

    def test_error_summary_falls_back_to_none(self, monkeypatch):
        monkeypatch.setattr(app, "get_stock_summary", lambda t: "Error fetching stock info for X: timeout")
        assert app.try_bare_ticker_fast_path("X") is None


# ---------------------------------------------------------------------------
# escape_dollars / wants_chart
# ---------------------------------------------------------------------------

class TestEscapeDollars:
    def test_escapes_every_dollar_sign(self):
        assert app.escape_dollars("Price rose from $11.62 to $47.06 billion") == (
            r"Price rose from \$11.62 to \$47.06 billion"
        )

    def test_text_without_dollars_is_unchanged(self):
        assert app.escape_dollars("No prices here") == "No prices here"

    def test_consecutive_dollars_each_escaped(self):
        assert app.escape_dollars("$$5") == r"\$\$5"


class TestWantsChart:
    @pytest.mark.parametrize("text", [
        "show me a chart of TSLA",
        "what's the trend for NVDA",
        "price action last month",
        "SHOW ME THE HISTORY",
        "can you visualize this",
        "can you visualise this",
        "plot AAPL over time",
    ])
    def test_detects_chart_intent(self, text):
        assert app.wants_chart(text) is True

    @pytest.mark.parametrize("text", [
        "what is the price of AAPL",
        "compare TSLA and NVDA valuations",
        "tell me about Apple",
    ])
    def test_no_chart_intent(self, text):
        assert app.wants_chart(text) is False


# ---------------------------------------------------------------------------
# render_chart_block guard clauses (no-network, no-data early returns)
# ---------------------------------------------------------------------------

class TestRenderChartBlockGuardClauses:
    def test_returns_early_when_no_history_available(self, monkeypatch):
        monkeypatch.setattr(app, "get_stock_history", lambda *a, **k: None)
        # Must not raise even though nothing downstream (st.caption etc.) runs.
        assert app.render_chart_block("ZZZZ") is None

    def test_returns_early_when_history_is_all_nan_close(self, monkeypatch):
        hist = pd.DataFrame({"Close": [float("nan"), float("nan")], "Volume": [1, 2]})
        monkeypatch.setattr(app, "get_stock_history", lambda *a, **k: hist)
        assert app.render_chart_block("AAPL") is None


# ---------------------------------------------------------------------------
# _pretty_reset
# ---------------------------------------------------------------------------

class TestPrettyReset:
    @pytest.mark.parametrize("raw, expected", [
        ("", "unknown"),
        (None, "unknown"),
        ("90ms", "moments"),
        ("52.8ms", "moments"),
        ("2m52.8s", "2m52.8s"),
        ("1h2m3s", "1h2m3s"),
    ])
    def test_pretty_reset(self, raw, expected):
        assert app._pretty_reset(raw) == expected


# ---------------------------------------------------------------------------
# TokenUsageTracker
# ---------------------------------------------------------------------------

def _streamed_response(input_tokens, output_tokens):
    message = SimpleNamespace(usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens})
    gen = SimpleNamespace(message=message)
    return SimpleNamespace(generations=[[gen]], llm_output=None)


def _non_streamed_response(prompt_tokens, completion_tokens):
    gen = SimpleNamespace(message=SimpleNamespace(usage_metadata=None))
    return SimpleNamespace(
        generations=[[gen]],
        llm_output={"token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
    )


class TestTokenUsageTrackerExtractUsage:
    def test_reads_usage_metadata_from_streamed_response(self):
        assert app.TokenUsageTracker._extract_usage(_streamed_response(10, 20)) == (10, 20)

    def test_falls_back_to_llm_output_token_usage(self):
        assert app.TokenUsageTracker._extract_usage(_non_streamed_response(5, 7)) == (5, 7)

    def test_neither_shape_present_returns_zeros(self):
        gen = SimpleNamespace(message=SimpleNamespace(usage_metadata=None))
        response = SimpleNamespace(generations=[[gen]], llm_output=None)
        assert app.TokenUsageTracker._extract_usage(response) == (0, 0)


class TestTokenUsageTrackerOnLlmEnd:
    def test_accumulates_stats_across_multiple_calls(self):
        stats = {
            "calls": 0, "calls_this_query": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "tokens_this_query": 0,
        }
        tracker = app.TokenUsageTracker(stats)

        tracker.on_llm_end(_streamed_response(10, 20))
        tracker.on_llm_end(_non_streamed_response(5, 7))

        assert stats["calls"] == 2
        assert stats["calls_this_query"] == 2
        assert stats["prompt_tokens"] == 15
        assert stats["completion_tokens"] == 27
        assert stats["total_tokens"] == 42
        assert stats["tokens_this_query"] == 42
