"""get_stock_info and its single-ticker helper. All yfinance access goes
through tools._fetch_yf_data, so every test here mocks that one seam
instead of touching the network."""
import pandas as pd
import pytest

import tools


def _hist(rows):
    """Builds a minimal OHLCV history DataFrame from a list of
    (open, close, high, low, volume) tuples."""
    return pd.DataFrame(
        rows,
        columns=["Open", "Close", "High", "Low", "Volume"],
    )


FULL_INFO = {
    "longName": "Apple Inc.",
    "currentPrice": 190.5,
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 3_000_000_000_000,
    "trailingPE": 30.1,
    "forwardPE": 28.4,
    "trailingEps": 6.2,
    "dividendYield": 0.55,
    "beta": 1.2,
    "fiftyTwoWeekLow": 150.0,
    "fiftyTwoWeekHigh": 200.0,
    "averageVolume": 50_000_000,
}


class TestGetStockInfoForOne:
    def test_invalid_ticker_format_short_circuits_before_any_fetch(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            tools, "_fetch_yf_data", lambda *a, **k: calls.append(1) or (FULL_INFO, _hist([]))
        )
        result = tools._get_stock_info_for_one("not a ticker!!")
        assert "Could not find valid stock data" in result
        assert calls == []  # never even tried to hit yfinance

    def test_fetch_exception_is_reported_not_raised(self, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("yfinance is down")

        monkeypatch.setattr(tools, "_fetch_yf_data", boom)
        result = tools._get_stock_info_for_one("AAPL")
        assert "Error fetching stock info for AAPL" in result
        assert "yfinance is down" in result

    def test_missing_name_and_price_reported_as_not_found(self, monkeypatch):
        # A delisted/private ticker: yfinance returns a near-empty info dict
        # rather than raising.
        monkeypatch.setattr(tools, "_fetch_yf_data", lambda *a, **k: ({}, _hist([])))
        result = tools._get_stock_info_for_one("ZZZZ")
        assert "Could not find valid stock data for 'ZZZZ'" in result

    def test_full_summary_includes_key_fields_and_period_change(self, monkeypatch):
        hist = _hist([
            (100, 105, 106, 99, 1_000_000),
            (105, 120, 121, 104, 1_200_000),  # 20% up over the period
        ])
        monkeypatch.setattr(tools, "_fetch_yf_data", lambda *a, **k: (FULL_INFO, hist))

        result = tools._get_stock_info_for_one("aapl")  # lowercase input

        assert "Apple Inc. (AAPL)" in result
        assert "Technology" in result
        assert "3,000,000,000,000" in result
        assert "0.55%" in result  # dividend yield formatted as a percentage
        # period_change is computed from Close[0] -> Close[-1] (105 -> 120),
        # not Open[0] -> Close[-1] - deliberately picked non-round numbers
        # here so a test that used the wrong baseline would be caught.
        assert "6-month change: 14.29%" in result

    def test_missing_dividend_yield_renders_as_na(self, monkeypatch):
        info = dict(FULL_INFO)
        del info["dividendYield"]
        monkeypatch.setattr(tools, "_fetch_yf_data", lambda *a, **k: (info, _hist([])))
        result = tools._get_stock_info_for_one("AAPL")
        assert "Dividend Yield: N/A" in result

    def test_empty_history_reports_no_historical_data(self, monkeypatch):
        monkeypatch.setattr(tools, "_fetch_yf_data", lambda *a, **k: (FULL_INFO, _hist([])))
        result = tools._get_stock_info_for_one("AAPL")
        assert "No historical price data available." in result
        assert "Open: N/A" in result

    def test_falls_back_to_regular_market_price_when_current_price_missing(self, monkeypatch):
        info = dict(FULL_INFO)
        del info["currentPrice"]
        info["regularMarketPrice"] = 188.0
        monkeypatch.setattr(tools, "_fetch_yf_data", lambda *a, **k: (info, _hist([])))
        result = tools._get_stock_info_for_one("AAPL")
        assert "Current Price: 188.0" in result


class TestGetStockInfoTool:
    """Exercises the @tool-wrapped, comma-separated-batch entry point via
    .func, bypassing LangChain's schema layer to test the pure logic."""

    def test_single_ticker(self, monkeypatch):
        monkeypatch.setattr(tools, "_fetch_yf_data", lambda *a, **k: (FULL_INFO, _hist([])))
        result = tools.get_stock_info.func("AAPL")
        assert "Apple Inc." in result
        assert "---" not in result  # only one summary, no separator

    def test_batches_multiple_tickers_into_one_call_per_unique_ticker(self, monkeypatch):
        seen = []

        def fake_fetch(ticker, period="6mo"):
            seen.append(ticker)
            info = dict(FULL_INFO, longName=f"{ticker} Inc.")
            return info, _hist([])

        monkeypatch.setattr(tools, "_fetch_yf_data", fake_fetch)
        result = tools.get_stock_info.func("TSLA, NVDA, AAPL")

        assert seen == ["TSLA", "NVDA", "AAPL"]
        assert result.count("---") == 2  # 3 summaries joined by 2 separators
        assert "TSLA Inc." in result and "NVDA Inc." in result and "AAPL Inc." in result

    def test_deduplicates_repeated_tickers_preserving_first_occurrence(self, monkeypatch):
        seen = []

        def fake_fetch(ticker, period="6mo"):
            seen.append(ticker)
            return dict(FULL_INFO), _hist([])

        monkeypatch.setattr(tools, "_fetch_yf_data", fake_fetch)
        tools.get_stock_info.func("AAPL, aapl, AAPL, TSLA")
        assert seen == ["AAPL", "TSLA"]  # case-insensitive dedup, order kept

    def test_empty_or_whitespace_input_reports_invalid_format(self, monkeypatch):
        called = []
        monkeypatch.setattr(tools, "_fetch_yf_data", lambda *a, **k: called.append(1))
        result = tools.get_stock_info.func("   ,  ,")
        assert "Could not find valid stock data" in result
        assert called == []

    def test_mixed_valid_and_invalid_tickers_reports_both(self, monkeypatch):
        def fake_fetch(ticker, period="6mo"):
            return dict(FULL_INFO, longName=f"{ticker} Inc."), _hist([])

        monkeypatch.setattr(tools, "_fetch_yf_data", fake_fetch)
        result = tools.get_stock_info.func("AAPL, NOT@VALID")
        assert "AAPL Inc." in result
        assert "Could not find valid stock data" in result
