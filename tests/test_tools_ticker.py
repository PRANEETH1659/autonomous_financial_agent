"""Ticker validation and number formatting - the small pure functions that
every other stock-data test implicitly relies on."""
import pytest

import tools


class TestValidateTicker:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("AAPL", "AAPL"),
            ("aapl", "AAPL"),           # lowercase gets normalized
            ("  tsla  ", "TSLA"),        # surrounding whitespace stripped
            ("BRK.B", "BRK.B"),          # dot allowed (share classes)
            ("brk-b", "BRK-B"),          # hyphen allowed
            ("A123456789", "A123456789"),  # exactly 10 chars: still valid
        ],
    )
    def test_valid_tickers_normalize_to_uppercase(self, raw, expected):
        assert tools.validate_ticker(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "   ",
            123,                      # not a string at all
            "1AAPL",                  # can't start with a digit
            "AA PL",                  # no internal whitespace
            "AAPL$",                  # no special characters beyond . and -
            "A1234567890",            # 11 chars - one over the limit
            "AAPL!",
        ],
    )
    def test_invalid_tickers_return_none(self, raw):
        assert tools.validate_ticker(raw) is None


class TestFormatNumber:
    def test_formats_int_with_thousands_separator(self):
        assert tools._format_number(1234567) == "1,234,567"

    def test_formats_float_with_two_decimals(self):
        assert tools._format_number(12.3456) == "12.35"

    def test_formats_negative_float(self):
        assert tools._format_number(-8.1) == "-8.10"

    def test_appends_suffix(self):
        assert tools._format_number(5, suffix="%") == "5%"
        assert tools._format_number(5.5, suffix="%") == "5.50%"

    @pytest.mark.parametrize("value", [None, "N/A", "not a number", [], {}])
    def test_non_numeric_input_returns_na(self, value):
        assert tools._format_number(value) == "N/A"

    def test_zero_is_still_formatted_not_treated_as_missing(self):
        # 0 is falsy in Python but a legitimate value (e.g. 0% dividend
        # yield) - must not be collapsed into "N/A".
        assert tools._format_number(0) == "0"
        assert tools._format_number(0.0) == "0.00"
