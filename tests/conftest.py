"""
Shared test setup for the whole suite.

Two things have to happen *before* anything under src/ is imported, at
module import time rather than inside a fixture:

1. st.secrets must behave like a plain dict. main.py and tools.py both read
   secrets at import/call time, and tests should never depend on (or risk
   touching) whatever is actually in .streamlit/secrets.toml on this
   machine.
2. langchain_groq.ChatGroq must not construct a real client. main.py builds
   `llm = ChatGroq(...)` at import time - the real constructor doesn't hit
   the network, but pinning it to a MagicMock means the test suite never
   depends on a real (or even well-formed) Groq API key existing, and never
   risks an accidental live call if agent internals change later.

Both patches are applied here, at conftest.py's own module level, because
`import main` only runs main.py's top-level code once per process. If the
patch instead lived inside a fixture, whichever test file gets collected
first would import main.py unpatched.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402


class FakeSecrets(dict):
    """Drop-in for st.secrets. The app only ever does .get(key) or
    secrets[key] on this object, both of which a plain dict already
    supports identically to the real Secrets class."""


_BASE_SECRETS = {
    "GROQ_API_KEY": "test-groq-api-key",
    "SERPER_API_KEY": "test-serper-api-key",
    # Empty by default so scrape_website tests exercise the bs4 fallback
    # unless a test opts into a Firecrawl key explicitly.
    "FIRECRAWL_API_KEY": "",
}

st.secrets = FakeSecrets(_BASE_SECRETS)

import langchain_groq  # noqa: E402
langchain_groq.ChatGroq = MagicMock(name="ChatGroqClass")


@pytest.fixture(autouse=True)
def _fresh_secrets(monkeypatch):
    """Gives every test its own copy of the base secrets dict, so a test
    that deletes/overwrites a key (e.g. to simulate a missing API key)
    can never leak that change into the next test."""
    monkeypatch.setattr(st, "secrets", FakeSecrets(_BASE_SECRETS), raising=False)


@pytest.fixture(autouse=True)
def _isolate_module_state():
    """Several modules keep process-global mutable state that would
    otherwise leak between tests:

    - tools._chunk_store: the BM25 index dict, keyed by collection name.
    - main._RATE_LIMITS: the last-seen Groq rate-limit headers.

    Both are cleared before *and* after every test so test order never
    matters and a failed test can't poison the ones after it.
    """
    import tools
    import main

    def _clear_cache(fn):
        # A test that monkeypatches tools._fetch_yf_data / _search_web
        # directly (most of them do, to avoid real network calls) replaces
        # the st.cache_data-wrapped function with a plain callable for the
        # duration of that test. monkeypatch's own teardown runs *after*
        # this fixture's, so at either point the attribute may or may not
        # still be the cache-wrapped original - only clear() it if present.
        if hasattr(fn, "clear"):
            fn.clear()

    tools._chunk_store.clear()
    main._RATE_LIMITS.clear()
    _clear_cache(tools._fetch_yf_data)
    _clear_cache(tools._search_web)
    yield
    tools._chunk_store.clear()
    main._RATE_LIMITS.clear()
    _clear_cache(tools._fetch_yf_data)
    _clear_cache(tools._search_web)

@pytest.fixture
def st_secrets_without():
    """Factory fixture: st_secrets_without("SERPER_API_KEY") removes that
    key from st.secrets for the duration of the current test only (the
    _fresh_secrets fixture above already guarantees a per-test copy)."""
    import streamlit as st

    def _remove(key):
        st.secrets.pop(key, None)

    return _remove
