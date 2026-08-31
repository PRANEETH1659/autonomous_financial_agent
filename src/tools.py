import re
import logging
import yfinance as yf
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

from rank_bm25 import BM25Okapi
from pypdf import PdfReader
import numpy as np

logger = logging.getLogger(__name__)
# Streamlit installs its own root-logger handlers before this module is
# imported, which makes a plain basicConfig() a silent no-op (it only acts
# when the root logger has no handlers) - that's why app-level INFO logs were
# invisible in production. Configuring this logger directly sidesteps the root
# logger entirely, so these lines appear regardless of Streamlit's setup.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Lets the UI show live pipeline progress. Assumes a single active research
# session at a time (fine for this app's usage) - not safe for concurrent users.
_progress_callback = None


def set_progress_callback(callback):
    """Registers a callback(str) invoked at each process_research step. Pass
    None to clear it once a research call finishes."""
    global _progress_callback
    _progress_callback = callback


def _report(message: str) -> None:
    logger.info(message)
    if _progress_callback:
        _progress_callback(message)


TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def validate_ticker(ticker: str) -> str | None:
    """Normalizes and validates a ticker symbol string. Returns the
    normalized (uppercase) ticker, or None if the format is invalid."""
    if not ticker or not isinstance(ticker, str):
        return None
    normalized = ticker.strip().upper()
    return normalized if TICKER_PATTERN.match(normalized) else None


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_yf_data(ticker: str, period: str = "6mo"):
    """Fetches and caches yfinance info + history for a ticker (5 min TTL)
    so a single ticker mention doesn't hit the API twice in one turn."""
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period=period)
    return info, hist


def get_stock_history(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    """Returns a closing-price history DataFrame for charting, or None if
    the ticker is invalid or has no data."""
    normalized = validate_ticker(ticker)
    if not normalized:
        return None
    try:
        _, hist = _fetch_yf_data(normalized, period)
        return hist if not hist.empty else None
    except Exception as e:
        logger.warning("Failed to fetch history for %s: %s", normalized, e)
        return None


def _format_number(value, suffix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.2f}{suffix}" if isinstance(value, float) else f"{value:,}{suffix}"
    return "N/A"


def _get_stock_info_for_one(ticker: str) -> str:
    normalized = validate_ticker(ticker)
    if not normalized:
        return f"Could not find valid stock data: '{ticker}' is not a valid ticker symbol format."

    try:
        info, hist = _fetch_yf_data(normalized)
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", normalized, e)
        return f"Error fetching stock info for {normalized}: {e}"

    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
    company_name = info.get('longName')

    if not company_name and current_price is None:
        return f"Could not find valid stock data for '{normalized}'. It may be an invalid ticker or a private/delisted company."

    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    market_cap_display = _format_number(info.get('marketCap'))
    trailing_pe = _format_number(info.get('trailingPE'))
    forward_pe = _format_number(info.get('forwardPE'))
    eps = _format_number(info.get('trailingEps'))
    dividend_yield = info.get('dividendYield')
    dividend_display = f"{dividend_yield:.2f}%" if isinstance(dividend_yield, (int, float)) else "N/A"
    beta = _format_number(info.get('beta'))
    week_52_low = _format_number(info.get('fiftyTwoWeekLow'))
    week_52_high = _format_number(info.get('fiftyTwoWeekHigh'))
    avg_volume = _format_number(info.get('averageVolume'))

    # During market hours yfinance appends a partial row for the in-progress
    # session: Volume is populated but OHLC is still NaN. `hist` is therefore
    # not empty, so the guard below passes and iloc[-1] reads those NaNs -
    # surfacing as "Open: nan ... 6-month change: nan%" both to the user and,
    # worse, to the LLM as tool output it then reasons from. app.py's
    # render_chart_block drops these rows for the chart; the text summary
    # needs the same treatment.
    hist = hist.dropna(subset=['Open', 'Close', 'High', 'Low'])

    if not hist.empty:
        last_day_data = hist.iloc[-1]
        last_open = f"{last_day_data['Open']:.2f}"
        last_close = f"{last_day_data['Close']:.2f}"
        last_high = f"{last_day_data['High']:.2f}"
        last_low = f"{last_day_data['Low']:.2f}"

        period_change = (
            (hist['Close'].iloc[-1] - hist['Close'].iloc[0])
            / hist['Close'].iloc[0] * 100
        )
        period_summary = f"6-month change: {period_change:.2f}%"
    else:
        last_open, last_close, last_high, last_low = 'N/A', 'N/A', 'N/A', 'N/A'
        period_summary = "No historical price data available."

    summary = (
        f"Company: {company_name or normalized} ({normalized})\n"
        f"Sector: {sector}, Industry: {industry}\n"
        f"Market Cap: {market_cap_display}\n"
        f"Current Price: {current_price if current_price is not None else 'N/A'}\n"
        f"Valuation: Trailing P/E {trailing_pe}, Forward P/E {forward_pe}, EPS {eps}, Beta {beta}\n"
        f"Dividend Yield: {dividend_display}\n"
        f"52-Week Range: {week_52_low} - {week_52_high}\n"
        f"Average Volume: {avg_volume}\n"
        f"Last Trading Day (Open: {last_open}, Close: {last_close}, "
        f"High: {last_high}, Low: {last_low})\n"
        f"{period_summary}"
    )

    return summary


def get_stock_summary(ticker: str) -> str:
    """Public single-ticker summary, used by the UI's zero-LLM fast path."""
    return _get_stock_info_for_one(ticker)


@tool
def get_stock_info(tickers: str) -> str:
    """Fetches current price, valuation metrics, and recent performance for
    one or more stock tickers. Pass a single ticker (e.g. 'AAPL') or, when
    comparing companies, ALL tickers comma-separated in one call
    (e.g. 'TSLA, NVDA, AAPL') instead of calling this tool once per company.
    Returns one summary per ticker."""
    # Dedupe while preserving order: the agent sometimes re-issues a ticker
    # twice in one call after a format retry, and each duplicate would
    # otherwise cost a redundant yfinance round-trip.
    seen = set()
    ticker_list = []
    for t in tickers.split(","):
        t = t.strip()
        if t and t.upper() not in seen:
            seen.add(t.upper())
            ticker_list.append(t)

    if not ticker_list:
        return f"Could not find valid stock data: '{tickers}' is not a valid ticker symbol format."

    return "\n---\n".join(_get_stock_info_for_one(t) for t in ticker_list)


class SearchError(Exception):
    """Raised when a web search fails. Raised rather than returned as a string
    so st.cache_data does not cache the failure - a cached error would keep a
    transient Serper outage 'sticky' for the full 10-minute TTL."""


@st.cache_data(ttl=600, show_spinner=False)
def _search_web(query: str) -> str:
    """Cached Serper lookup (10 min TTL). Shared by perform_web_search and
    process_research so an escalation from one to the other on the same
    query doesn't pay for a second identical API round-trip."""
    try:
        serper_api_key = st.secrets.get("SERPER_API_KEY")

        if not serper_api_key:
            raise SearchError("Error: SERPER_API_KEY not found in Streamlit secrets.")

        search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
        results = search.results(query)

        formatted_results = []
        if 'organic' in results:
            for i, res in enumerate(results['organic'][:3]):  # Top 3 results
                title = res.get('title', 'No Title')
                link = res.get('link', 'No Link')
                snippet = res.get('snippet', 'No Snippet')
                formatted_results.append(
                    f"Result {i+1}:\nTitle: {title}\nLink: {link}\nSnippet: {snippet}\n"
                )

        if not formatted_results:
            return f"No search results found for '{query}'."

        return "\n".join(formatted_results)

    except requests.exceptions.RequestException as e:
        logger.warning("Web search network error for '%s': %s", query, e)
        raise SearchError(
            f"Error performing web search for '{query}': network issue, please try again."
        ) from e
    except SearchError:
        raise
    except Exception as e:
        logger.exception("Unexpected error performing web search for '%s'", query)
        raise SearchError(f"Error performing web search for '{query}': {e}") from e


@tool
def perform_web_search(query: str) -> str:
    """Answers questions using a quick web search of headlines and snippets.
    Best for simple, factual, current-events questions (e.g. 'why did Tesla
    stock drop today'). If the user asks for deep research, analysis, or a
    detailed summary of a topic, use process_research instead - do NOT call
    both tools for the same question."""
    try:
        return _search_web(query)
    except SearchError as e:
        return str(e)


FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"


def _scrape_with_firecrawl(url: str, api_key: str, timeout: int) -> str | None:
    """Single Firecrawl scrape attempt. Returns markdown content, or None on
    any failure (network error, non-200, empty content) so the caller can
    retry or fall back."""
    try:
        response = requests.post(
            FIRECRAWL_SCRAPE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"url": url, "formats": ["markdown"], "timeout": timeout * 1000},
            timeout=timeout + 15,
        )
    except requests.exceptions.RequestException as e:
        logger.warning("Firecrawl request failed for %s: %s", url, e)
        return None

    if response.status_code != 200:
        logger.warning("Firecrawl returned HTTP %d for %s", response.status_code, url)
        return None

    # A 200 doesn't guarantee JSON - a proxy or CDN error page would make
    # .json() raise, which previously escaped this function and crashed the
    # whole research pipeline instead of falling back to the bs4 scraper.
    try:
        payload = response.json()
    except ValueError:
        logger.warning("Firecrawl returned non-JSON body for %s", url)
        return None

    if not isinstance(payload, dict):
        logger.warning("Firecrawl returned unexpected JSON shape for %s", url)
        return None

    data = payload.get("data")
    if not isinstance(data, dict):
        return None

    markdown = data.get("markdown")
    return markdown or None


def _scrape_with_bs4(url: str, retries: int, timeout: int) -> str:
    """Fallback scraper: raw HTML fetch + <p> tag extraction. Used when
    Firecrawl has no key configured, or fails after its retries."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0 Safari/537.36"
        )
    }

    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)

            if response.status_code != 200:
                return f"Failed to retrieve the webpage. Status code {response.status_code}"

            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            page_text = "\n".join([p.get_text() for p in paragraphs])

            return page_text if page_text else "No content found on this page."

        except requests.exceptions.RequestException as e:
            logger.warning("bs4 scrape attempt %d/%d failed for %s: %s", attempt, attempts, url, e)
            if attempt == attempts:
                return f"Error during scraping: {e}"
        except Exception as e:
            logger.exception("Unexpected error scraping %s", url)
            return f"Error during scraping: {e}"

    # Defensive: every path above returns, but an implicit None here would
    # crash process_research's `scraped.startswith(...)` check.
    return f"Error during scraping: {url} could not be retrieved."


def scrape_website(url: str, retries: int = 1, timeout: int = 10) -> str:
    """Scrapes clean page content via Firecrawl (handles JS-rendered pages),
    retrying once on failure, then falls back to a raw HTML/<p>-tag scrape
    if Firecrawl has no API key configured or keeps failing."""
    firecrawl_api_key = st.secrets.get("FIRECRAWL_API_KEY")

    if firecrawl_api_key:
        attempts = retries + 1
        for attempt in range(1, attempts + 1):
            markdown = _scrape_with_firecrawl(url, firecrawl_api_key, timeout)
            if markdown:
                return markdown
            logger.warning("Firecrawl scrape attempt %d/%d failed for %s", attempt, attempts, url)
        logger.warning("Firecrawl exhausted retries for %s, falling back to raw HTML scrape", url)
    else:
        logger.info("FIRECRAWL_API_KEY not set, using fallback scraper for %s", url)

    return _scrape_with_bs4(url, retries, timeout)


def chunk_text(text: str):
    """Splits long text into overlapping chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.create_documents([text])
    return [chunk.page_content for chunk in chunks]


_chunk_store = {}


def _tokenize(text: str) -> list[str]:
    """Lowercase word-tokenizer for BM25. BM25Okapi scores token lists, not
    raw strings, so both stored chunks and queries must go through this."""
    return re.findall(r"\w+", text.lower())


def store_in_vector_db(chunks: list, collection_name: str = "financial_research"):
    """
    Takes text chunks and builds a BM25 index for keyword search.
    Replaces the old ChromaDB + HuggingFace embeddings implementation.
    """
    try:
        if not chunks:
            return None

        # Keep only chunks that produce at least one token. BM25Okapi divides
        # by the corpus's average document length, so a corpus where every
        # chunk tokenizes to nothing (e.g. a PDF page of only punctuation)
        # raises ZeroDivisionError. Filtering keeps `chunks` and the BM25
        # index index-aligned, which retrieve_context relies on.
        tokenized = [(c, _tokenize(c)) for c in chunks]
        usable = [(c, toks) for c, toks in tokenized if toks]

        if not usable:
            logger.warning(
                "No searchable text in %d chunk(s) for collection '%s'",
                len(chunks), collection_name,
            )
            return None

        kept_chunks = [c for c, _ in usable]
        bm25 = BM25Okapi([toks for _, toks in usable])

        _chunk_store[collection_name] = {
            "chunks": kept_chunks,
            "bm25": bm25,
        }
        return _chunk_store[collection_name]

    except Exception:
        logger.exception("Error storing in Vector DB")
        return None


def retrieve_context(query: str, k: int = 2, collection_name: str = "financial_research"):
    """
    Takes a query, scores it against the stored BM25 index, and returns the
    top-k most relevant chunks as a formatted string.
    """
    try:
        store = _chunk_store.get(collection_name)
        if not store:
            return "No data stored yet."

        scores = store["bm25"].get_scores(_tokenize(query))

        k = min(k, len(store["chunks"]))
        top_idx = np.argsort(scores)[-k:][::-1]

        context = "\n---\n".join([store["chunks"][i] for i in top_idx])
        return context

    except Exception as e:
        return f"Error retrieving context: {e}"


def clear_vector_db(collection_name: str = "financial_research") -> None:
    """Removes a stored collection, if present. Used to reset the uploaded
    document's index when the user starts a new conversation."""
    _chunk_store.pop(collection_name, None)


def parse_pdf(file) -> str:
    """Extracts text from a PDF (a file-like object, e.g. from Streamlit's
    file_uploader). Returns the concatenated text of all pages, or an empty
    string if extraction fails or the PDF has no extractable text (e.g. it's
    a scanned image with no selectable text layer)."""
    try:
        reader = PdfReader(file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    except Exception:
        logger.exception("Failed to parse uploaded PDF")
        return ""


@tool
def query_uploaded_document(question: str) -> str:
    """Answers questions about a PDF the user has uploaded (e.g. an earnings
    report or 10-K). Only use this when the user's question refers to an
    uploaded document, report, or file. Returns the most relevant excerpt(s),
    or a message saying no document has been uploaded yet."""
    return retrieve_context(question, k=4, collection_name="uploaded_document")


@tool
def process_research(query: str):
    """Runs deep research on a topic: searches the web, scrapes the best
    source's full article text, and returns the most relevant passages.
    Use this - NOT perform_web_search - whenever the user asks to 'research',
    'deep dive', 'analyze', 'summarize', or wants detail beyond a headline.
    Returns enough context to answer in one call; do not call it twice for
    the same question."""

    # 1. Search
    _report(f"🔎 Searching the web for: {query}")
    try:
        search_results = _search_web(query)
    except SearchError as e:
        _report("⚠️ Search failed")
        return str(e)
    _report("✅ Search completed")

    # 2. Scrape
    links = re.findall(r'Link:\s*(https?://\S+)', search_results)

    if not links:
        return "No links found to research."

    raw_text = None
    for link in links:
        _report(f"🌐 Scraping: {link}")
        scraped = scrape_website(link)
        if not scraped.startswith("Failed to retrieve") and not scraped.startswith("Error"):
            raw_text = scraped
            break

    if not raw_text:
        return "All Links are getting blocked. Try New URL..."

    _report(f"✅ Scraped {len(raw_text)} characters")

    # 3. Chunk
    chunks = chunk_text(raw_text)
    _report(f"✂️ Created {len(chunks)} chunks")

    # 4. Store (BM25)
    store_in_vector_db(chunks)
    _report("📊 Indexed chunks for relevance ranking")

    # 5. Retrieve
    # k=4 (not 2) and a 2000-char cap (not 500): the old limits returned so
    # little context that the agent often had to make another tool call to
    # answer, costing more LLM calls than the larger observation does.
    context = retrieve_context(query, k=4)
    _report("✅ Context retrieved")

    return context[:2000]


if __name__ == "__main__":
    pass