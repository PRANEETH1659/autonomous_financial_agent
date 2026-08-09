import re
import logging
import yfinance as yf
import pandas as pd
import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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


@tool
def get_stock_info(ticker: str) -> str:
    """Fetches current price, valuation metrics, and recent performance for a given ticker symbol.
    Returns a string summary of the stock's performance."""
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

    if not hist.empty:
        last_day_data = hist.iloc[-1]
        last_open = last_day_data['Open']
        last_close = last_day_data['Close']
        last_high = last_day_data['High']
        last_low = last_day_data['Low']

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


@tool
def perform_web_search(query: str) -> str:
    """Performs a web search using Serper.dev and returns the top results."""
    try:
        serper_api_key = st.secrets.get("SERPER_API_KEY")

        if not serper_api_key:
            return "Error: SERPER_API_KEY not found in Streamlit secrets."

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
        return f"Error performing web search for '{query}': network issue, please try again."
    except Exception as e:
        logger.exception("Unexpected error performing web search for '%s'", query)
        return f"Error performing web search for '{query}': {e}"


def scrape_website(url: str, retries: int = 1, timeout: int = 10) -> str:
    """Scrapes text content from a webpage, retrying once on transient network errors."""
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
            logger.warning("Scrape attempt %d/%d failed for %s: %s", attempt, attempts, url, e)
            if attempt == attempts:
                return f"Error during scraping: {e}"
        except Exception as e:
            logger.exception("Unexpected error scraping %s", url)
            return f"Error during scraping: {e}"


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


def store_in_vector_db(chunks: list, collection_name: str = "financial_research"):
    """
    Takes text chunks and builds a TF-IDF matrix for similarity search.
    Replaces the old ChromaDB + HuggingFace embeddings implementation.
    """
    try:
        if not chunks:
            return None

        vectorizer = TfidfVectorizer(stop_words='english')
        matrix = vectorizer.fit_transform(chunks)

        _chunk_store[collection_name] = {
            "chunks": chunks,
            "vectorizer": vectorizer,
            "matrix": matrix,
        }
        return _chunk_store[collection_name]

    except Exception:
        logger.exception("Error storing in Vector DB")
        return None


def retrieve_context(query: str, k: int = 2, collection_name: str = "financial_research"):
    """
    Takes a query, computes cosine similarity against stored TF-IDF vectors,
    and returns the top-k most relevant chunks as a formatted string.
    """
    try:
        store = _chunk_store.get(collection_name)
        if not store:
            return "No data stored yet."

        query_vec = store["vectorizer"].transform([query])
        scores = cosine_similarity(query_vec, store["matrix"]).flatten()

        k = min(k, len(store["chunks"]))
        top_idx = np.argsort(scores)[-k:][::-1]

        context = "\n---\n".join([store["chunks"][i] for i in top_idx])
        return context

    except Exception as e:
        return f"Error retrieving context: {e}"


@tool
def process_research(query: str):
    """
    The full Phase 2 Pipeline: Search -> Scrape -> Chunk -> Store -> Retrieve
    """

    # 1. Search
    logger.info("[STEP 1] Searching web for: %s", query)
    search_results = perform_web_search.invoke(query)
    logger.info("[OK] Search completed")

    # 2. Scrape
    links = re.findall(r'Link:\s*(https?://\S+)', search_results)

    if not links:
        return "No links found to research."

    raw_text = None
    for link in links:
        logger.info("[STEP 2] Scraping: %s", link)
        scraped = scrape_website(link)
        if not scraped.startswith("Failed to retrieve") and not scraped.startswith("Error"):
            raw_text = scraped
            break

    if not raw_text:
        return "All Links are getting blocked. Try New URL..."

    logger.info("[OK] Scraping successful, %d characters extracted", len(raw_text))

    # 3. Chunk
    chunks = chunk_text(raw_text)
    logger.info("[OK] Created %d chunks", len(chunks))

    # 4. Store (TF-IDF)
    store_in_vector_db(chunks)
    logger.info("[OK] Stored successfully")

    # 5. Retrieve
    context = retrieve_context(query)
    logger.info("[OK] Context retrieved")

    return context[:500]


if __name__ == "__main__":
    pass