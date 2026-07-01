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


@tool
def get_stock_info(ticker: str) -> str:
    """Fetches current and historical stock information for a given ticker symbol.
    Returns a string summary of the stock's performance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        current_price = info.get('currentPrice', 'N/A')
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        if isinstance(market_cap, (int, float)):
            market_cap_display = f"{market_cap:,}"
        else:
            market_cap_display = "N/A"

        hist = stock.history(period="7d")

        if not hist.empty:
            last_day_data = hist.iloc[-1]
            last_open = last_day_data['Open']
            last_close = last_day_data['Close']
            last_high = last_day_data['High']
            last_low = last_day_data['Low']

            seven_day_change = (
                (hist['Close'].iloc[-1] - hist['Close'].iloc[0])
                / hist['Close'].iloc[0] * 100
            )
            seven_day_summary = f"7-day change: {seven_day_change:.2f}%"
        else:
            last_open, last_close, last_high, last_low = 'N/A', 'N/A', 'N/A', 'N/A'
            seven_day_summary = "No historical data for the last 7 days."

        summary = (
            f"Company: {company_name} ({ticker})\n"
            f"Sector: {sector}, Industry: {industry}\n"
            f"Market Cap: {market_cap_display}\n"
            f"Current Price: {current_price}\n"
            f"Last Trading Day (Open: {last_open}, Close: {last_close}, "
            f"High: {last_high}, Low: {last_low})\n"
            f"{seven_day_summary}"
        )

        return summary

    except Exception as e:
        return f"Error fetching stock info for {ticker}: {e}"


@tool
def perform_web_search(query: str) -> str:
    """Performs a web search using Serper.dev and returns the top results."""
    try:
        serper_api_key = st.secrets["SERPER_API_KEY"]

        if not serper_api_key:
            return "Error: SERPER_API_KEY not found in environment variables."

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

    except Exception as e:
        return f"Error performing web search for '{query}': {e}"


def scrape_website(url: str) -> str:
    """Scrapes text content from a webpage."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return f"Failed to retrieve the webpage. Status code {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        page_text = "\n".join([p.get_text() for p in paragraphs])

        return page_text if page_text else "No content found on this page."

    except Exception as e:
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


# ----------------------------------------------------------------------
# TF-IDF based "vector store" replacement (no chromadb, no torch,
# no sentence-transformers). Stored in memory per Streamlit session.
# ----------------------------------------------------------------------

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

    except Exception as e:
        print(f"Error storing in Vector DB: {e}")
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
    print(f"\n[STEP 1] Searching Web for : {query}")
    search_results = perform_web_search.invoke(query)
    print("[OK] Search Completed")

    # 2. Scrape
    import re
    links = re.findall(r'Link:\s*(https?://\S+)', search_results)

    if not links:
        return "No links found to research."

    raw_text = None
    for link in links:
        print(f"\n[STEP 2] Scraping: {link}")
        scraped = scrape_website(link)
        if not scraped.startswith("Failed to retrieve") and not scraped.startswith("Error"):
            raw_text = scraped
            break

    if not raw_text:
        return "All Links are getting blocked. Try New URL..."

    print("[OK] Scraping Successful")
    print(f"Characters Extracted: {len(raw_text)}")

    # 3. Chunk
    print("\n[STEP 3] Chunking Text...")
    chunks = chunk_text(raw_text)
    print(f"[OK] Created {len(chunks)} chunks")

    # 4. Store (TF-IDF)
    print("\n[STEP 4] Building TF-IDF Index...")
    store_in_vector_db(chunks)
    print("[OK] Stored Successfully")

    # 5. Retrieve
    print("\n[STEP 5] Retrieving Relevant Context...")
    context = retrieve_context(query)
    print("[OK] Context Retrieved")

    return context[:500]


if __name__ == "__main__":
    pass