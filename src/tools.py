import yfinance as yf
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
            f"Market Cap: {market_cap:,}\n"
            f"Current Price: {current_price}\n"
            f"Last Trading Day (Open: {last_open}, Close: {last_close}, "
            f"High: {last_high}, Low: {last_low})\n"
            f"{seven_day_summary}"
        )

        return summary

    except Exception as e:
        return f"Error fetching stock info for {ticker}: {e}"


def perform_web_search(query: str) -> str:
    """Performs a web search using Serper.dev and returns the top results."""
    try:
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            return "Error: SERPER_API_KEY not found in environment variables."

        search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
        results = search.results(query)

        formatted_results = []
        if 'organic' in results:
            for i, res in enumerate(results['organic'][:5]):  # Top 5 results
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
        chunk_size=1000,    # reasonable size
        chunk_overlap=100,  # overlap for context
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.create_documents([text])
    return [chunk.page_content for chunk in chunks]


if __name__ == "__main__":
    load_dotenv()

    print("\n--- Testing Financial Data Tool ---")
    apple_info = get_stock_info("AAPL")
    print(apple_info)

    print("\n--- Testing Web Search Tool ---")
    news_query = "latest news on Apple Inc."
    apple_news = perform_web_search(news_query)
    print(apple_news)

    print("\n--- Testing Invalid Search ---")
    invalid_search = perform_web_search("asdfghjklqwertyuiopzxcvbnm_nonexistent_query")
    print(invalid_search)

    print("\n--- Testing Web Scraper ---")
    scraped_text = scrape_website("https://en.wikipedia.org/wiki/Artificial_intelligence")
    print(scraped_text[:500], "...")  # print first 500 chars

    print("\n--- Testing Text Chunking ---")
    chunks = chunk_text(scraped_text)
    print(f"Generated {len(chunks)} chunks. First chunk:\n{chunks[0]}")
