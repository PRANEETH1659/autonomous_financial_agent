import yfinance as yf
import pandas as pd
import os
from langchain_community.utilities import GoogleSerperAPIWrapper
from datetime import datetime,timedelta

def get_stock_info(ticker:str)->str:
    
    """ Fetches current and historical stock information for a given ticker symbol.
    Returns a string summary of the stock's performance. """
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info 
        
        current_price = info.get('currentPrice','N/A')
        company_name = info.get('longName',ticker)
        sector=info.get('sector','N/A')
        industry = info.get('industry','N/A')
        market_cap = info.get('marketCap','N/A')
        
        hist = stock.history(period="7d")
        
        if not hist.empty:
            last_day_data = hist.iloc[-1]
            last_open = last_day_data['Open']
            last_close=last_day_data['Close']
            last_high = last_day_data['High']
            last_low=last_day_data['Low']
            
            seven_day_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
            seven_day_summary = f"7-day change: {seven_day_change:.2f}%"
        else:
            last_open, last_close, last_high, last_low = 'N/A', 'N/A', 'N/A', 'N/A'
            seven_day_summary = "No historical data for the last 7 days."
        
        summary = (
            f"Company: {company_name} ({ticker})\n"
            f"Sector: {sector}, Industry: {industry}\n"
            f"Market Cap: {market_cap:,}\n"
            f"Current Price: {current_price}\n"
            f"Last Trading Day (Open: {last_open}, Close: {last_close}, High: {last_high}, Low: {last_low})\n"
            f"{seven_day_summary}"
        )
        
        return summary

    except Exception as e:
        return f"Error fetching stock info for {ticker}: {e}"


def perform_web_search(query: str) -> str:
    """
    Performs a web search using Serper.dev and returns the top results.
    """
    try:
        # Ensure SERPER_API_KEY is loaded from .env
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            return "Error: SERPER_API_KEY not found in environment variables."

        search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
        results = search.results(query)

        # Format the results for better readability
        formatted_results = []
        if 'organic' in results:
            for i, res in enumerate(results['organic'][:5]): # Get top 5 organic results
                title = res.get('title', 'No Title')
                link = res.get('link', 'No Link')
                snippet = res.get('snippet', 'No Snippet')
                formatted_results.append(f"Result {i+1}:\nTitle: {title}\nLink: {link}\nSnippet: {snippet}\n")
        
        if not formatted_results:
            return f"No search results found for '{query}'."

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error performing web search for '{query}': {e}"
# Example usage (for quick testing)
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("\n--- Testing Financial Data Tool ---")
    apple_info = get_stock_info("AAPL")
    print(apple_info)

    print("\n--- Testing Web Search Tool ---")
    news_query = "latest news on Apple Inc."
    apple_news = perform_web_search(news_query)
    print(apple_news)

    invalid_search = perform_web_search("asdfghjklqwertyuiopzxcvbnm_nonexistent_query")
    print(invalid_search)