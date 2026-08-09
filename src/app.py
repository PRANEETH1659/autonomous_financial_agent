import logging
import re

import streamlit as st

from main import agent_executor
from tools import get_stock_history, validate_ticker

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Financial Research Agent",
    page_icon="📈",
    layout="wide",
)

TICKER_CANDIDATE_PATTERN = re.compile(r"\b[A-Z]{2,6}(?:\.[A-Z]{1,3})?\b")
MAX_TICKER_CANDIDATES_CHECKED = 5


def find_ticker_in_text(text: str) -> str | None:
    """Best-effort scan of free text for a mentionable, valid ticker symbol."""
    candidates = TICKER_CANDIDATE_PATTERN.findall(text)[:MAX_TICKER_CANDIDATES_CHECKED]
    for candidate in candidates:
        normalized = validate_ticker(candidate)
        if normalized:
            hist = get_stock_history(normalized, period="1mo")
            if hist is not None:
                return normalized
    return None


with st.sidebar:
    st.header("📈 Financial Research Agent")
    st.markdown(
        "Ask about stock prices, valuation metrics, company news, "
        "or comparisons between companies."
    )
    st.markdown("**Available tools:**\n- Live stock data (yfinance)\n- Web search\n- Deep research pipeline")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        agent_executor.memory.clear()
        st.rerun()

st.title("📈 Autonomous Financial Research Agent")
st.markdown(
    "Ask me about stock prices, company news, company comparisons, or financial research."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        ticker = message.get("ticker")
        if ticker:
            hist = get_stock_history(ticker, period="6mo")
            if hist is not None:
                st.line_chart(hist["Close"])

if prompt := st.chat_input("What would you like to research?"):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):

            try:
                response = agent_executor.invoke(
                    {"input": prompt}
                )

                answer = response.get(
                    "output",
                    "No response generated."
                )

            except Exception as e:
                logger.exception("Agent execution failed for prompt: %s", prompt)
                answer = (
                    "Something went wrong while researching that. "
                    f"Details: {e}"
                )

            st.markdown(answer)

            ticker = find_ticker_in_text(prompt)
            if ticker:
                hist = get_stock_history(ticker, period="6mo")
                if hist is not None:
                    latest = hist.iloc[-1]
                    first_close = hist["Close"].iloc[0]
                    change_pct = (latest["Close"] - first_close) / first_close * 100

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Last Close", f"${latest['Close']:.2f}")
                    col2.metric("6-Month Change", f"{change_pct:+.2f}%")
                    col3.metric("Day Volume", f"{int(latest['Volume']):,}")

                    st.line_chart(hist["Close"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "ticker": ticker}
    )
