import logging
import re

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

from main import agent_executor
from tools import (
    get_stock_history,
    validate_ticker,
    set_progress_callback,
    parse_pdf,
    chunk_text,
    store_in_vector_db,
    clear_vector_db,
    get_stock_summary,
)

logger = logging.getLogger(__name__)


class ToolSelectionReporter(BaseCallbackHandler):
    """Writes which tool the agent picks into the live status panel, so the
    research process reads as an agent reasoning, not a fixed script."""

    def __init__(self, status):
        self.status = status

    def on_agent_action(self, action, **kwargs):
        self.status.write(f"🤖 Using tool: **{action.tool}** — `{action.tool_input}`")

st.set_page_config(
    page_title="AI Financial Research Agent",
    page_icon="📈",
    layout="wide",
)

TICKER_CANDIDATE_PATTERN = re.compile(r"\b[A-Z]{2,6}(?:\.[A-Z]{1,3})?\b")
MAX_TICKER_CANDIDATES_CHECKED = 5

# Uppercase words common in financial writing that are not the company being
# discussed. Skipping them avoids a wasted yfinance round-trip per word (some,
# like AI and PE, are real tickers and would otherwise chart the wrong thing).
TICKER_STOPWORDS = frozenset({
    "AI", "PE", "EPS", "IPO", "GDP", "ETF", "CEO", "CFO", "USD", "EUR",
    "NYSE", "NASDAQ", "SEC", "FED", "API", "USA", "UK", "EV", "ESG",
    "YOY", "QOQ", "TTM", "ROI", "ROE", "EBITDA", "PDF", "AND", "THE",
    "FOR", "WITH", "FROM", "THIS", "THAT", "KEY", "OVERALL",
})


def find_ticker_in_text(text: str) -> str | None:
    """Best-effort scan of free text for a mentionable, valid ticker symbol."""
    seen = set()
    checked = 0
    for candidate in TICKER_CANDIDATE_PATTERN.findall(text):
        if candidate in TICKER_STOPWORDS or candidate in seen:
            continue
        seen.add(candidate)
        checked += 1
        if checked > MAX_TICKER_CANDIDATES_CHECKED:
            break
        normalized = validate_ticker(candidate)
        if normalized:
            hist = get_stock_history(normalized, period="1mo")
            if hist is not None:
                return normalized
    return None


# A message that is nothing but a ticker (e.g. "TSLA" or "tsla?") is answered
# straight from yfinance with zero LLM calls - this is the single most common
# demo query, and routing it through the ReAct loop costs 2-3 Groq requests
# for data the agent would only read back verbatim anyway.
BARE_TICKER_PATTERN = re.compile(r"^[A-Za-z]{1,6}(?:\.[A-Za-z]{1,3})?$")


def try_bare_ticker_fast_path(text: str) -> tuple[str, str] | None:
    """Returns (answer, ticker) if the whole message is just a valid ticker,
    otherwise None so the normal agent path runs."""
    candidate = text.strip().rstrip("?").strip()
    if not BARE_TICKER_PATTERN.match(candidate):
        return None
    normalized = validate_ticker(candidate)
    if not normalized or normalized in TICKER_STOPWORDS:
        return None
    summary = get_stock_summary(normalized)
    if summary.startswith("Could not find") or summary.startswith("Error"):
        return None
    return summary, normalized


def escape_dollars(text: str) -> str:
    """Streamlit renders markdown, where a `$...$` pair is LaTeX math. Stock
    answers are full of dollar amounts ("$11.62 ... $47.06 billion"), so an
    even number of them silently swallows the text in between into a math
    block. Escaping every $ keeps prices rendering as prices."""
    return text.replace("$", r"\$")


# Charts are supplementary, not part of every answer. Rendering one every turn
# also costs extra yfinance calls - find_ticker_in_text probes up to 5 candidate
# tickers at a different cache key than the agent's own lookup - and those extra
# calls are what get this app rate-limited on shared cloud IPs.
CHART_REQUEST_PATTERN = re.compile(
    r"(chart|charts|graph|plot|trend|trending|history|historical|"
    r"performance|price action|movement|visuali[sz]e|over time)",
    re.IGNORECASE,
)


def wants_chart(text: str) -> bool:
    """True when the user actually asked to see a chart."""
    return bool(CHART_REQUEST_PATTERN.search(text))


def render_chart_block(ticker: str, show_metrics: bool = True) -> None:
    """Draws the price chart, and optionally the metric row, for a ticker.

    Rows with a NaN Close are dropped first: yfinance returns a partial row for
    the in-progress trading session with Volume already filled in but OHLC still
    empty, which otherwise renders as "$nan" and "+nan%" while Day Volume looks
    fine. Dropping them also takes the trailing gap out of the chart line."""
    hist = get_stock_history(ticker, period="6mo")
    if hist is None:
        return

    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        return

    st.caption(f"📊 Chart & metrics: {ticker}")

    if show_metrics:
        latest = hist.iloc[-1]
        first_close = hist["Close"].iloc[0]
        change_pct = (latest["Close"] - first_close) / first_close * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Last Close", f"${latest['Close']:.2f}")
        col2.metric("6-Month Change", f"{change_pct:+.2f}%")
        col3.metric("Day Volume", f"{int(latest['Volume']):,}")

    st.line_chart(hist["Close"])


with st.sidebar:
    st.header("📈 Financial Research Agent")
    st.markdown(
        "Ask about stock prices, valuation metrics, company news, "
        "or comparisons between companies."
    )
    st.markdown(
        "**Available tools:**\n- Live stock data (yfinance)\n- Web search\n"
        "- Deep research pipeline\n- Uploaded document Q&A"
    )

    always_show_charts = st.toggle(
        "Always show charts",
        value=False,
        help="Off by default. A chart is still drawn automatically whenever "
             "you ask for one, e.g. 'show me Tesla's performance'.",
    )

    st.divider()
    st.subheader("📄 Upload a report")
    uploaded_file = st.file_uploader(
        "Text-based PDF (10-K, earnings report, etc.)",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state.get("uploaded_doc_name"):
        with st.spinner(f"Reading {uploaded_file.name}..."):
            text = parse_pdf(uploaded_file)
            if not text:
                st.error(
                    f"Couldn't extract text from {uploaded_file.name} — "
                    "it may be a scanned/image-only PDF, which isn't supported yet."
                )
            else:
                chunks = chunk_text(text)
                # store_in_vector_db returns None when nothing indexable came
                # out (e.g. a page of only punctuation). Without this check the
                # UI would claim success and every later question would answer
                # "No data stored yet."
                stored = store_in_vector_db(chunks, collection_name="uploaded_document")
                if stored:
                    st.session_state.uploaded_doc_name = uploaded_file.name
                    st.success(
                        f"Indexed {len(stored['chunks'])} chunks from "
                        f"{uploaded_file.name}. Ask me about it!"
                    )
                else:
                    st.session_state.uploaded_doc_name = None
                    st.error(
                        f"{uploaded_file.name} had no searchable text to index."
                    )
    elif st.session_state.get("uploaded_doc_name"):
        st.caption(f"📎 Active: {st.session_state.uploaded_doc_name}")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploaded_doc_name = None
        clear_vector_db("uploaded_document")
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
        st.markdown(escape_dollars(message["content"]))
        # Only set when a chart was actually asked for that turn, so replaying
        # history doesn't resurrect charts the user never wanted.
        if message.get("ticker"):
            render_chart_block(message["ticker"], show_metrics=False)

if prompt := st.chat_input("What would you like to research?"):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        fast_path = try_bare_ticker_fast_path(prompt)

        if fast_path:
            answer, fast_ticker = fast_path
            st.caption("⚡ Answered directly from live market data (no LLM call)")
            # The agent never saw this exchange, so record it manually -
            # otherwise a follow-up like "is it overvalued?" would have no
            # antecedent for "it" in chat history.
            agent_executor.memory.save_context({"input": prompt}, {"output": answer})
        else:
            fast_ticker = None
            with st.status("Researching...", expanded=True) as status:
                set_progress_callback(status.write)
                reporter = ToolSelectionReporter(status)

                try:
                    response = agent_executor.invoke(
                        {"input": prompt},
                        config={"callbacks": [reporter]},
                    )

                    answer = response.get(
                        "output",
                        "No response generated."
                    )
                    status.update(label="Research complete", state="complete")

                except Exception as e:
                    logger.exception("Agent execution failed for prompt: %s", prompt)
                    answer = (
                        "Something went wrong while researching that. "
                        f"Details: {e}"
                    )
                    status.update(label="Research failed", state="error")

                finally:
                    set_progress_callback(None)

        st.markdown(escape_dollars(answer))

        # A chart is supplementary detail, not part of every answer - draw one
        # only when the user asked, or when they've pinned the sidebar toggle
        # on. Skipping this also skips find_ticker_in_text's yfinance probes.
        ticker = None
        if always_show_charts or wants_chart(prompt):
            # Prefer a ticker actually named in the answer (what the agent's
            # conclusion is about) over one merely mentioned in the user's
            # prompt - otherwise a 3-way comparison would silently chart
            # whichever company was typed first, regardless of the answer.
            ticker = fast_ticker or find_ticker_in_text(answer) or find_ticker_in_text(prompt)
            if ticker:
                render_chart_block(ticker)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "ticker": ticker}
    )
