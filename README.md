# Autonomous Financial Research Agent

**🔗 Live app: [https://autonomousfinancialagent-kveykyk53v4s7oflywacwm.streamlit.app](https://autonomousfinancialagent-kveykyk53v4s7oflywacwm.streamlit.app)**

A Streamlit chat app backed by a LangChain **ReAct agent**. Ask about stock prices, company news, comparisons, or upload a report and ask questions about it. The agent picks the right tool, shows its steps live, and writes the answer.

## What it can do

- **Live stock data**: price, market cap, P/E, EPS, dividends and 6-month performance via `yfinance`. A bare ticker like `TSLA` is answered instantly with no LLM call.
- **Web search**: quick current-events answers via Serper (Google).
- **Deep research**: search, scrape (Firecrawl, with a BeautifulSoup fallback), chunk, then **BM25** retrieval for questions that need synthesis across sources.
- **Document Q&A**: upload a PDF (earnings report, 10-K) and ask questions about it.
- **Multi-company comparisons** in a single batched data call.
- **Live usage panel**: tokens used and remaining Groq quota, read from response headers with no extra API calls.

## Tech stack

Python, Streamlit, LangChain (ReAct agent), Groq (Qwen model), yfinance, Serper, Firecrawl, BM25 (`rank_bm25`).

## Run it locally

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m streamlit run src\app.py
```

Create `src/.streamlit/secrets.toml` with your own keys (this file is git-ignored, never commit it):

```toml
GROQ_API_KEY = "..."
SERPER_API_KEY = "..."
FIRECRAWL_API_KEY = "..."   # optional, falls back to BeautifulSoup
```

When deploying on Streamlit Cloud, put the same keys in the app's **Settings → Secrets**.

## Tests

```powershell
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
venv\Scripts\python.exe -m pytest -q
```

The unit tests are mocked and offline. `src/eval_agent.py` is a separate live eval that checks which tools the agent actually uses. It calls real APIs and uses quota, so run it manually.

## Reliability notes

- Groq retires models without notice. At startup the app checks that the configured model still exists and, if not, switches to the newest Qwen model Groq offers.
- Model-unavailable and rate-limit errors are shown as plain-language messages.
- The Groq free-tier quota is shared by everyone using the app.

## Project layout

```
src/app.py          Streamlit UI
src/main.py         Agent, prompt, model selection
src/tools.py        Stock, search, scrape and retrieval tools
src/eval_agent.py   Live tool-selection eval
tests/              Offline unit tests
PROGRESS.md         Development log
```

> Not financial advice. Answers come from an LLM and public data and can be wrong.
