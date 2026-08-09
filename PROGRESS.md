# Autonomous Financial Research Agent — Progress Tracker

Last updated: 2026-08-08

A living record of what this project is, what's been built, what's currently broken or in flux, and what's next. Sections with `<details>` are collapsible — click to expand.

---

## 1. What this project is

A Streamlit chat app backed by a LangChain **ReAct agent** that can:
- Look up live stock data and valuation metrics (`yfinance`)
- Search the web (Serper.dev / Google)
- Run a lightweight research pipeline: search → scrape → chunk → TF-IDF retrieve, for questions that need synthesis across sources

The LLM is the decision-maker: it reads the user's question, picks a tool (or two), reads the tool's result back, and writes a final answer — the defining trait of an *agent* versus a plain chatbot wrapper.

---

## 2. Current status

<details>
<summary><strong>✅ Working and verified live (2026-08-08)</strong></summary>

- [x] Chat UI (Streamlit) — stock charts, comparison metrics, sidebar
- [x] `get_stock_info` tool (yfinance) — price, valuation, 52-week range
- [x] `perform_web_search` tool (Serper.dev)
- [x] `process_research` pipeline — search → scrape → chunk → TF-IDF → retrieve
- [x] Multi-turn memory (resolves "it", "compare it with Y", etc.)
- [x] Comparison-table formatting for explicit compare requests
- [x] LLM provider: **Groq**, `llama-3.3-70b-versatile` — confirmed working end-to-end via live browser test
- [x] Dependency set fully pinned and conflict-free (`pip check` clean)
- [x] `.gitignore` fixed (was UTF-16-encoded and silently non-functional)
- [x] Repo cleaned up — dead artifacts removed, reference PDFs moved to `docs/`

</details>

<details>
<summary><strong>🚧 Known limitations / not yet done</strong></summary>

- [ ] Chat memory is RAM-only and **shared across all users** of the deployed app (not per-session) — real bug, not yet hit because testing has been solo
- [ ] Web scraper (`scrape_website`) gets blocked by anti-bot pages on some sites — has a fallback-to-next-link workaround, not a real fix
- [ ] No persistent/searchable chat history yet (see backlog)
- [ ] No live visualization of the research pipeline's progress while it runs
- [ ] Streamlit is the interim host, not the final deployment target

</details>

---

## 3. Problems hit today, and how they were resolved

<details>
<summary><strong>🔴 "Agent stopped due to iteration limit" (the original complaint)</strong></summary>

**Symptom:** big/multi-part questions would fail outright with a generic stopped message.

**Root cause, found via verbose tracing:** the agent's `max_iterations` was capped at 5, and the ReAct prompt format requires the LLM to literally type the phrase `Final Answer:` before LangChain accepts the response as complete. Models sometimes write the correct answer but skip that exact phrase on the first try, burning iterations on retries.

**Fix:** raised `max_iterations` to 15 and added `early_stopping_method="generate"` (so even a worst-case timeout still produces a best-effort answer instead of a bare error). Verified by replaying the exact original failing question (Nvidia performance + investors) — now completes cleanly in 3 iterations.

</details>

<details>
<summary><strong>🟡 requirements.txt / .gitignore silently broken (UTF-16 encoding)</strong></summary>

**Symptom:** deploy-time dependency failures; `.gitignore` rules not actually excluding files from `git status`.

**Root cause:** both files were saved in UTF-16 at some point, which pip and git don't parse correctly, but don't loudly error on either — they just silently fail to work as intended.

**Fix:** rewrote both as plain UTF-8/ASCII. Also discovered the dev `venv/` had drifted to 176 installed packages against only 12 declared ones (leftovers from an earlier ChromaDB/HuggingFace-embeddings architecture) — rebuilt the venv from scratch against a fully-pinned `requirements.txt` (full `pip freeze` output, not just direct dependencies) to guarantee dev/deploy parity.

</details>

<details>
<summary><strong>🔵 The Groq → Gemini → Groq detour</strong></summary>

**Why it happened:** wanted to try Gemini as a "modern" alternative to Groq's Llama 3.3 70B.

**What went wrong:**
1. `langchain-google-genai`'s latest version required `langchain-core>=1.0`, breaking the rest of the pinned stack — fixed by pinning `langchain-google-genai==2.1.10` + `langchain-core==0.3.86`.
2. Gemini's free tier turned out to be a hard **20 requests/day** cap — far too tight for an agent that costs 2-4 LLM calls per question. Tried other Gemini models hoping for separate quota buckets; one 404'd (deprecated for new users), one had a `limit: 0` (no free access at all on this project). Lesson: don't guess model names, check the live model list for the actual account.
3. Researched alternatives properly (live web search, since API pricing changes fast): **Groq's free tier is 1,000 requests/day** for this exact model — 50x more headroom than Gemini, and the only genuinely-free option that comfortably clears a "100-150 requests/day" target.

**Decision:** reverted to Groq.

</details>

<details>
<summary><strong>🟠 Groq's native tool-calling is broken (the twist)</strong></summary>

While reverting to Groq, rewrote the agent to use `create_tool_calling_agent` (LangChain's structured tool-calling, which had been the fix for Gemini's format-compliance problem). On Groq, this reproducibly failed:

```
groq.APIError: Failed to call a function. Please adjust your prompt.
failed_generation: '<function=get_stock_info{"ticker": "TSLA"}</function>'
```

Confirmed deterministic (not a streaming artifact — same failure with `disable_streaming="tool_calling"` set). **The fix that helped Gemini actively breaks Groq.** Tested the original legacy ReAct text-parsing format directly against Groq instead — worked perfectly, zero parsing failures, immediate correct `Final Answer:` compliance.

**Lesson learned, logged for future-me:** the right agent architecture (`create_react_agent` vs `create_tool_calling_agent`) is provider-specific, not a universal best practice. Re-verify empirically if the LLM provider ever changes again — don't assume the last answer transfers.

</details>

<details>
<summary><strong>🟢 Zombie server processes causing flaky test results</strong></summary>

After many restarts during the Gemini/Groq back-and-forth, two separate Streamlit processes ended up both bound to port 8501 simultaneously (Windows doesn't always cleanly kill child processes when a wrapper script is stopped). The browser was randomly hitting whichever stale process answered first — one still running old Gemini-era code. Killed both, confirmed exactly one clean process bound to the port, re-verified.

</details>

---

## 4. Development timeline

| Phase | What happened |
|---|---|
| Phase 1 | Project foundation — yfinance tool, Serper.dev search tool, project structure |
| Phase 2 | RAG pipeline v1 — web scraper, recursive chunker, HuggingFace embeddings + local ChromaDB |
| Phase 3 | Rebuilt on Python 3.11, added conversation memory, launched the Streamlit UI |
| Post-Phase 3 | Dependency cleanup, swapped ChromaDB+embeddings → TF-IDF (lighter, no heavy ML deps for free-tier hosting), added comparison-table formatting |
| 2026-08-08 | Full dependency reliability pass, repo cleanup, iteration-limit fix, Groq↔Gemini evaluation, agent architecture fix, this document |

---

## 5. Backlog — what's next

<details>
<summary><strong>Phase 1 (deployment reliability) — mostly done as of today</strong></summary>

- [x] Pin exact dependency versions
- [x] Verify clean-venv install matches deployment
- [x] Fix `.gitignore` encoding
- [ ] Final check: deploy to Streamlit Cloud (or equivalent) and confirm the pinned `requirements.txt` installs cleanly there too

</details>

<details>
<summary><strong>Phase 2 — scraper reliability</strong></summary>

- [ ] Replace the raw `requests` + BeautifulSoup scraper (`scrape_website` in `tools.py`) with Firecrawl's free tier — current scraper gets blocked by anti-bot pages

</details>

<details>
<summary><strong>Phase 3 — new features</strong></summary>

- [ ] **Persistent + searchable chat history** — SQLite for persistence, reusing the existing TF-IDF pattern (`store_in_vector_db`/`retrieve_context`) for search, rather than reintroducing ChromaDB. Open question: Streamlit has no auth, so there's no identity yet to key "your chats" on.
- [ ] **Live pipeline progress visualization** — show the search → scrape → chunk → retrieve steps as they happen, not just a spinner

</details>

<details>
<summary><strong>Longer-term — deployment rebuild</strong></summary>

- [ ] Keep this Streamlit build as a showcase
- [ ] Rebuild for production on a genuinely free host (Render / Hugging Face Spaces / similar) — target stack: Python backend (FastAPI/Flask) + plain HTML/JS frontend, budget strictly $0

</details>

---

## 6. Current architecture at a glance

```
User (Streamlit chat)
        │
        ▼
  ReAct Agent (Groq · llama-3.3-70b-versatile)
        │  reasons, picks a tool, reads the result, repeats (max 15x)
        ▼
   ┌────────────┬──────────────────┬─────────────────────┐
   │ get_stock_  │ perform_web_     │ process_research     │
   │ info        │ search           │ (search→scrape→      │
   │ (yfinance)  │ (Serper.dev)     │  chunk→TF-IDF)        │
   └────────────┴──────────────────┴─────────────────────┘
        │
        ▼
  Final answer + optional stock chart back to the user
```