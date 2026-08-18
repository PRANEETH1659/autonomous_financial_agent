# Autonomous Financial Research Agent — Progress Tracker

Last updated: 2026-08-15

A living record of what this project is, what's been built, what's currently broken or in flux, and what's next. Sections with `<details>` are collapsible — click to expand.

---

## 1. What this project is

A Streamlit chat app backed by a LangChain **ReAct agent** that can:
- Look up live stock data and valuation metrics (`yfinance`)
- Search the web (Serper.dev / Google)
- Run a research pipeline: search (Serper) → scrape (Firecrawl, with a BeautifulSoup fallback) → chunk → **BM25** retrieve, for questions that need synthesis across sources — with the steps shown live in the UI as they happen, not hidden behind a spinner
- Answer questions about a **user-uploaded PDF** (earnings report, 10-K) using that same retrieval pipeline

The LLM is the decision-maker: it reads the user's question, picks a tool (or two), reads the tool's result back, and writes a final answer — the defining trait of an *agent* versus a plain chatbot wrapper.

---

## 2. Current status

<details open>
<summary><strong>✅ Robustness pass — 6 latent crash/correctness bugs found and fixed, eval now 11/11 (2026-08-15)</strong></summary>

A deliberate hunt for remaining problems rather than a feature pass. Each was reproduced with a probe before being fixed, and re-verified after:

- **`ZeroDivisionError` crash on unusual PDFs** — `BM25Okapi` divides by the corpus's average document length, so a document whose chunks all tokenize to nothing (a page of only punctuation/symbols) crashed indexing. Now filters to chunks with at least one token, keeping `chunks` and the BM25 index **index-aligned** (which `retrieve_context` depends on). Side benefit: punctuation-only chunks no longer surface as junk retrieval results.
- **Upload reported false success** — `app.py` ignored `store_in_vector_db`'s return value, so a failed index still showed *"Indexed N chunks… Ask me about it!"* while every later question answered "No data stored yet." Now checks the result and reports an honest error; also reports the *kept* chunk count, not the pre-filter count.
- **Failed web searches were cached for 10 minutes** — `_search_web` returned errors as ordinary strings, so `@st.cache_data` happily cached them: one transient Serper blip would keep returning the same stale error for the full TTL. Errors now raise a `SearchError` (exceptions aren't cached) and the tool wrappers convert it back to a string for the agent, so the ReAct loop still sees a normal observation. Verified: a simulated outage then a real call returns fresh results, not the cached failure.
- **Firecrawl could crash the whole pipeline** — `response.json()` sat outside the `try`, so a 200 response with a non-JSON body (a proxy/CDN error page) raised out of `_scrape_with_firecrawl`, past `scrape_website`, and killed `process_research` instead of falling back to the bs4 scraper. Now guarded, including unexpected JSON shapes.
- **`_scrape_with_bs4` could return `None`** — every path happened to return, but an implicit `None` would have crashed `process_research`'s `scraped.startswith(...)` check. Made explicit.
- **App logs invisible in production (open since 2026-08-09)** — now fixed. `logging.basicConfig()` is a no-op once Streamlit has configured the root logger, so the app's own INFO lines never appeared. Configuring the `tools` logger directly (own handler, `propagate = False`) sidesteps the root logger entirely. Verified: pipeline steps log correctly even with the root logger pre-set to WARNING.
- Also removed dead imports (`os`, `datetime`, `timedelta`) left over from earlier architectures.

**Eval suite extended to 11 cases**, closing the two gaps that would have let today's work silently regress:
- `llm_call_budget_respected` — fails if any query exceeds **4** LLM calls (observed: 2). This makes the optimization below a permanent, automatically-enforced guarantee rather than a one-off measurement.
- `uploaded_document_question_uses_doc_tool` — seeds the document store directly (no PDF fixture needed) and asserts the agent routes a document question to `query_uploaded_document` *and* cites the right figure from it.

</details>

<details open>
<summary><strong>✅ LLM-call optimization pass — every query costs exactly 2 calls (2026-08-15)</strong></summary>

The agent was burning 2-4+ Groq calls per question, with the eval suite's two long-standing failures both traceable to wasted/misrouted calls. Rather than treat Groq's `429`s as an external limit to live with, the fix was to cut the number of calls the app actually needs. **Result: 9/9 eval at the time (up from 7/9; the suite was then extended to 11 cases — see the robustness entry above), and every representative query now runs at exactly 2 LLM calls — the theoretical floor for a ReAct agent that uses one tool (1 call to choose the tool, 1 to write the answer), with zero format-retries.**

Measured after the change, via an `on_llm_start` counter:

| Query | LLM calls |
|---|---|
| "What's the current stock price of AAPL?" | 2 |
| "Compare TSLA, NVDA, and AAPL by market cap and valuation." | 2 |
| "Do a deep research on Nvidia's recent AI investments…" | 2 |
| Bare ticker (`TSLA`) — **fast path** | **0** |

What changed:

- **Zero-LLM fast path for bare tickers** (`app.py`) — a message that is *only* a ticker (`TSLA`, `tsla?`) is answered straight from yfinance, skipping the agent entirely. This is the single most common demo query and previously cost 2-3 Groq calls to have the LLM read tool output back nearly verbatim. The exchange is still written into agent memory via `save_context`, so follow-ups ("is it overvalued?") resolve the pronoun correctly. Verified it fires on `TSLA`/`tsla?` and correctly does **not** fire on `Compare TSLA and AAPL`, `ZZZZINVALID`, or off-topic questions.
- **Prompt rewrite to stop wasted calls** (`main.py`) — added an explicit `TOOL SELECTION` section (one line per tool, with the "research/deep dive/analyze/summarize → `process_research`, never pair it with `perform_web_search`" rule), a hard "NEVER repeat a tool call with the same input" rule, "every response must contain either an `Action:` or a `Final Answer:` line" (the `_Exception: Please provide a Final Answer.` retries were pure wasted calls), and two worked ReAct **examples** showing the exact expected format. Both previously-failing eval cases now pass.
- **`max_iterations` 15 → 8** — each iteration is one LLM call, so this halves the worst-case cost of a runaway loop. Normal answers take 2-3, so the headroom is still ample.
- **Serper search cached** (`@st.cache_data`, 10 min) — `perform_web_search` and `process_research` now share one `_search_web` helper, so escalating from one to the other on the same query no longer pays for a duplicate API round-trip. Measured: 2.58s → 0.000s on the repeat call.
- **`get_stock_info` dedupes tickers** — the eval's `multi_company_batches_into_one_call` failure involved the model re-issuing an identical call after a format stumble; repeated tickers within one call are now collapsed. Verified `"TSLA, TSLA, tsla"` returns 1 summary, not 3.
- **`process_research` returns usable context** — was truncating to `context[:500]` with `k=2`, so little that the agent often needed *another* tool call to actually answer. Now `k=4` and a 2000-char cap: a larger observation is far cheaper than an extra round-trip.
- **`max_retries=3` on `ChatGroq`** — makes transient 429s recover cleanly instead of surfacing as a failed answer.

</details>

<details open>
<summary><strong>✅ BM25 retrieval + PDF document Q&A (2026-08-15)</strong></summary>

- [x] **TF-IDF + cosine → BM25** (`tools.py`) — `store_in_vector_db`/`retrieve_context` now use `rank_bm25`'s `BM25Okapi`. BM25 keeps TF-IDF's rare-terms-matter idea but adds **term-frequency saturation** (`k1`) and **length normalization** (`b`), so a chunk repeating "revenue" 10× no longer scores linearly above one mentioning it twice — better suited to scraped financial articles where company names repeat heavily without adding signal. **Dropped `scikit-learn`, `scipy`, `joblib`, `threadpoolctl` entirely** (nothing else used them); `rank-bm25` pulls in only `numpy`, which was already present. Smaller install = faster, less fragile free-tier deploys.
- [x] **Upload a PDF and ask about it** — sidebar `st.file_uploader` → `parse_pdf` (pypdf) → `chunk_text` → BM25-indexed under its own `"uploaded_document"` collection → new `query_uploaded_document` agent tool. Reuses the existing pipeline wholesale; `_chunk_store` was already keyed by collection name, so the web-research and uploaded-doc indexes coexist without interfering. "Clear conversation" also clears the doc. Verified end-to-end against a real PDF: extract → chunk → index → retrieve correct excerpt → clear resets to "No data stored yet."
- **Scope note:** text-based PDFs only. Scanned/image-only PDFs have no text layer and surface a clear error rather than failing silently — OCR was deliberately declined to avoid a Tesseract system dependency that would complicate free-tier hosting.

</details>

<details>
<summary><strong>✅ Working and verified live (2026-08-09)</strong></summary>

- [x] **Firecrawl scraping** — `scrape_website` now tries Firecrawl's `/scrape` API first (handles JS-rendered pages the old scraper couldn't), falls back to the original BeautifulSoup scraper if Firecrawl has no key or fails after retries. Verified with a raw API smoke test, a full browser-driven UI test, and a direct pipeline invocation with log capture — confirmed real markdown content coming back via Firecrawl, no fallback triggered.
- [x] **Live pipeline visualization** — `st.status()` panel replaces the old plain spinner. Shows which tool the agent picks (`get_stock_info` / `perform_web_search` / `process_research`) live via a LangChain callback, plus `process_research`'s own steps (`🔎 Searching`, `✅ Search completed`, `🌐 Scraping: <url>`, `✂️ Chunked`, `📊 Indexed`, `✅ Context retrieved`) streaming in as they happen. Verified via Playwright: watched the agent try `perform_web_search`, escalate to `process_research`, and stream its internal steps in real time; screenshot confirms clean rendering with no console errors.
- [x] **Chart/answer ticker mismatch fixed** — found via manual stress-testing: a 3-way comparison ("Compare TSLA, NVDA, AAPL...") had the text answer conclude NVDA, while the chart underneath silently showed TSLA (confirmed by matching the exact $328.58 figure from an earlier query) — because the chart picker (`find_ticker_in_text` in `app.py`) scanned the raw *prompt* and grabbed whichever ticker appeared first, unrelated to what the answer was actually about, with no label to reveal the mismatch. Fixed by preferring a ticker found in the *answer* text over the prompt, and adding a `📊 Chart & metrics: <ticker>` caption so it's never ambiguous even when the heuristic has to fall back. Verified live: chart now correctly shows NVDA, labeled.
- [x] **Batched multi-ticker `get_stock_info`** — the same stress test showed a 3-company comparison burning 3 separate `get_stock_info` calls (one per ticker), blowing past the prompt's "at most 2 tool calls" guidance and leaving no budget for the news-catalyst part of the question. `get_stock_info` now accepts one *or* comma-separated multiple tickers (`"TSLA, NVDA, AAPL"`) in a single call — the old single-ticker logic was extracted into `_get_stock_info_for_one` and is reused per ticker internally, so it's still one function, just batched. System prompt updated to instruct the agent to batch instead of repeating the call. Verified live: the same 3-company query now does exactly one `get_stock_info` call for all three companies (down from three), confirmed via a mid-run status-panel capture.

</details>

<details>
<summary><strong>✅ Guardrails / eval suite built — `src/eval_agent.py` (2026-08-09 night)</strong></summary>

A standalone script (not pytest — hits real Groq/yfinance/Serper/Firecrawl APIs, so it's run manually, not on every commit) that runs 9 representative queries against the real `agent_executor` and checks **which tools actually fired**, via a callback tracker, not just whether the final answer sounds plausible. First run: **7/9 passed**. Exactly the outcome evals are for — it immediately surfaced more precise findings than the manual stress-testing earlier today:

- **`multi_company_batches_into_one_call` — FAILED.** The batching fix (below) does work on a clean run, but when a parsing-error retry happens mid-run (see the `_Exception` problem, logged 2026-08-08), the model sometimes re-issues the *identical* `get_stock_info("TSLA, NVDA, AAPL")` call a second time instead of reusing data it already has. So "1 call for N companies" holds only when the model doesn't stumble on ReAct's output format first.
- **`deep_research_query_uses_pipeline` — FAILED.** Asking for "deep research on Nvidia's AI investments" did not call `process_research` — it used `perform_web_search` + `get_stock_info` instead, never touching the actual search→scrape→chunk→retrieve pipeline. The word "research" isn't a strong enough signal in the tool's docstring/prompt for the LLM to prefer it over a plain web search.
- **Notable inconsistency:** manual testing earlier today showed the agent skip the "recent news catalyst" part of a 3-company comparison every time. The automated eval's equivalent case (`comparison_plus_catalyst_uses_search_tool`) **passed** — a search tool *was* called. Same category of question, different outcome on a different run. Worth treating tool-selection behavior as probabilistic, not fixed, until proven otherwise across multiple runs.
- Also observed, not a failure: Groq returned `429 Too Many Requests` partway through the 9-query run (reasonable, given ~15+ LLM calls in a couple minutes) — its client auto-retried with backoff and every test still completed. No fix needed, just noted as expected behavior under burst load.

**Update 2026-08-15 — both failures fixed, suite now 9/9.** `multi_company_batches_into_one_call` was resolved by the "NEVER repeat a tool call with the same input" prompt rule plus ticker dedupe inside `get_stock_info`; `deep_research_query_uses_pipeline` was resolved by rewriting the tool docstrings and adding an explicit `TOOL SELECTION` block + a worked `process_research` example to the prompt (the word "research" alone was too weak a signal). See the LLM-call optimization entry at the top of this section.

</details>

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
- [ ] The live-progress callback in `tools.py` (`set_progress_callback`) is **module-level global state** — assumes one active research session at a time. Fine for solo use/demos, not safe if multiple users hit the deployed app concurrently (their progress messages could cross-talk)
- [ ] Firecrawl's free tier is 1,000 credits/month, non-rolling — no usage tracking/alerting yet if it runs out mid-month (falls back to BS4 silently, which still works, just lower quality on JS-heavy pages)
- [ ] No persistent/searchable chat history yet (see backlog)
- [x] ~~No automated guardrails/evals yet~~ — `src/eval_agent.py` built 2026-08-09, now **11/11** (includes an LLM-call budget guard and an uploaded-document routing case)
- [ ] Streamlit is the interim host, not the final deployment target
- [ ] **Retrieval depth on long documents** — `retrieve_context` returns k chunks × 500 chars (≈2000 chars at k=4). Fine for "what does the report say about X"; weak for "summarize this entire 10-K", where a 100-page filing gets represented by ~4 scattered passages. Would need a map-reduce style summarization pass to do properly — but that costs multiple LLM calls per document, which cuts directly against the call-minimization work above. Deliberately not done.
- [ ] **Single-source research** — `process_research` stops at the first link that scrapes successfully and never cross-checks a second source. Firecrawl timeouts (observed a live HTTP 408 on CNBC) are handled by retry/fallback, but a single unreliable source still shapes the answer.
- [ ] **Uploaded PDFs are OCR-free** — scanned/image-only PDFs are rejected with a clear message rather than parsed (deliberate scope choice, see above)

</details>

---

## 3. Problems hit today, and how they were resolved

<details>
<summary><strong>🔴 2026-08-15 — App crashed on startup: <code>ImportError: cannot import name 'ModelProfile'</code> (wrong Python interpreter)</strong></summary>

**Symptom:** `streamlit run app.py` failed immediately with `ImportError: cannot import name 'ModelProfile' from 'langchain_core.language_models'`, raised from `langchain_groq/chat_models.py`.

**Root cause — not a code bug.** The traceback's paths gave it away: `C:\Users\PRANEETH\AppData\Roaming\Python\Python313\site-packages\...`. The app was being launched by the **global Python 3.13 install**, not the project's `venv` (Python 3.11.9). The two environments have incompatible package sets:

| | Interpreter | `langchain-core` | `langchain-groq` |
|---|---|---|---|
| Project venv (correct) | `venv\Scripts\python.exe` (3.11.9) | 0.3.86 | 0.3.8 |
| Global (was being used) | `C:\Python313\python.exe` | 0.3.86 | **1.1.3** |

`langchain-groq` 1.1.3 imports `ModelProfile`, which only exists in `langchain-core` **1.x+** — so that global combination was never valid. It also lacked `rank_bm25` entirely, which would have been the next error. The global env also has `langgraph`, `langchain-chroma`, `langchain-huggingface` installed from unrelated work.

**Fix:** launch through the venv — either `.\venv\Scripts\Activate.ps1` first (prompt shows `(venv)`), or call it directly: `& ".\venv\Scripts\python.exe" -m streamlit run src\app.py`. Deliberately did **not** "fix" the global environment by downgrading `langchain-groq` there — it's broken only relative to this project, and changing it could disturb other work.

**Lesson:** this is the same dependency-version class of failure the 2026-08-08 pinning pass was meant to prevent — but pinning only protects the environment you actually run. When an `ImportError` names a package you didn't touch, **read the site-packages path in the traceback first** to confirm which interpreter is running before debugging the code.

</details>

<details>
<summary><strong>🟢 2026-08-15 — Latent bug found while optimizing: "AI" is a real ticker</strong></summary>

While adding a stopword filter to `find_ticker_in_text` (to stop pointless yfinance probes on every uppercase word in an answer), found that several common financial-writing acronyms are *themselves* valid tickers — most notably **`AI` (C3.ai)**, plus `PE`, `EV`. Since the chart picker scans the answer text and takes the **first** candidate that resolves, any answer mentioning "AI" — extremely likely in this app — could silently chart C3.ai underneath an answer about Nvidia. Same class of bug as the 2026-08-09 chart/answer mismatch, different trigger. Fixed with a `TICKER_STOPWORDS` blocklist; verified that `"The AI and EPS and PE ratios for NVDA look strong"` now correctly resolves to **NVDA**.

</details>

<details>
<summary><strong>🟣 2026-08-09 — Suspicious pasted "onboarding doc" with an embedded API key</strong></summary>

A message pasted into the chat was formatted like an official Firecrawl agent-onboarding skill, ending with a live-looking `FIRECRAWL_API_KEY` and instructions to run an `npx` install + browser auth flow automatically. Flagged this directly instead of acting on it — an agent being handed a credential plus a command to run, wrapped in trusted-looking instructions, is exactly the shape of a prompt-injection payload, regardless of whether that particular instance was genuine. Asked where it came from and had the user generate their own key from their own Firecrawl dashboard instead, added directly to `.streamlit/secrets.toml` by the user (never pasted into chat).

**Lesson:** verify credential provenance before use, even when the surrounding document looks legitimate and well-formatted — the formatting is not the trust signal.

</details>

<details>
<summary><strong>🟤 2026-08-09 — Streamlit silently swallows the app's own INFO-level logs</strong></summary>

**Symptom:** while verifying the Firecrawl integration, `tools.py`'s `[STEP 1]/[OK]`-style log lines never appeared in the redirected server log, even across multiple live queries that clearly succeeded.

**Root cause:** Streamlit configures the root logger before the app script runs. `tools.py`'s `logging.basicConfig(level=logging.INFO, ...)` is a no-op in that situation — Python's `basicConfig()` only takes effect if the root logger has no handlers yet, and Streamlit's already does.

**Workaround for verification (not a code fix):** attached a handler directly to the named `tools` logger from a standalone test script and called `process_research` directly — bypasses the root-logger issue since a logger's own handlers fire regardless of root config.

**Resolved 2026-08-15:** fixed by the second option — the `tools` logger now gets its own `StreamHandler` with `propagate = False`, so it emits regardless of what Streamlit did to the root logger. Verified by pre-configuring the root logger to WARNING and confirming the pipeline's INFO lines still appear.

</details>

<details>
<summary><strong>⚪ 2026-08-09 — `lsof` doesn't exist in Windows Git Bash</strong></summary>

Used `lsof -ti:PORT | xargs kill` to free the test port between runs, copied from a generic Unix pattern. It failed silently (piped into `xargs -r`, which no-ops on empty input) — so "stopped" was reported when nothing had actually happened, and old test server processes kept stacking up on the same port across restarts. Switched to `netstat -ano | grep LISTENING` + `taskkill //F //PID`, which are the real Windows equivalents, and cleaned up 4 stacked stale processes as a result.

**Lesson:** on this machine, prefer Windows-native process tools over assumed-Unix ones; verify a cleanup command's own output rather than trusting a "no error" exit code when its main pipeline stage could be silently absent.

</details>

<details>
<summary><strong>🔴 2026-08-08 — "Agent stopped due to iteration limit" (the original complaint)</strong></summary>

**Symptom:** big/multi-part questions would fail outright with a generic stopped message.

**Root cause, found via verbose tracing:** the agent's `max_iterations` was capped at 5, and the ReAct prompt format requires the LLM to literally type the phrase `Final Answer:` before LangChain accepts the response as complete. Models sometimes write the correct answer but skip that exact phrase on the first try, burning iterations on retries.

**Fix:** raised `max_iterations` to 15 and added `early_stopping_method="generate"` (so even a worst-case timeout still produces a best-effort answer instead of a bare error). Verified by replaying the exact original failing question (Nvidia performance + investors) — now completes cleanly in 3 iterations.

</details>

<details>
<summary><strong>🟡 2026-08-08 — requirements.txt / .gitignore silently broken (UTF-16 encoding)</strong></summary>

**Symptom:** deploy-time dependency failures; `.gitignore` rules not actually excluding files from `git status`.

**Root cause:** both files were saved in UTF-16 at some point, which pip and git don't parse correctly, but don't loudly error on either — they just silently fail to work as intended.

**Fix:** rewrote both as plain UTF-8/ASCII. Also discovered the dev `venv/` had drifted to 176 installed packages against only 12 declared ones (leftovers from an earlier ChromaDB/HuggingFace-embeddings architecture) — rebuilt the venv from scratch against a fully-pinned `requirements.txt` (full `pip freeze` output, not just direct dependencies) to guarantee dev/deploy parity.

</details>

<details>
<summary><strong>🔵 2026-08-08 — The Groq → Gemini → Groq detour</strong></summary>

**Why it happened:** wanted to try Gemini as a "modern" alternative to Groq's Llama 3.3 70B.

**What went wrong:**
1. `langchain-google-genai`'s latest version required `langchain-core>=1.0`, breaking the rest of the pinned stack — fixed by pinning `langchain-google-genai==2.1.10` + `langchain-core==0.3.86`.
2. Gemini's free tier turned out to be a hard **20 requests/day** cap — far too tight for an agent that costs 2-4 LLM calls per question. Tried other Gemini models hoping for separate quota buckets; one 404'd (deprecated for new users), one had a `limit: 0` (no free access at all on this project). Lesson: don't guess model names, check the live model list for the actual account.
3. Researched alternatives properly (live web search, since API pricing changes fast): **Groq's free tier is 1,000 requests/day** for this exact model — 50x more headroom than Gemini, and the only genuinely-free option that comfortably clears a "100-150 requests/day" target.

**Decision:** reverted to Groq.

</details>

<details>
<summary><strong>🟠 2026-08-08 — Groq's native tool-calling is broken (the twist)</strong></summary>

While reverting to Groq, rewrote the agent to use `create_tool_calling_agent` (LangChain's structured tool-calling, which had been the fix for Gemini's format-compliance problem). On Groq, this reproducibly failed:

```
groq.APIError: Failed to call a function. Please adjust your prompt.
failed_generation: '<function=get_stock_info{"ticker": "TSLA"}</function>'
```

Confirmed deterministic (not a streaming artifact — same failure with `disable_streaming="tool_calling"` set). **The fix that helped Gemini actively breaks Groq.** Tested the original legacy ReAct text-parsing format directly against Groq instead — worked perfectly, zero parsing failures, immediate correct `Final Answer:` compliance.

**Lesson learned, logged for future-me:** the right agent architecture (`create_react_agent` vs `create_tool_calling_agent`) is provider-specific, not a universal best practice. Re-verify empirically if the LLM provider ever changes again — don't assume the last answer transfers.

</details>

<details>
<summary><strong>🟢 2026-08-08 — Zombie server processes causing flaky test results</strong></summary>

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
| 2026-08-08 | Full dependency reliability pass, repo cleanup, iteration-limit fix, Groq↔Gemini evaluation, agent architecture fix |
| 2026-08-09 | Firecrawl scraping integration (with BS4 fallback) + live pipeline visualization (`st.status` panel + tool-selection callback) — both verified via live browser tests |
| 2026-08-09 (evening) | Manual stress-testing surfaced two real bugs, both fixed same-day: chart/answer ticker mismatch, and 3x redundant `get_stock_info` calls on multi-company questions (fixed via batched comma-separated ticker input) — all verified via live browser + status-panel captures |
| 2026-08-15 | Retrieval upgraded TF-IDF → **BM25** (dropped sklearn/scipy); **PDF upload + document Q&A** added; **LLM-call optimization pass** (every query down to exactly 2 calls, bare tickers to 0 via a fast path); **robustness pass** fixing 6 latent crash/correctness bugs. Eval 7/9 → **11/11**, suite extended with a call-budget guard and a document-routing case |

---

## 5. Backlog — what's next

<details>
<summary><strong>Phase 1 (deployment reliability) — mostly done as of 2026-08-08</strong></summary>

- [x] Pin exact dependency versions
- [x] Verify clean-venv install matches deployment
- [x] Fix `.gitignore` encoding
- [ ] Final check: deploy to Streamlit Cloud (or equivalent) and confirm the pinned `requirements.txt` installs cleanly there too

</details>

<details>
<summary><strong>Phase 2 — scraper reliability — done as of 2026-08-09</strong></summary>

- [x] Replace the raw `requests` + BeautifulSoup scraper with Firecrawl's free tier (BS4 kept as automatic fallback, not removed)

</details>

<details>
<summary><strong>Phase 3 — new features</strong></summary>

- [x] **Live pipeline progress visualization** — done 2026-08-09
- [x] **Guardrails / evals** — done 2026-08-09 (`src/eval_agent.py`), passing **11/11** as of 2026-08-15
- [x] **Document upload + Q&A** — done 2026-08-15 (text-based PDFs)
- [ ] **Persistent + searchable chat history** — SQLite for persistence, reusing the existing **BM25** pattern (`store_in_vector_db`/`retrieve_context`) for search, rather than reintroducing ChromaDB. Open question: Streamlit has no auth, so there's no identity yet to key "your chats" on.
- [ ] **Richer charts** — scoped but not built: normalized multi-ticker comparison (indexed to 100), volume bars, 50/200-day moving averages, valuation bar chart for comparisons. Currently only a single Close-price line chart + 3 metric tiles.

</details>

<details>
<summary><strong>Longer-term — deployment rebuild</strong></summary>

- [ ] Keep this Streamlit build as a showcase
- [ ] Rebuild for production on a genuinely free host (Render / Hugging Face Spaces / similar) — target stack: Python backend (FastAPI/Flask) + plain HTML/JS frontend, budget strictly $0

</details>

<details>
<summary><strong>Not urgent, noted for later</strong></summary>

- [x] ~~Fix `logging.basicConfig()` being a no-op under Streamlit~~ — done 2026-08-15, `tools` logger now configured directly with its own handler
- [ ] Per-session (not global) progress-callback state in `tools.py`, if the app ever needs to support concurrent users

</details>

---

## 6. Current architecture at a glance

```
User (Streamlit chat)
        │
        ├─── bare ticker? ("TSLA") ──► yfinance ──► answer   ⚡ 0 LLM calls
        │                                (also written into agent memory
        │                                 so follow-ups still resolve "it")
        ▼
  ReAct Agent (Groq · llama-3.3-70b-versatile, max_retries=3)
        │  picks a tool, reads the result, answers (max 8 iterations)
        │  typical cost: 2 LLM calls — 1 to choose the tool, 1 to answer
        │  ── tool choice + pipeline steps streamed live to an st.status() panel ──
        ▼
   ┌─────────────┬──────────────────┬────────────────────────────┬──────────────────┐
   │ get_stock_  │ perform_web_     │ process_research           │ query_uploaded_  │
   │ info        │ search           │ search (Serper, cached) →  │ document         │
   │ (yfinance,  │ (Serper.dev,     │ scrape (Firecrawl, BS4     │ (BM25 over an    │
   │ cached 5m,  │ cached 10m)      │ fallback) → chunk → BM25   │ uploaded PDF,    │
   │ deduped)    │                  │ retrieve (k=4, 2000 chars) │ pypdf-parsed)    │
   └─────────────┴──────────────────┴────────────────────────────┴──────────────────┘
        │                    └── both share one cached _search_web() ──┘
        ▼
  Final answer + optional stock chart back to the user
```

**Shared retrieval store:** `_chunk_store` is a dict keyed by collection name — `"financial_research"` (web pipeline) and `"uploaded_document"` (PDF) use the same BM25 code paths without interfering. Adding a third retrieval source means adding a collection name, not new retrieval logic.
