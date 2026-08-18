"""
Manual eval suite for the financial research agent. Runs a small set of
representative queries against the real agent_executor and checks which
tools actually fired - not just whether the final answer looks plausible.

Hits real APIs (Groq, yfinance, Serper, Firecrawl), so this is meant to be
run manually when checking agent behavior, not on every commit.

Usage: python eval_agent.py
"""
import sys

from langchain_core.callbacks import BaseCallbackHandler

from main import agent_executor
from tools import chunk_text, store_in_vector_db, clear_vector_db


# A ReAct answer that uses one tool costs 2 LLM calls (choose tool, write
# answer). 4 allows a two-tool question plus one format retry; above that
# something has regressed - a repeated tool call, or the model failing to
# produce "Final Answer:" and burning retries.
MAX_LLM_CALLS_PER_QUERY = 4


class ToolCallTracker(BaseCallbackHandler):
    """Records every (tool_name, tool_input) the agent chooses during a run,
    plus how many LLM calls that run cost."""

    def __init__(self):
        self.calls = []
        self.llm_calls = 0

    def on_agent_action(self, action, **kwargs):
        self.calls.append((action.tool, str(action.tool_input)))

    def on_llm_start(self, *args, **kwargs):
        self.llm_calls += 1


def invoke_and_track(query: str) -> tuple[str, list[tuple[str, str]], int]:
    tracker = ToolCallTracker()
    response = agent_executor.invoke(
        {"input": query},
        config={"callbacks": [tracker]},
    )
    return response.get("output", ""), tracker.calls, tracker.llm_calls


def tool_was_called(calls, name: str) -> bool:
    return any(c[0] == name for c in calls)


def count_calls(calls, name: str) -> int:
    return sum(1 for c in calls if c[0] == name)


def tool_called_with_all(calls, name: str, must_contain: list) -> bool:
    for tool_name, tool_input in calls:
        if tool_name == name and all(term.lower() in tool_input.lower() for term in must_contain):
            return True
    return False


TEST_CASES = [
    {
        "name": "single_ticker_lookup",
        "query": "What's the current stock price of AAPL?",
        "check": lambda calls, answer: (
            tool_was_called(calls, "get_stock_info"),
            "expected get_stock_info to be called",
        ),
    },
    {
        "name": "bare_ticker_word",
        "query": "TSLA",
        "check": lambda calls, answer: (
            tool_was_called(calls, "get_stock_info"),
            "a bare ticker-shaped word should trigger get_stock_info directly",
        ),
    },
    {
        "name": "multi_company_batches_into_one_call",
        "query": "Compare TSLA, NVDA, and AAPL by market cap and valuation.",
        "check": lambda calls, answer: (
            count_calls(calls, "get_stock_info") == 1
            and tool_called_with_all(calls, "get_stock_info", ["TSLA", "NVDA", "AAPL"]),
            f"expected exactly 1 batched get_stock_info call covering all 3 tickers, "
            f"got {count_calls(calls, 'get_stock_info')} call(s): {calls}",
        ),
    },
    {
        "name": "comparison_plus_catalyst_uses_search_tool",
        "query": (
            "Compare TSLA, NVDA, and AAPL - give price, valuation, and the biggest "
            "recent news catalyst for each, then say which is most undervalued."
        ),
        "check": lambda calls, answer: (
            tool_was_called(calls, "perform_web_search") or tool_was_called(calls, "process_research"),
            "question explicitly asks for news catalysts - expected a search-family "
            f"tool to be called, only got: {[c[0] for c in calls]}",
        ),
    },
    {
        "name": "deep_research_query_uses_pipeline",
        "query": "Do a deep research on Nvidia's recent AI investments and summarize the key points.",
        "check": lambda calls, answer: (
            tool_was_called(calls, "process_research"),
            f"'deep research' phrasing should trigger process_research, got: {[c[0] for c in calls]}",
        ),
    },
    {
        "name": "off_topic_refusal",
        "query": "What is the capital of France?",
        "check": lambda calls, answer: (
            "financial research" in answer.lower() or "can only assist" in answer.lower(),
            f"expected the canned refusal message, got: {answer[:150]!r}",
        ),
    },
    {
        "name": "invalid_ticker_handled_gracefully",
        "query": "What's the stock price of ZZZZINVALID?",
        "check": lambda calls, answer: (
            any(w in answer.lower() for w in ("invalid", "could not find", "not a valid", "not publicly traded")),
            f"expected the answer to flag the ticker as invalid, got: {answer[:150]!r}",
        ),
    },
    {
        "name": "single_company_not_forced_into_table_format",
        "query": "Tell me about Apple's stock.",
        "check": lambda calls, answer: (
            answer.count("|") < 5,
            "single-company queries should get prose, not the comparison-table format",
        ),
    },
]


def run_single_turn_cases():
    results = []
    budget_violations = []
    for case in TEST_CASES:
        agent_executor.memory.clear()
        try:
            answer, calls, llm_calls = invoke_and_track(case["query"])
            passed, reason = case["check"](calls, answer)
            if llm_calls > MAX_LLM_CALLS_PER_QUERY:
                budget_violations.append(f"{case['name']} used {llm_calls}")
        except Exception as e:
            passed, reason = False, f"raised an exception: {e}"
        results.append((case["name"], passed, reason))

    results.append((
        "llm_call_budget_respected",
        not budget_violations,
        f"queries exceeded the {MAX_LLM_CALLS_PER_QUERY}-call budget: "
        f"{', '.join(budget_violations)}",
    ))
    return results


def run_uploaded_document_case():
    """Seeds the document store directly (no PDF fixture needed) and checks the
    agent routes a document question to query_uploaded_document rather than
    searching the web for it."""
    agent_executor.memory.clear()
    clear_vector_db("uploaded_document")
    store_in_vector_db(
        chunk_text(
            "Acme Corporation annual report. Fiscal 2025 revenue was 4.2 billion "
            "dollars, up 18 percent year over year. The board approved a special "
            "dividend of 1.50 dollars per share payable in March."
        ),
        collection_name="uploaded_document",
    )
    try:
        answer, calls, _ = invoke_and_track(
            "According to the uploaded report, what was Acme's fiscal 2025 revenue?"
        )
    except Exception as e:
        return ("uploaded_document_question_uses_doc_tool", False, f"raised: {e}")
    finally:
        clear_vector_db("uploaded_document")

    passed = tool_was_called(calls, "query_uploaded_document") and "4.2" in answer
    return (
        "uploaded_document_question_uses_doc_tool",
        passed,
        f"expected query_uploaded_document to be called and the answer to cite "
        f"4.2 billion; got tools={[c[0] for c in calls]}, answer={answer[:150]!r}",
    )


def run_followup_memory_case():
    agent_executor.memory.clear()
    invoke_and_track("What's Tesla's stock price?")
    answer2, calls2, _ = invoke_and_track("Is it overvalued compared to Ford?")
    passed = tool_called_with_all(calls2, "get_stock_info", ["TSLA"]) or "tesla" in answer2.lower()
    reason = (
        "expected the follow-up ('it') to resolve to Tesla via chat memory, "
        f"got tool calls: {calls2}, answer: {answer2[:150]!r}"
    )
    return ("followup_pronoun_resolution", passed, reason)


def main():
    print("Running agent eval suite - this hits real Groq/yfinance/Serper/Firecrawl APIs...\n")

    results = run_single_turn_cases()
    results.append(run_uploaded_document_case())
    results.append(run_followup_memory_case())

    print(f"{'TEST':<45} {'RESULT':<6}")
    print("-" * 70)
    for name, passed, reason in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<45} {status:<6}")
        if not passed:
            print(f"    -> {reason}")

    passed_count = sum(1 for _, p, _ in results if p)
    total = len(results)
    print("-" * 70)
    print(f"{passed_count}/{total} passed")

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()