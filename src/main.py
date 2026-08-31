import httpx
import streamlit as st
from langchain_groq import ChatGroq

from tools import (
    get_stock_info,
    perform_web_search,
    process_research,
    query_uploaded_document
)

from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

# Groq retires models without notice - when a model is decommissioned the API
# starts returning 404 model_not_found and every query fails. That is what
# killed the previous llama-3.3-70b-versatile pin. Keep the name in one
# constant so the next retirement is a one-line fix, and re-check it against
# GET https://api.groq.com/openai/v1/models if that 404 ever shows up again.
#
# Model choice is tied to the agent type below and was picked empirically, not
# by size. openai/gpt-oss-120b is the bigger model but is unusable with the
# ReAct agent: it emits a *native* tool call instead of typing "Action:", and
# Groq rejects that with "Tool choice is none, but model called a tool" because
# a ReAct agent binds no tools. qwen3.6-27b follows the text protocol reliably.
# If this model is ever swapped again, re-test the agent type too.
GROQ_MODEL = "qwen/qwen3.6-27b"

# --- Live Groq quota tracking -------------------------------------------------
# Groq reports the account's remaining budget in rate-limit response headers,
# but langchain-groq surfaces only the parsed body, not the headers. Rather
# than spend a request pinging the API just to read them, an httpx response
# hook reads them off the agent's own calls - so this costs zero extra calls
# and is always as fresh as the last thing the agent did.
#
# Account-wide by nature (the quota belongs to the API key, not the visitor),
# so module-level state is the right scope here - unlike per-user chat state.
_RATE_LIMITS: dict[str, str] = {}


def _capture_rate_limits(response: httpx.Response) -> None:
    _RATE_LIMITS.update(
        {k.lower(): v for k, v in response.headers.items()
         if k.lower().startswith("x-ratelimit-")}
    )


def get_rate_limits() -> dict[str, str]:
    """Latest Groq rate-limit headers seen. Empty until the first LLM call."""
    return dict(_RATE_LIMITS)


llm = ChatGroq(
    model_name=GROQ_MODEL,
    temperature=0,
    api_key=st.secrets["GROQ_API_KEY"],
    max_retries=3,
    # gpt-oss is a reasoning model. Left on the default it can emit its chain of
    # thought into the message body, which the ReAct parser then reads as a
    # malformed Thought/Action block. "hidden" keeps the body to the answer only.
    reasoning_format="hidden",
    http_client=httpx.Client(event_hooks={"response": [_capture_rate_limits]}),
)

tools = [
    get_stock_info,
    perform_web_search,
    process_research,
    query_uploaded_document
]

template = """
You are an expert financial research assistant with memory of the conversation so far.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: input to the action
Observation: result of the action
(repeat Thought/Action/Action Input/Observation if needed)

Thought: I now know the final answer
Final Answer: your response

IMPORTANT RULES:
* Use at most 2 tool calls, then answer. Fewer is better.
* Never write Action: None
* Never invent tool names.
* NEVER repeat a tool call you have already made with the same input. If the
  Observation above already contains the data, use it — do not call again.
* After an Observation gives you enough to answer, immediately write
  "Thought: I now know the final answer" then "Final Answer:". Do not call
  another tool just to double-check.
* Every response must contain either an "Action:" line or a "Final Answer:"
  line — never neither.

TOOL SELECTION (pick exactly one per step):
* get_stock_info — any question about price, valuation, market cap, P/E, EPS,
  dividends, or performance of specific companies. Batch all tickers into ONE call.
* perform_web_search — quick factual/current-events questions answerable from
  headlines (e.g. "why did Tesla drop today").
* process_research — when the user says "research", "deep dive", "analyze",
  "summarize", or wants detail beyond a headline. Never pair this with
  perform_web_search for the same question.
* query_uploaded_document — any question about an uploaded document/report/PDF.

BEHAVIOUR RULES:
* You have memory of the previous conversation — use it to resolve pronouns like "it", "they", "this company" by referring to the chat history below.
* If the user asks about a short uppercase word (e.g. TTRO, AAPL, NVDA, TEAM), always treat it as a potential stock ticker and call get_stock_info with that ticker directly.
* If the user asks about two or more companies (e.g. a comparison), call get_stock_info ONCE with all their tickers comma-separated (e.g. "TSLA, NVDA, AAPL") instead of calling it once per company.
* If get_stock_info returns "Could not find valid stock data", inform the user that the ticker is invalid or the company is private/not publicly traded.

EXAMPLES:
Question: Do a deep research on Nvidia's AI investments
Thought: The user asked for deep research, so I should use process_research.
Action: process_research
Action Input: Nvidia AI investments
Observation: <research passages>
Thought: I now know the final answer
Final Answer: <summary of the passages>

Question: Compare TSLA and AAPL
Thought: This is a valuation comparison, so one batched get_stock_info call covers both.
Action: get_stock_info
Action Input: TSLA, AAPL
Observation: <two summaries>
Thought: I now know the final answer
Final Answer: <comparison table>

OUTPUT FORMAT RULES:
* Only use the structured comparison format (Industry / Market Cap / Recent Performance / Key Strengths / Key Risks / Overall Comparison) when the user EXPLICITLY asks to compare two or more companies.
* For single company queries, respond in clean concise paragraphs — no bullet points unless the user asks.
* For follow-up questions like "is it better than X" or "compare it with Y", use the previous company from chat history as the first company.

If the query is completely unrelated to finance, investing, stocks, companies, or business:
Final Answer: I can only assist with financial research, stock data, and company analysis.

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# Keeps last 6 exchanges (3 user + 3 assistant) in memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=6,
    return_messages=False
)

agent = create_react_agent(
    llm,
    tools,
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors="Please provide a Final Answer.",
    # Each iteration is one LLM call. A normal answer takes 2-3; 8 leaves
    # headroom for a format retry while capping the worst case at roughly
    # half what 15 could burn on a runaway loop.
    max_iterations=8,
    # "generate" was accepted by the LangChain version this was written
    # against, but current versions raise ValueError("Got unsupported
    # early_stopping_method") the moment max_iterations is actually hit -
    # turning a graceful stop into a crash. "force" is the supported value:
    # it stops and returns the stop message instead of a broken run.
    early_stopping_method="force",
    memory=memory
)