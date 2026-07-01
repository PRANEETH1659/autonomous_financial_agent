import streamlit as st
from langchain_groq import ChatGroq

from tools import (
    get_stock_info,
    perform_web_search,
    process_research
)

from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=st.secrets["GROQ_API_KEY"]
)

tools = [
    get_stock_info,
    perform_web_search,
    process_research
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
* Use at most 2 tool calls.
* Never write Action: None
* Never invent tool names.
* Always finish with Final Answer.
* You have memory of the previous conversation — use it to resolve pronouns like "it", "they", "this company" by referring to the chat history below.
* If the user asks about a short uppercase word (e.g. TTRO, AAPL, NVDA, TEAM), always treat it as a potential stock ticker and call get_stock_info with that ticker directly.
* If get_stock_info returns "Could not find valid stock data", inform the user that the ticker is invalid or the company is private/not publicly traded.

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
    max_iterations=5,
    memory=memory
)