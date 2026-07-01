import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

from tools import (
get_stock_info,
perform_web_search,
process_research
)

from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor

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
Answer the following questions as best you can.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question

Thought: think about what to do

Action: one of [{tool_names}]

Action Input: input to the action

Observation: result of the action

(repeat Thought/Action/Observation if needed)

When you have enough information:

Thought: I now know the final answer

Final Answer: your response

IMPORTANT:

* Use at most 2 tool calls.
* Never write Action: None
* Never invent tool names.
* Always finish with Final Answer.

Only use the comparison format (Industry/Market Cap/Strengths/Risks/Overall) 
when the user EXPLICITLY asks to compare two or more companies. 
For single company or general queries, respond in plain concise paragraphs.

If the query is unrelated to finance,
investing, stocks, companies, or business:

Final Answer: No meaningful financial query detected.

Previous conversation:
{chat_history}


Question: {input}

Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
llm,
tools,
prompt
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
agent=agent,
tools=tools,
verbose=False,
handle_parsing_errors="Please provide a Final Answer.",
max_iterations=5,
memory=memory
)
