# main.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools import get_stock_info, perform_web_search, process_research

# ✅ Correct import for Hub
from langchain_core.prompts  import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import tool
from langchain.agents.agent import AgentExecutor


# Load environment variables from .env file
load_dotenv()

# 1. Initialize the LLM (THE BRAIN)
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Define the list of tools for the agent
tools = [get_stock_info, perform_web_search, process_research]


#Initializing Memory
#Memory_key  must watch the vairable in your template!!!

# Define the ReAct prompt manually (No more Hub errors!)
template = """
Answer the following questions as best you can.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: think about what to do

Action: one of [{tool_names}]

Action Input: input to the action

Observation: result of the action

(Thought/Action/Observation can repeat)

IMPORTANT:
Use at most 2 tool calls before producing a Final Answer.

Thought: I now know the final answer

Final Answer: the final answer to the original question

Question: {input}

Thought: {agent_scratchpad}
"""



# ✅ Pull the prompt from LangChain Hub
prompt = PromptTemplate.from_template(template)

# Create the agent
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)


