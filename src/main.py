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
    temperature=0,  # factual and consistent
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Define the list of tools for the agent
tools = [get_stock_info, perform_web_search, process_research]


# Define the ReAct prompt manually (No more Hub errors!)
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# 3. Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# ✅ Pull the prompt from LangChain Hub
prompt = PromptTemplate.from_template(template)

# Create the agent
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

print("\n -- Testing the Agent ---")
response = agent_executor.invoke({
    "input": "What is the current stock price of Reliance Industries and what is the latest news about Reliance Telecommunication (JIO)"
})
print(response["output"])
