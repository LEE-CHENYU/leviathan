import getpass
import os
from typing import Annotated, List, Tuple, Union, Any, Dict, Optional, Sequence, TypedDict
import functools
import operator

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

# Set environment variables for API keys
os.environ['OPENAI_API_KEY'] = 'sk-proj-s7fFmCJSnEJf9gBaQJuTT3BlbkFJn6EaTN3r4R1MLeH855Ld'
os.environ['LANGCHAIN_API_KEY'] = "ls__8f6eb48fea634a6d97aa95e4fd56e4d2"
os.environ['TAVILY_API_KEY'] = "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"

print("import")

# Initialize tools
tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool()  # This executes code locally, which can be unsafe

# Create agents
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Define members and system prompt
members = ["Researcher", "Analyst", "finance", "aggregator"]
system_prompt = (
    "You are leading a team of private equity analysts to generate an investment report. Below are your detailed workflows. "
    "You are a supervisor tasked with managing a conversation between the following workers: {members}. Given the following user request, "
    "respond with the worker to act next. Each worker will perform a task and respond with their results and status. Review the quality of the work first, ask the worker to re-perform the task if the quality does not satisfy the objective of the task. "
    "At the end of the workflow, you should always pass all outputs from other workers to the aggregator for him to generate the final report. "
    "You will continue to ask the aggregator to generate the report until the result fits the requirement. When finished, respond with FINISH."
)

# Supervisor chain setup
options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [{"enum": options}],
            }
        },
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4-turbo")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# Define agent states
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # The annotation tells the graph that new messages will always be added to the current states
    next: str  # The 'next' field indicates where to route to next

# Create agents for each role
research_agent = create_agent(llm, [tavily_tool], "You are a web researcher responsible for gathering all potentially useful data and information online.")
analyst_agent = create_agent(llm, [python_repl_tool], "You are an analyst writing an analysis based on information gathered by the researcher.")
finance_agent = create_agent(llm, [python_repl_tool], "You are a finance expert generating financial reports based on information from the researcher and analyst.")
aggregator_agent = create_agent(llm, [python_repl_tool], "You are an aggregator combining all the information from the researcher, analyst, and finance to generate a comprehensive investment report of at least 2000 words.")

# Define agent nodes
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")
analyst_node = functools.partial(agent_node, agent=analyst_agent, name="Analyst")
finance_node = functools.partial(agent_node, agent=finance_agent, name="finance")
aggregator_node = functools.partial(agent_node, agent=aggregator_agent, name="aggregator")

# Define workflow
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("finance", finance_node)
workflow.add_node("aggregator", aggregator_node)
workflow.add_node("supervisor", supervisor_chain)

# Connect all nodes
for member in members:
    workflow.add_edge(member, "supervisor")
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()

user_prompt_tem = "WRITE YOUR PROMPT HERE"

print(user_prompt_tem)

# Execute the workflow
for s in graph.stream({"messages": [HumanMessage(content=user_prompt_tem)]}):
    if "__end__" not in s:
        print(s)
        print("----")
  
rule_of_the_game = {
  "description": "This action board represents the sequence of actions taken by different players in the same round of the game.",
  "rules": {
    "ordering": "Agents take turns making decisions in a specified order.",
    "action_limit": "Each agent may perform only one action per round from the available types {offer, attack, reproduce, offer land}.",
    "decision_loop": "The decision-making loop continues until each agent has either taken an action or passed (given up) up to three times.",
    "initial_decision": "Each agent initially decides whether to take an action or pass for that round.",
    "follow_or_lead": "Agents decide whether to react to the actions already taken on the board (follow) or initiate a new action chain (lead).",
    "reaction": "Later agents in the turn order make decisions based on earlier actions in the round, choosing to cooperate or counter the earlier decisions.",
    "action_choice": "Agents specify the action they will take, detailing the type of action and the targets involved."
  }
}

rule_of_the_decison = {
  "initial_decision": "Decide whether to take action or pass.",
  "action_flow": "Choose whether to follow an existing action chain or initiate a new action sequence.",
  "chain_selection": "Specify the chain number to follow, or indicate 'new' for starting a new sequence.",
  "action_details": "Select the specific action to take, including action type and target agent."
}