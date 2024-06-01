import getpass
import os

os.environ['OPENAI_API_KEY'] = 'sk-proj-s7fFmCJSnEJf9gBaQJuTT3BlbkFJn6EaTN3r4R1MLeH855Ld'
os.environ['LANGCHAIN_API_KEY'] = "ls__8f6eb48fea634a6d97aa95e4fd56e4d2"
os.environ['TAVILY_API_KEY'] = "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"


print("import")

from typing import Annotated, List, Tuple, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

print("import")
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
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

# agent supervisor
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

print("import002")

members = ["Researcher", "Analyst", "finance", "aggregator"]
system_prompt = (
    "You are leading a team of private equity analyst to generate investment report, below are your detailed workflows."
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. Review the work quality of the work first, ask the worker to re-perform the task if the quality satisfy the objective of the task. "
    "At the end of the work flow, you should always pass all outputs from other workers to the aggregator for him to generate the final report, "
    "you will continue to ask the aggregator to generate the report untill the result fits the requirment."
    "When finished,"
    " respond with FINISH."
)
print(system_prompt)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
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
            "Given the conversation above, who should act next?"
            f" Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4-turbo")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# building structure
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


research_agent = create_agent(llm, [tavily_tool], """You are a web researcher who is in charge of gathering all potential useful data and information from online.
You should always grab all the data that is useful!""")
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may write analysis for the target firm based on information gathered by Researcher.",
)
analyst_node = functools.partial(agent_node, agent=code_agent, name="Analyst")



finance_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate financial report based on information gathered by Researcher and Analyst.",
)
finance_node = functools.partial(agent_node, agent=finance_agent, name="finance")

agrre_agent = create_agent(
    llm,
    [python_repl_tool],
    """You may aggregate all the information from Researcher, Analyst, and finance to generate a comprehensive investment report, which is going to be AT LEAST 2000 WORDS LONG! 
    You should include all the information other workers collected. Most importantly, your final product will be the resport itself with no blank section.""",
)
agrre_node = functools.partial(agent_node, agent=agrre_agent, name="aggregator")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("finance", finance_node)
workflow.add_node("aggregator", agrre_node)
workflow.add_node("supervisor", supervisor_chain)
#workflow.add_node("finance", finance_node)
#workflow.add_node("aggregator", agrre_node)

#connect all nodes
for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("supervisor")

graph = workflow.compile()

user_prompt_tem = """WRITE YOUR PROMPT HERE"""

print(user_prompt_tem)

for s in graph.stream(
    {
        "messages": [
            HumanMessage(content=user_prompt_tem
            
            )
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")