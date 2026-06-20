from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from state import AgentState, PlanningState

from nodes.planning import (
    planning_init_node, execution_node,
    replanner_node, merge_tools_output_node,
    planning_routing_logic,
)

def invoke_planning_agent(state: AgentState):
    result = planning_agent.invoke({
        "original_text": state["original_text"],
        "plan": [],
        "past_steps": [],
        "response": "",
    })
    return {
        "response": result["response"],
        "messages": [AIMessage(content=result["response"])],
    }

_builder = StateGraph(PlanningState)
_builder.add_node("planning_init", planning_init_node)
_builder.add_node("planning_execution", execution_node)
_builder.add_node("replanner", replanner_node)
_builder.add_node("merge_tools_output", merge_tools_output_node)

_builder.add_edge(START, "planning_init")
_builder.add_edge("planning_init", "planning_execution")
_builder.add_edge("planning_execution", "replanner")
_builder.add_conditional_edges("replanner", planning_routing_logic, {
    "continue_execution": "planning_execution",
    "go_to_final_response": "merge_tools_output",
})
_builder.add_edge("merge_tools_output", END)

planning_agent = _builder.compile()