from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from agents.executor_graph import invoke_execution_agent
from agents.utils import _get_history
from state import AgentState, PlanningState

from nodes.planning import (
    planning_init_node,
    replanner_node, merge_tools_output_node,
    planning_routing_logic,
    planner_checker,
    planner_checker_result_router
)

def invoke_planning_agent(state: AgentState):
    
    
    past_conversation = _get_history(state)
    result = planning_agent.invoke({
        "original_text": state["original_text"],
        "plan": [],
        "past_steps": [],
        "response": "",
        "past_conversation": past_conversation if past_conversation else ""
    })
    return {
        "response": result["response"],
        "messages": [AIMessage(content=result["response"])],
    }

_builder = StateGraph(PlanningState)
_builder.add_node("planning_init", planning_init_node)
_builder.add_node("tools_supervisor", invoke_execution_agent)
_builder.add_node("replanner", replanner_node)
_builder.add_node("merge_tools_output", merge_tools_output_node)
_builder.add_node("planner_checker", planner_checker)

_builder.add_edge(START, "planning_init")
_builder.add_edge("planning_init", "planner_checker")
_builder.add_edge("tools_supervisor", "replanner")

_builder.add_conditional_edges("replanner", planning_routing_logic, {
    "continue_execution": "tools_supervisor",
    "go_to_final_response": "merge_tools_output",
})

_builder.add_conditional_edges("planner_checker", planner_checker_result_router, {
    "good_plan" : "tools_supervisor",
    "bad_plan" : "replanner"                        
})


# TODO: here instead to go ahead without check ("go_to_final_response": "merge_tools_output")
# TODO: i need to check if the results are coherent with the user request and retry a few times
# TODO: .................
_builder.add_edge("merge_tools_output", END)

planning_agent = _builder.compile()