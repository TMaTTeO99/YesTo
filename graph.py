from langgraph.graph import StateGraph, START, END
from state import AgentState
from nodes.router import router_node, routing_logic, incomprensibile_node
from nodes.qa import domanda_node, critique_node, clean_state_node, reflection_routing, exit_or_plan_router
from nodes.planning import (
    planning_init_node,
    execution_node,
    replanner_node,
    merge_tools_output_node,
    planning_routing_logic,
)

_builder = StateGraph(AgentState)

# --- Nodes ---
_builder.add_node("router", router_node)
_builder.add_node("domanda", domanda_node)
_builder.add_node("incomprensibile", incomprensibile_node)
_builder.add_node("critica_domanda", critique_node)
_builder.add_node("clean_and_exit", clean_state_node)
_builder.add_node("planning_init", planning_init_node)
_builder.add_node("planning_execution", execution_node)
_builder.add_node("replanner", replanner_node)
_builder.add_node("merge_tools_output", merge_tools_output_node)

# --- Edges ---
_builder.add_edge(START, "router")

_builder.add_conditional_edges("router", routing_logic, {
    "domanda": "domanda",
    "incomprensibile": "incomprensibile",
})

_builder.add_edge("incomprensibile", END)
_builder.add_edge("domanda", "critica_domanda")

_builder.add_conditional_edges("critica_domanda", reflection_routing, {
    "correggi_domanda": "domanda",
    "clean_and_exit": "clean_and_exit",
})

_builder.add_conditional_edges("clean_and_exit", exit_or_plan_router, {
    "go_to_planning": "planning_init",
    "go_to_end": END,
})

_builder.add_edge("planning_init", "planning_execution")
_builder.add_edge("planning_execution", "replanner")

_builder.add_conditional_edges("replanner", planning_routing_logic, {
    "continue_execution": "planning_execution",
    "go_to_final_response": "merge_tools_output",
})

_builder.add_edge("merge_tools_output", END)

workflow = _builder
