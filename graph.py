from langgraph.graph import StateGraph, START, END
from state import AgentState
from nodes.router import router_node, routing_logic, incomprensibile_node
from nodes.qa import domanda_node, critique_node, clean_state_node, reflection_routing, exit_or_plan_router
from agents.planning_graph import invoke_planning_agent

_builder = StateGraph(AgentState)

# --- Nodes ---
_builder.add_node("router", router_node)
_builder.add_node("domanda", domanda_node)
_builder.add_node("incomprensibile", incomprensibile_node)
_builder.add_node("critica_domanda", critique_node)
_builder.add_node("clean_and_exit", clean_state_node)
_builder.add_node("planning_agent", invoke_planning_agent)

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
    "go_to_planning": "planning_agent",
    "go_to_end": END,
})
_builder.add_edge("planning_agent", END)

workflow = _builder
