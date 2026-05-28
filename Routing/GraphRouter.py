from langgraph.graph import StateGraph, START, END
from Routing.Graph import reflection_routing_domanda
from Routing.Graph import (
    AgentState,
    router_node,
    domanda_node, 
    incomprensibile_node, 
    critique_question_node, 
    routing_logic, 
    clean_state_node
)

from PlanningAgent.SubGraph import (
    merge_tools_output_node,
    planning_init_node, 
    execution_node, 
    planning_routing_logic,
    exit_or_plan_router,
    replanner_node
)

# 5. Costruzione del Grafo
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("domanda", domanda_node)
workflow.add_node("incomprensibile", incomprensibile_node)
workflow.add_node("critica_domanda", critique_question_node)
workflow.add_node("clean_and_exit", clean_state_node)
workflow.add_node("planning_init", planning_init_node)
workflow.add_node("planning_execution", execution_node)
workflow.add_node("replanner", replanner_node)
workflow.add_node("merge_tools_output", merge_tools_output_node)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    routing_logic,
    {
        "domanda": "domanda",
        "incomprensibile": "incomprensibile"
    }
)

workflow.add_edge("incomprensibile", END)

# Flusso Domanda -> Critica -> Loop Condizionale
workflow.add_edge("domanda", "critica_domanda")
workflow.add_conditional_edges(
    "critica_domanda",
    reflection_routing_domanda,
    {
        "correggi_domanda": "domanda",
        "clean_and_exit": "clean_and_exit"
    }
)

workflow.add_conditional_edges(
    "clean_and_exit",
    exit_or_plan_router,
    {
        "go_to_planning" : "planning_init",
        "go_to_end" : END 
    }
)

workflow.add_edge("planning_init", "planning_execution")
workflow.add_edge("planning_execution", "replanner")

workflow.add_conditional_edges(
    "replanner",
    planning_routing_logic,
    {
        "continue_execution": "planning_execution",
        "go_to_final_response": "merge_tools_output"
    }
)

workflow.add_edge("merge_tools_output", END)

uncompiled_workflow = workflow