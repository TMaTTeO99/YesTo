from langgraph.graph import StateGraph, START, END
from state import PlanningState, ToolsState
from nodes.executor_nodes import (
    exec_db_node,
    exec_web_search_node,
    supervisor_node,
    tools_agent_routing_node,
    exec_rag_node
)


def invoke_execution_agent(state: PlanningState):
    result = execution_agent.invoke({
        "plan": state.get("plan", []),
        "past_steps": state.get("past_steps", []),
        "original_text": state.get("original_text", "")
    })
    return {
        "plan": result.get("plan", []),
        "past_steps": result.get("past_steps", []),
    }

_builder = StateGraph(ToolsState)
_builder.add_node("init_execution", supervisor_node)
_builder.add_node("exec_web_search_node",  exec_web_search_node)
_builder.add_node("exec_db_node", exec_db_node)
_builder.add_node("rag_agent", exec_rag_node)


_builder.add_edge(START, "init_execution")

_builder.add_edge("exec_web_search_node", END)
_builder.add_edge("exec_db_node", END)
_builder.add_edge("rag_agent", END)

_builder.add_conditional_edges("init_execution", tools_agent_routing_node, {
    "web_search_agent" : "exec_web_search_node", 
    "db_agent" : "exec_db_node",
    "rag_agent" : "rag_agent"
})


execution_agent = _builder.compile()