from chains.executor_chain import web_executor_chain, db_tools_map, web_tools_map, observation_chain, db_executor_chain
from config import debug_print
from nodes.planning import _normalize_tool_result
from state import PlanningState, ToolsState
from chains.tools_supervisor import supervisor_chain
from config import debug_print

def supervisor_node(state: PlanningState):

    debug_print(f"🛠️ [SUPERVISOR] Nodo SUPERVISOR - Stato corrente: {state}")    
    supervisor_result = supervisor_chain.invoke({"original_text": state.get("original_text"), "plan": state.get("plan")})
    
    return {
        "plan" : state.get("plan", []),
        "past_steps": state.get("past_steps", []),
        "tool_agent": supervisor_result.selected_agent
    }

def tools_agent_routing_node(state: ToolsState):
    
    debug_print(f"🛠️ [TOOLS AGENT ROUTING] Nodo TOOLS AGENT ROUTING - Stato corrente: {state}")
    return state.get("tool_agent")
     
def exec_db_node(state: ToolsState):
    debug_print(f"🛠️ [TOOLS AGENT] Nodo EXEC DB - Stato corrente: {state}")
    return tool_executor(state, db_executor_chain, db_tools_map)

def exec_web_search_node(state: ToolsState):
    debug_print(f"🛠️ [TOOLS AGENT] Nodo EXEC WEB SEARCH - Stato corrente: {state}")
    return tool_executor(state, web_executor_chain, web_tools_map)

def tool_executor(state: ToolsState, chain, tools_map = None):
    
    current_plan = list(state.get("plan", []))
    if not current_plan:
        return {"plan": []}

    task = current_plan.pop(0)
    debug_print(f"🎯 [PLANNING] Nodo EXECUTION - Task attuale: '{task}'")

    past_steps = state.get("past_steps", [])
    context = "".join(f"- Task: {s['task']} -> Risultato: {s['details']}\n" for s in past_steps) or "Nessun task eseguito in precedenza."

    result = chain.invoke({"current_task": task, "past_steps_context": context})

    new_steps = list(past_steps)

    if hasattr(result, "tool_calls") and result.tool_calls:
        for call in result.tool_calls:
            name = call["name"]
            args = call.get("args", {})
            debug_print(f"   [LOG EXECUTOR] Chiamata tool: {name} con args {args}")

            if name in tools_map:

                raw = tools_map[name].invoke(args)
                normalized = _normalize_tool_result(name, raw)

                debug_print(f"[LOG EXECUTOR] Risultato '{name}': {normalized['details']}")
                
                obs = observation_chain.invoke({
                    "current_task": task,
                    "tool_result": normalized["details"],
                })
                debug_print(f"   [LOG EXECUTOR] Osservazione: {obs.content}")
                new_steps.append({"task": task, **normalized, "summary": obs.content})
            else:
                err = f"[ERROR]: tool '{name}' non riconosciuto."
                new_steps.append({"task": task, "tool_name": name, "receipt": err, "summary": err, "details": err})
    else:


        text = result.content if hasattr(result, "content") else str(result)
        debug_print(f"[LOG EXECUTOR] Nessuna tool call. Risposta diretta: {text}")
        new_steps.append({"task": task, "tool_name": "no_tool", "receipt": text, "summary": text, "details": text})
    
    debug_print("   [LOG EXECUTOR] Task completato.")
    return {"plan": current_plan, "past_steps": new_steps}