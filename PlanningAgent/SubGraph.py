import json
from typing import Literal
from langchain_core.messages import AIMessage
from Routing.GraphRouter import AgentState 
from PromptChaining.Tasks.Planner import planner_chain, replanner_chain
from PromptChaining.Tasks.Executor import executor_chain
from PromptChaining.Tasks.SessionSummary import summary_chain
from Shared.shared import debug_print
from PromptChaining.Tasks.ToolsOutput import final_tools_output_chain

from Tools.Tools import Find_table_info, elenco_tabelle_db, Create_table, Cerca_su_Web

tools_map = {
    "elenco_tabelle_db" : elenco_tabelle_db, 
    "Create_table" : Create_table,
    "Find_table_info": Find_table_info,
    "Cerca_su_Web": Cerca_su_Web
}

MAX_PAST_STEP_BUFFER = 4


def _compact_past_steps(past_steps: list[dict]) -> list[dict]:
    if len(past_steps) <= MAX_PAST_STEP_BUFFER:
        return past_steps

    recent_steps = past_steps[-MAX_PAST_STEP_BUFFER:]
    old_steps = past_steps[:-MAX_PAST_STEP_BUFFER]
    compact_text = "\n".join([
        f"Task: {step['task']}\nRicevuta: {step['receipt']}" for step in old_steps
    ])
    summarized_text = summary_chain.invoke({
        "existing_summary": "",
        "input_text": compact_text
    })

    summary_entry = {
        "task": "Passi precedenti compressi",
        "receipt": f"Storico compresso: {summarized_text}",
        "summary": summarized_text,
        "details": summarized_text
    }
    return [summary_entry] + recent_steps


def _build_replanner_context(past_steps: list[dict]) -> str:
    compact_steps = _compact_past_steps(past_steps)
    return "\n".join([
        f"Task: {step['task']}\nRicevuta: {step['receipt']}" for step in compact_steps
    ])


def _build_final_context(past_steps: list[dict]) -> str:
    return "\n---\n".join([
        f"Operazione pianificata: {step['task']}\nSintesi: {step['summary']}\nDettagli: {step['details']}" for step in past_steps
    ])


def _normalize_tool_result(tool_name: str, tool_result) -> dict:
    try:
        data = json.loads(str(tool_result))
        if isinstance(data, dict):
            return {
                "tool_name": tool_name,
                "receipt": str(data.get("receipt", "")),
                "summary": str(data.get("summary", "")),
                "details": str(data.get("details", ""))
            }
    except Exception:
        pass

    text = str(tool_result)
    return {
        "tool_name": tool_name,
        "receipt": text,
        "summary": text,
        "details": text
    }


def planning_init_node(state: AgentState):
    """Genera il piano di lavoro iniziale basandosi sulla richiesta dell'utente."""
    debug_print("📋 [PLANNING] Nodo INIT - Generazione del piano di lavoro...")
    
    planner_res = planner_chain.invoke({"original_text": state["original_text"]})
    debug_print(f"   [LOG PLANNER] Task pianificati: {planner_res.sotto_task}")
    
    return {
        "plan": planner_res.sotto_task,
        "past_steps": [],
        "response": "" 
    }


def execution_node(state: AgentState):
    """Prende il primo task della lista, lo esegue e lo rimuove dal piano corrente."""
    current_plan = list(state.get("plan", []))
    if not current_plan:
        return {"plan": []} 
    
    task_da_fare = current_plan.pop(0)
    debug_print(f"🎯 [PLANNING] Nodo EXECUTION - Task attuale: '{task_da_fare}'")
    
    past_steps = state.get("past_steps", [])
    context_str = "".join([f"- Task: {step['task']} -> Ricevuta: {step['receipt']}\n" for step in past_steps])
    if not context_str:
        context_str = "Nessun task eseguito in precedenza."
        
    risultato_task = executor_chain.invoke({
        "current_task": task_da_fare,
        "past_steps_context": context_str
    })

    if hasattr(risultato_task, 'tool_calls') and risultato_task.tool_calls:
        result_function = []
        nuovi_passi = list(past_steps)
        for tool_call in risultato_task.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            debug_print(f"   [LOG EXECUTOR] Rilevato chiamata a tool: {tool_name} con argomenti {tool_args}")
            
            if tool_name in tools_map:
                tool_func = tools_map[tool_name]
                tool_result = tool_func.invoke(tool_args)
                normalized = _normalize_tool_result(tool_name, tool_result)
                debug_print(f"   [LOG EXECUTOR] Risultato del tool '{tool_name}': {normalized['receipt']}")
                result_function.append(f"[{tool_name}]: {normalized['receipt']}")
                nuovi_passi.append({
                    "task": task_da_fare,
                    "tool_name": tool_name,
                    "receipt": normalized['receipt'],
                    "summary": normalized['summary'],
                    "details": normalized['details']
                })
            else:
                error_entry = f"[ERROR]: tool richiesto '{tool_name}' non riconosciuto."
                result_function.append(error_entry)
                nuovi_passi.append({
                    "task": task_da_fare,
                    "tool_name": tool_name,
                    "receipt": error_entry,
                    "summary": error_entry,
                    "details": error_entry
                })
        risultato_output = "\n".join(result_function)
    else:
        risultato_output = risultato_task.content if hasattr(risultato_task, 'content') else str(risultato_task)
        nuovi_passi = list(past_steps)
        nuovi_passi.append({
            "task": task_da_fare,
            "tool_name": "no_tool",
            "receipt": risultato_output,
            "summary": risultato_output,
            "details": risultato_output
        })

    debug_print(f"   [LOG EXECUTOR] Task completato.")

    return {
        "plan": current_plan,
        "past_steps": nuovi_passi
    }

def replanner_node(state: AgentState):
    debug_print("🧠 [PLANNING] Nodo RE-PLANNER - Valutazione dello stato del piano...")
    
    current_plan = list(state.get("plan", []))
    past_steps = state.get("past_steps", [])
    
    if not past_steps:
        return {}

    ultimo_risultato = str(past_steps[-1]["receipt"])
    debug_print(f"TEST: ************* ultimo_risultato: {ultimo_risultato}")

    if "[ERROR]" not in ultimo_risultato and len(current_plan) == 0:
        debug_print("   🛑 [GUARDRAIL PYTHON] Obiettivo raggiunto con successo e piano esaurito. Forzo l'uscita.")
        return {"plan": []}

    if "[ERROR]" not in ultimo_risultato and len(current_plan) > 0:
        debug_print(f"   [LOG RE-PLANNER] Tutto procede bene. Task rimanenti nel buffer: {len(current_plan)}")
        return {}

    debug_print("   ⚠️ [LOG RE-PLANNER] Rilevato un [ERROR] reale nell'ultimo task. Interpello l'LLM per ri-pianificare...")
    
    context_str = _build_replanner_context(past_steps)

    replanner_res = replanner_chain.invoke({
        "original_text": state["original_text"],
        "past_steps_context": context_str
    })

    if replanner_res.stop:
        debug_print("   [LOG RE-PLANNER] L'LLM ha confermato di interrompere o che l'obiettivo è stato raggiunto.")
        return {"plan": []}

    new_plan = replanner_res.new_plan or []
    debug_print(f"   [LOG RE-PLANNER] Nuovo piano iniettato dall'LLM per recuperare l'errore: {new_plan}")
    return {"plan": new_plan}
    

def merge_tools_output_node(state: AgentState):
    debug_print("🎯 [PLANNING] Nodo MERGE - Confezionamento risposta finale...")
    past_steps = state.get("past_steps", [])
    
    if not past_steps:
        msg = "Mi dispiace, si è verificato un problema durante l'esecuzione delle operazioni sul database."
        return {"response": msg, "messages": [AIMessage(content=msg)]}
    
    # Costruiamo il contesto delle operazioni per il compilatore finale.
    # Qui usiamo le summary/dettagli strutturati restituiti dai tool.
    context_str = _build_final_context(past_steps)
    
    # Passiamo sia gli esiti dei tool che la richiesta originaria!
    final_response = final_tools_output_chain.invoke({
        "tools_outputs": context_str,
        "original_text": state["original_text"]
    })
    
    debug_print(f"   [LOG MERGE] Risposta completata con successo.")
    return {
        "plan": [],
        "response": final_response,
        "messages": [AIMessage(content=final_response)]
    }



def planning_routing_logic(state: AgentState) -> Literal["continue_execution", "go_to_final_response"]:
    """Controlla in modo rigoroso se ci sono elementi residui nel piano."""
    plan = state.get("plan", [])
    if len(plan) > 0: 
        return "continue_execution"
    return "go_to_final_response"


def exit_or_plan_router(state: AgentState) -> Literal["go_to_end", "go_to_planning"]:
    last_message = state["messages"][-1].content

    if last_message.strip() == "tools_needed":
        debug_print("🔀 [GRAFO] Rilevato 'tools_needed'. Deviazione verso il Sotto-Grafo di Planning!")
        return "go_to_planning"
    
    return "go_to_end"