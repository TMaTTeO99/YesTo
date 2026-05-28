import json
from typing import Literal
from langchain_core.messages import AIMessage
from Routing.GraphRouter import AgentState 
from PromptChaining.Tasks.Planner import planner_chain, replanner_chain
from PromptChaining.Tasks.Executor import executor_chain
from Shared.shared import debug_print
from PromptChaining.Tasks.ToolsOutput import final_tools_output_chain

from Tools.Tools import Find_table_info, elenco_tabelle_db, Create_table

tools_map = {
    "elenco_tabelle_db" : elenco_tabelle_db, 
    "Create_table" : Create_table,
    "Find_table_info": Find_table_info
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
    context_str = "".join([f"- Task: {task} -> Risultato: {ris}\n" for task, ris in past_steps])
    if not context_str:
        context_str = "Nessun task eseguito in precedenza."
        
    risultato_task = executor_chain.invoke({
        "current_task": task_da_fare,
        "past_steps_context": context_str
    })

    if hasattr(risultato_task, 'tool_calls') and risultato_task.tool_calls:
        result_function = []
        for tool_call in risultato_task.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            debug_print(f"   [LOG EXECUTOR] Rilevato chiamata a tool: {tool_name} con argomenti {tool_args}")
            
            if tool_name in tools_map:
                tool_func = tools_map[tool_name]
                tool_result = tool_func.invoke(tool_args)
                result_function.append(f"[{tool_name}]: {tool_result}")
            else:
                result_function.append(f"[ERROR]: tool richiesto '{tool_name}' non riconosciuto.")
        risultato_output = "\n".join(result_function)
    else:
        risultato_output = risultato_task.content if hasattr(risultato_task, 'content') else str(risultato_task)

    debug_print(f"   [LOG EXECUTOR] Task completato.")

    nuovi_passi = list(past_steps)
    nuovi_passi.append((task_da_fare, risultato_output))
    
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

    ultimo_risultato = str(past_steps[-1][1])
    
    if "[ERROR]" not in ultimo_risultato and len(current_plan) > 0:
        debug_print(f"   [LOG RE-PLANNER] Tutto procede bene. Rimanenti: {len(current_plan)}")
        return {}

    if "[ERROR]" not in ultimo_risultato and len(current_plan) == 0:
        debug_print("   [LOG RE-PLANNER] Tutti i task eseguiti con successo. Procedo al confezionamento.")
        return {"plan": []} 

    debug_print("   ⚠️ [LOG RE-PLANNER] Rilevato errore o anomalia nell'ultimo task. Interpello l'LLM per ri-pianificare...")
    context_str = "".join([f"Task: {task}\nRisultato: {res}\n" for task, res in past_steps])

    replanner_res = replanner_chain.invoke({
        "original_text": state["original_text"],
        "past_steps_context" : context_str
    })

    if replanner_res.stop:
        debug_print("   [LOG RE-PLANNER] L'LLM ha deciso di interrompere e generare la risposta con i dati disponibili.")
        return {"plan": []} 
    else:
        debug_print(f"   [LOG RE-PLANNER] Nuovo piano iniettato dall'LLM per recuperare l'errore: {replanner_res.new_plan}")
        return {"plan": replanner_res.new_plan}
    

def merge_tools_output_node(state: AgentState):
    debug_print("🎯 [PLANNING] Nodo MERGE - Confezionamento risposta finale...")
    past_steps = state.get("past_steps", [])
    
    if not past_steps:
        msg = "Mi dispiace, si è verificato un problema durante l'esecuzione delle operazioni sul database."
        return {"response": msg, "messages": [AIMessage(content=msg)]}
    
    # Costruiamo il contesto delle operazioni per il compilatore finale
    context_str = "".join([f"Operazione pianificata: {task}\nEsito dell'azione: {res}\n---\n" for task, res in past_steps])
    
    # Passiamo sia gli esiti dei tool che la richiesta originaria!
    final_response = final_tools_output_chain.invoke({
        "tools_outputs": context_str,
        "original_text": state["original_text"] # Risolve la mancanza di contesto iniziale
    })
    
    debug_print(f"   [LOG MERGE] Risposta completata con successo.")
    return {
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

    if "tools_needed" in last_message: 
        debug_print("🔀 [GRAFO] Rilevato 'tools_needed'. Deviazione verso il Sotto-Grafo di Planning!")
        return "go_to_planning" 
    
    return "go_to_end"