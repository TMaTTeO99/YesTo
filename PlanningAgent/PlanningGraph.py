from typing import Literal
from langchain_core.messages import AIMessage
from Routing.GraphRouter import AgentState 
from PlanningAgent.Planner import planner_chain, replanner_chain
from PlanningAgent.Executor import executor_chain

from Tools.Tools import Find_table_info, elenco_tabelle_db, Create_table

tools_map = {
    "elenco_tabelle_db" : elenco_tabelle_db, 
    "Create_table" : Create_table,
    "Find_table_info": Find_table_info
}

# ==========================================
# 1. I NODI DEL SOTTO-FLUSSO DI PLANNING
# ==========================================

def planning_init_node(state: AgentState):
    """Genera il piano di lavoro iniziale basandosi sulla richiesta dell'utente."""
    print("📋 [PLANNING] Nodo INIT - Generazione del piano di lavoro...")
    
    # Invochiamo la catena del planner sul testo originale
    planner_res = planner_chain.invoke({"original_text": state["original_text"]})
    print(f"   [LOG PLANNER] Task pianificati: {planner_res.sotto_task}")
    
    # Salviamo i task nel piano e inizializziamo il diario di bordo vuoto
    return {
        "plan": planner_res.sotto_task,
        "past_steps": [],
        "response": "" 
    }


def execution_node(state: AgentState):
    """Prende il primo task della lista, lo esegue e lo rimuove dal piano corrente."""
    current_plan = list(state.get("plan", []))
    if not current_plan:
        return {}
    
    # 1. Estraiamo il task attuale (pop in testa)
    task_da_fare = current_plan.pop(0)
    print(f"🎯 [PLANNING] Nodo EXECUTION - Task attuale: '{task_da_fare}'")
    
    past_steps = state.get("past_steps", [])
    context_str = "".join([f"- Task: {task} -> Risultato: {ris}\n" for task, ris in past_steps])
    if not context_str:
        context_str = "Nessun task eseguito in precedenza."
        
    # Invocazione Executor
    risultato_task = executor_chain.invoke({
        "current_task": task_da_fare,
        "past_steps_context": context_str
    })

    if risultato_task.tool_calls:
        result_function = []
        for tool_call in risultato_task.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            print(f"   [LOG EXECUTOR] Rilevato chiamata a tool: {tool_name} con argomenti {tool_args}")
            
            if tool_name in tools_map:
                tool_func = tools_map[tool_name]
                tool_result = tool_func.invoke(tool_args)
                result_function.append(f"[{tool_name}]: {tool_result}")
            else:
                result_function.append(f"[ERROR]: tool richiesto '{tool_name}' non riconosciuto.")
        risultato_task = "\n".join(result_function)
    else:
        risultato_task = risultato_task.content

    print(f"   [LOG EXECUTOR] Task completato con successo.")

    nuovi_passi = list(past_steps)
    nuovi_passi.append((task_da_fare, risultato_task))
    
    # 2. Restituiamo il piano aggiornato (senza il task appena eseguito) e la cronologia
    return {
        "plan": current_plan,
        "past_steps": nuovi_passi
    }

def replanner_node(state: AgentState):
    print("🧠 [PLANNING] Nodo RE-PLANNER - Valutazione dello stato del piano...")
    
    current_plan = state.get("plan", [])
    past_step = state.get("past_steps", [])
    
    # Se ci sono ancora task nel piano e l'ultimo NON è un errore, andiamo avanti spediti
    if current_plan and "ERROR" not in str(past_step[-1][1]):
        print(f"   [LOG RE-PLANNER] Tutto procede secondo i piani. Task rimanenti nel buffer: {len(current_plan)}")
        return {} # Non sovrascrive il piano, lascia che continui l'esecuzione

    # Altrimenti (piano vuoto o presenza di errori), interpelliamo l'LLM per decidere il da farsi
    context_str = "".join([f"Task: {step}\nRisultato:{result}\n" for step, result in past_step])

    replanner_res = replanner_chain.invoke({
        "original_text": state["original_text"],
        "past_steps_context" : context_str
    })

    if replanner_res.stop or not current_plan:
        print("   [LOG RE-PLANNER] Obiettivo raggiunto! Confezionamento risposta finale.")
        risposta_da_inviare = replanner_res.final_answer if replanner_res.final_answer else "Ho completato le analisi richieste."
        return {
            "plan": [],                        
            "response": risposta_da_inviare,
            "messages": [AIMessage(content=risposta_da_inviare)] 
        }
    else:
        print(f"   [LOG RE-PLANNER] Rilevato imprevisto. Piano aggiornato dall'LLM: {replanner_res.new_plan}")
        return {
            "plan" : replanner_res.new_plan
        }


# ==========================================
# 2. LOGICA DI ROUTING (IL BIVIO)
# ==========================================

def planning_routing_logic(state: AgentState) -> Literal["continue_execution", "go_to_final_response"]:
    """Controlla se ci sono ancora task nel piano."""
    if state.get("plan") : 
        return "continue_execution"

    return "go_to_final_response"

def exit_or_plan_router(state: AgentState) -> Literal["go_to_end", "go_to_planning"]:

    last_message = state["messages"][-1].content

    if "tools_needed" in last_message : 
        print("🔀 [GRAFO] Rilevato 'tools_needed'. Deviazione verso il Sotto-Grafo di Planning!")
        return "go_to_planning" 
    
    print("🛑 [GRAFO] Risposta standard completata. Uscita normale.")
    return "go_to_end"
