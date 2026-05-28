from typing import Annotated, Literal, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage

from Routing.StandardRouter import coordinator_router_chain
from Shared.shared import debug_print
from PromptChaining.Tasks.DomandaGenerica import parallel_question_chain
from PromptChaining.Critiques.CritiqueChain import critique_question_chain


class AgentState(TypedDict):
    original_text: str
    messages: Annotated[list[BaseMessage], add_messages]
    critique_approvata: bool
    critique_punti: list[str]
    threshold: int

    plan: List[str]                        
    past_steps: List[tuple[str, str]]      
    response: str

def routing_logic(state: AgentState) -> Literal["reclamo", "domanda", "incomprensibile"]:
    decision = coordinator_router_chain.invoke({"original_text": state["original_text"]}).strip().lower()
    debug_print(f"🔮 [GRAFO - ROUTER] Categoria rilevata: [{decision}]")
    if decision in ["reclamo", "domanda"]:
        return decision
    return "incomprensibile"

def reflection_routing_domanda(state: AgentState) -> Literal["correggi_domanda", "clean_and_exit"]:
    if state.get("critique_approvata") is True or state.get("threshold", 0) >= 5:
        return "clean_and_exit"
    return "correggi_domanda"

# 3. Nodi operativi
def router_node(state: AgentState):
    return {"original_text": state["original_text"], "threshold": state.get("threshold", 0)}

def domanda_node(state: AgentState):
    debug_print(f"🤖 [GRAFO] Nodo DOMANDA - Esecuzione (Tentativo {state.get('threshold', 0) + 1})")
    testo = state["original_text"]
    if state.get("critique_punti"):
        testo += f"\n\n⚠️ Correggi la risposta precedente seguendo queste indicazioni del supervisore: {', '.join(state['critique_punti'])}"
        
    res = parallel_question_chain.invoke({"original_text": testo, "chat_history": state["messages"]})
    return {"messages": [AIMessage(content=res)], "threshold": state.get("threshold", 0) + 1}

def incomprensibile_node(state: AgentState):
    return {"messages": [AIMessage(content="Mi dispiace, non ho capito la richiesta. Puoi riformulare?")]}

def critique_question_node(state: AgentState):
    debug_print("🔎 [GRAFO] Nodo CRITICA DOMANDA")
    
    last_response = state["messages"][-1].content

    if last_response.strip() == "tools_needed":
        debug_print("   [LOG CRITICA] Rilevato 'tools_needed'. Approvazione automatica via Python (No LLM).")
        return {
            "critique_approvata": True, 
            "critique_punti": []
        }

    critique_res = critique_question_chain.invoke({
        "risposta": last_response, 
        "original_text": state["original_text"]
    })
    
    approvato = critique_res.approvato
    punti = critique_res.punti_da_correggere

    if not approvato and (not punti or len(punti) == 0):
        debug_print("   ⚠️ [GUARDRAIL] Il critico ha restituito 0 punti da correggere ma approvato=False. Forzo l'approvazione a True.")
        approvato = True

    debug_print(f"   [LOG CRITICA] Risultato Finale -> Approvato: {approvato} | Note: {punti}")
    
    return {
        "critique_approvata": approvato, 
        "critique_punti": punti
    }

def clean_state_node(state: AgentState): 
    debug_print("🧹 [GRAFO] Pulizia dello Stato completata con successo per il prossimo turno.")
    return {
        "threshold" : 0,
        "critique_approvata": False,
        "critique_punti": []
    }