from typing import Annotated, Literal, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage

from Routing.StandardRouter import coordinator_router_chain
from Shared.shared import debug_print
from PromptChaining.Tasks.DomandaGenerica import parallel_question_chain
from PromptChaining.Critiques.CritiqueChain import critique_question_chain
from PromptChaining.Tasks.SessionSummary import summary_chain


MAX_CHAT_BUFFER = 5


def render_messages_for_summary(messages: list[BaseMessage]) -> str:
    lines = []
    for msg in messages:
        role = msg.type if hasattr(msg, 'type') else 'message'
        lines.append(f"[{role}]: {msg.content}")
    return "\n".join(lines)


def summarize_history(existing_summary: str, messages_to_summarize: list[BaseMessage]) -> str:
    if not messages_to_summarize:
        return existing_summary or ""

    text_block = render_messages_for_summary(messages_to_summarize)
    summary_input = existing_summary or ""
    return summary_chain.invoke({
        "existing_summary": summary_input,
        "input_text": text_block
    })


def build_chat_history(state) -> list[BaseMessage]:

    all_messages = state.get("messages", [])
    
    if len(all_messages) <= 6:
        return all_messages
        
    recent_messages = all_messages[-6:]
    
    return recent_messages


def manage_history_node(state):
    all_messages = list(state.get("messages", []))
    existing_summary = state.get("message_summary", "")

    if len(all_messages) <= MAX_CHAT_BUFFER:
        return {
            "message_summary": existing_summary,
            "message_buffer": all_messages
        }

    messages_to_keep = all_messages[-MAX_CHAT_BUFFER:]
    messages_to_summarize = all_messages[:-MAX_CHAT_BUFFER]
    new_summary = summarize_history(existing_summary, messages_to_summarize)

    return {
        "message_summary": new_summary,
        "message_buffer": messages_to_keep
    }


class AgentState(TypedDict):
    original_text: str
    messages: Annotated[list[BaseMessage], add_messages]
    message_summary: str
    message_buffer: list[BaseMessage]
    critique_approvata: bool
    critique_punti: list[str]
    threshold: int

    plan: List[str]                        
    past_steps: List[dict]
    response: str

def routing_logic(state: AgentState) -> Literal["reclamo", "domanda", "saluto", "incomprensibile"]:
    router_result = coordinator_router_chain.invoke({"original_text": state["original_text"]})
    decision = router_result.classification.value
    debug_print(f"🔮 [GRAFO - ROUTER] Categoria rilevata: [{decision}] | Motivazione: {router_result.justification}")
    
    if decision not in ["domanda", "incomprensibile"]:
        debug_print(f"   ⚠️ [GUARDRAIL] Categoria '{decision}' non supportata dai nodi attuali. Forzo su 'domanda'.")
        return "domanda"
        
    return decision

def reflection_routing_domanda(state: AgentState) -> Literal["correggi_domanda", "clean_and_exit"]:
    if state.get("critique_approvata") is True or state.get("threshold", 0) >= 5:
        return "clean_and_exit"
    return "correggi_domanda"

# 3. Nodi operativi
def router_node(state: AgentState):
    return {"original_text": state["original_text"], "threshold": state.get("threshold", 0)}

def domanda_node(state: AgentState):
    debug_print(f"🤖 [GRAFO] Nodo DOMANDA - Esecuzione (Tentativo {state.get('threshold', 0) + 1})")
    
    testo_attuale = state["original_text"]
    
    if state.get("critique_punti"):
        testo_attuale += f"\n\n⚠️ Correggi la risposta precedente seguendo queste indicazioni: {', '.join(state['critique_punti'])}"
        
    chat_history = build_chat_history(state)
    
    debug_print(f"   [LOG MEMORIA] Chat history inviata al modello con {len(chat_history)} messaggi.")

    res = parallel_question_chain.invoke({
        "original_text": testo_attuale, 
        "chat_history": chat_history
    })
    
    return {
        "messages": [AIMessage(content=res)], 
        "threshold": state.get("threshold", 0) + 1
    }

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
        "critique_punti": [],
        
        "plan": [],
        "past_steps": [],
        "response": ""
    }