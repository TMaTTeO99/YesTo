from typing import Literal
from langchain_core.messages import AIMessage
from state import AgentState
from chains.qa_chain import qa_chain
from chains.critique_chain import critique_chain
from config import debug_print
from memory.vector_store import search_all, save_conversation
from nodes.node_utils import _build_chat_history

def domanda_node(state: AgentState):
    debug_print(f"🤖 [GRAFO] Nodo DOMANDA - Esecuzione (Tentativo {state.get('threshold', 0) + 1})")
    text = state["original_text"]

    if state.get("critique_punti"):
        text += f"\n\n⚠️ Correggi la risposta precedente seguendo queste indicazioni: {', '.join(state['critique_punti'])}"

    chat_history = _build_chat_history(state)
    debug_print(f"   [LOG MEMORIA] Chat history inviata al modello con {len(chat_history)} messaggi.")

    session_id = state.get("session_id", "")
    rag_context = search_all(text, session_id=session_id or None)
    if rag_context:
        rag_block = f"### Contesto recuperato dalla memoria/documenti (usa queste info se pertinenti):\n{rag_context}\n\n"
        debug_print(f"   [LOG RAG] Contesto recuperato ({len(rag_context)} chars).")
    else:
        rag_block = ""

    res = qa_chain.invoke({"original_text": text, "chat_history": chat_history, "rag_context": rag_block})
    debug_print(f"   [LOG DOMANDA] needs_tools={res.needs_tools}")

    if res.needs_tools:
        enriched = text
        if chat_history:
            context_lines = "\n".join([
                f"{'Utente' if m.type == 'human' else 'Assistente'}: {m.content}"
                for m in chat_history
                if m.content.strip() and m.content.strip() != "tools_needed"
            ])
            if context_lines:
                enriched = (
                    f"Contesto della conversazione precedente:\n{context_lines}\n\n"
                    f"Richiesta attuale dell'utente: {text}"
                )
                debug_print("   [LOG DOMANDA] original_text arricchito con chat history.")

        return {
            "messages": [AIMessage(content="tools_needed")],
            "original_text": enriched,
            "threshold": state.get("threshold", 0) + 1,
        }

    return {
        "messages": [AIMessage(content=res.answer)],
        "threshold": state.get("threshold", 0) + 1,
    }


def critique_node(state: AgentState):
    debug_print("🔎 [GRAFO] Nodo CRITICA DOMANDA")
    last_response = state["messages"][-1].content

    if last_response.strip() == "tools_needed":
        debug_print("   [LOG CRITICA] Rilevato 'tools_needed'. Approvazione automatica via Python (No LLM).")
        return {"critique_approvata": True, "critique_punti": []}

    res = critique_chain.invoke({"risposta": last_response, "original_text": state["original_text"]})
    approved = res.approvato
    points = res.punti_da_correggere

    if not approved and not points:
        debug_print("   ⚠️ [GUARDRAIL] Critico ha restituito 0 punti ma approvato=False. Forzo approvazione.")
        approved = True

    debug_print(f"   [LOG CRITICA] Approvato: {approved} | Note: {points}")
    return {"critique_approvata": approved, "critique_punti": points}


def clean_state_node(state: AgentState):
    debug_print("🧹 [GRAFO] Pulizia dello Stato completata con successo per il prossimo turno.")

    # Salva il turno corrente nel vector store per memoria futura
    last_ai = state["messages"][-1].content
    if last_ai.strip() not in ("tools_needed", ""):
        try:
            session_id = state.get("session_id", "")
            save_conversation(
                session_id=session_id or "default",
                human_msg=state.get("original_text", ""),
                ai_msg=last_ai,
            )
            debug_print("   [LOG RAG] Turno salvato nel vector store.")
        except Exception as e:
            debug_print(f"   [WARN RAG] Salvataggio fallito: {e}")

    return {
        "threshold": 0,
        "critique_approvata": False,
        "critique_punti": [],
        "plan": [],
        "past_steps": [],
        "response": "",
    }


def reflection_routing(state: AgentState) -> Literal["correggi_domanda", "clean_and_exit"]:
    if state.get("critique_approvata") is True or state.get("threshold", 0) >= 5:
        return "clean_and_exit"
    return "correggi_domanda"


def exit_or_plan_router(state: AgentState) -> Literal["go_to_end", "go_to_planning"]:
    last_message = state["messages"][-1].content
    if last_message.strip() == "tools_needed":
        debug_print("🔀 [GRAFO] Rilevato 'tools_needed'. Deviazione verso il Sotto-Grafo di Planning!")
        return "go_to_planning"
    return "go_to_end"
