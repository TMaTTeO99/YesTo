from typing import Literal
from state import AgentState
from chains.router_chain import router_chain
from config import debug_print
from nodes.node_utils import _build_chat_history

def router_node(state: AgentState):
    return {"original_text": state["original_text"], "threshold": state.get("threshold", 0)}


def routing_logic(state: AgentState) -> Literal["domanda", "incomprensibile"]:
    result = router_chain.invoke({"original_text": state["original_text"], "message_summary" : "\n".join(f"{m.type} : {m.content}" for m in _build_chat_history(state))})
    decision = result.classification.value
    debug_print(f"🔮 [GRAFO - ROUTER] Categoria rilevata: [{decision}] | Motivazione: {result.justification}")

    if decision not in ("domanda", "incomprensibile"):
        debug_print(f"   ⚠️ [GUARDRAIL] Categoria '{decision}' non supportata. Forzo su 'domanda'.")
        return "domanda"

    return decision


def incomprensibile_node(state: AgentState):
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content="Mi dispiace, non ho capito la richiesta. Puoi riformulare?")]}
