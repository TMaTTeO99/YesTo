from langchain_core.messages import BaseMessage, SystemMessage
from state import AgentState
from chains.compact_text_chain import compact_chain
from chains.planner_chain import replanner_chain
from config import debug_print
from config import MAX_CHAT_HISTORY

def _build_chat_history(state: AgentState) -> list[BaseMessage]:

    all_messages = state.get("messages", [])
    if len(all_messages) > MAX_CHAT_HISTORY:

        conversation_text = "\n".join(f"{m.type} : {m.content}" for m in all_messages)
        result = compact_chain.invoke({
            "conversation_text": conversation_text,
        })
        return [SystemMessage(content=f"Riassunto della conversazione precedente: {result.compacted_text}")] + all_messages[-2:]
    
    return all_messages

def _call_replanner_node(state: AgentState, context: str):

    res = replanner_chain.invoke({"original_text": state["original_text"], "past_steps_context": context})

    if res.stop:
        debug_print("[LOG RE-PLANNER] LLM ha confermato di interrompere.")
        return {"plan": []}

    return {"plan": res.new_plan or [], "old_steps_summary": context}