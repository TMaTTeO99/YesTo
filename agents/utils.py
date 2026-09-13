from nodes.node_utils import _build_chat_history
from state import AgentState


def _get_history(state: AgentState):
    
    chat_history = _build_chat_history(state)
    past_conversation = ""
    if chat_history:
        past_conversation = "\n".join([
        
            f"{'User' if m.type == 'human' else 'Agent'}: {m.content}"
            for m in chat_history
                if m.content.strip() and m.content.strip() != "tools_needed"
        ])

    return past_conversation