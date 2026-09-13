from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import llm

class CompactTextSchema(BaseModel):
    compacted_text: str = Field(
        description="Compacted text of the conversation between the user and the agent. Must be concise, clear, and preserve the essential context.\n"
    )



_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a Text Compactor for an agentic system.\n"
        "Your job is to take the conversation between the user and the agent and compact it into concise, clear, coherent text.\n\n"
        "RULES:\n"
        "- Preserve the essential context of the conversation.\n"
        "- Avoid repetition and unnecessary details.\n"
        "- Make sure the compacted text is easy to understand.\n"
    )),
    ("user", (
        "Here is the conversation to compact: {conversation_text}\n"
    ))
])

compact_chain = _prompt | llm.with_structured_output(CompactTextSchema)
