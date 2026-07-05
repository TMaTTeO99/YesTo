from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import llm

class CompactTextSchema(BaseModel):
    compacted_text: str = Field(
        description="Testo compattato della conversazione tra l'utente e l'agente. Deve essere conciso, chiaro e mantenere il contesto essenziale.\n"
    )



_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei un Compattatore di Testo per un sistema agentico.\n"
        "Il tuo compito è prendere la conversazione tra l'utente e l'agente e compattarla in un testo conciso, chiaro e coerente.\n\n"
        "REGOLE:\n"
        "- Mantieni il contesto essenziale della conversazione.\n"
        "- Evita ripetizioni e dettagli non necessari.\n"
        "- Assicurati che il testo compattato sia facilmente comprensibile.\n"
    )),
    ("user", (
        "Ecco la conversazione da compattare: {conversation_text}\n"
    ))
])

compact_chain = _prompt | llm.with_structured_output(CompactTextSchema)
