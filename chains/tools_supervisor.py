from langchain_core.prompts import ChatPromptTemplate
from config import llm
from pydantic import BaseModel, Field
from typing import Literal

class ToolsSupervisorSchema(BaseModel):

    selected_agent: Literal["web_search_agent", "db_agent"] = Field(
        description= "Il sotto agente selezionato per eseguire la richiesta dell'utente in base al contesto della conversazione e alla richiesta dell'utente.\n" \
        "Devi rispondere esclusivamente con il nome del sotto agente selezionato, senza aggiungere ulteriori spiegazioni o commenti.\n" \
        "Sotto agenti disponibili:\n" \
        "web_search_agent: Se la richiesta dell'utente richiede di cercare informazioni sul web.\n" \
        "db_agent: Se la richiesta dell'utente richiede di accedere a dati aziendali o tabelle.\n" 
    )

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei un agente supervisore specializzato nel scegliere quale sotto agente eseguire in base alla richiesta dell'utente e al piano proposto dall'agente di pianificazione.\n"
    )),
    ("user", (
        "Richiesta attuale dell'utente: {original_text}\n\n"
        "Contesto della conversazione precedente: {plan}\n\n"
    ))
])

supervisor_chain = _prompt | llm.with_structured_output(ToolsSupervisorSchema)