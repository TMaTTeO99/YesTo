from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm


class CritiqueSchema(BaseModel):
    approvato: bool = Field(
        description="True se i dati prodotti dall'agente forniscono realmente le informazioni richieste dall'utente. False altrimenti."
    )

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei un esperto analista di di testo. Il tuo compito è valutare se i dati prodotti dall'agente forniscono realmente le informazioni richieste dall'utente.\n"
        "Se i dati prodotti dall'agente sono sufficienti e rispondono alla richiesta dell'utente, imposta approvato=True. Se i dati prodotti dall'agente sono insufficienti o non rispondono alla richiesta dell'utente, imposta approvato=False.\n"
        "REGOLE:\n"
        "- I dati devono essere completi e accurati.\n"
        "- I dati devono essere rilevanti per la richiesta dell'utente.\n"
    )),
    ("user", (
        "Richiesta utente: {original_text}\n\n"
        "Dati prodotti dall'agente: {past_steps_context}\n\n"
        "Valuta se i dati prodotti dall'agente rispondono realmente alla richiesta dell'utente e imposta il campo 'approvato' di conseguenza."
    ))
])

critique_plan_result_chain = _prompt | llm.with_structured_output(CritiqueSchema)
