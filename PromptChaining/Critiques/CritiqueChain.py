from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from Shared.shared import llm

# Definizione dello schema di output strutturato
class CritiqueSchema(BaseModel):
    approvato: bool = Field(description="True se la risposta non ha bisogno di modifiche, False altrimenti.")
    punti_da_correggere: List[str] = Field(description="Elenco dettagliato dei punti deboli da correggere. Lasciare vuoto se approvato.")

# Forziamo il modello (Gemma o Gemini) a rispondere SOLO ed ESATTAMENTE con lo schema JSON
structured_critic_llm = llm.with_structured_output(CritiqueSchema)

# CRITICO DOMANDE
critique_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei un agente specializzato alle controllo delle risposte alle domande dell'utente.\n"
        "Se la risposta contiene: 'tools_needed' approva direttamente la risposta perche per servire la risposta servono tools esterni altrimenti"
        "valuta se la risposta risponde al quesito dell'utente e soprattuto che NON sia nulla di inventato.\n\n"
        "Genera l'output strutturato richiesto."
    )),
    ("user", "Ecco il testo originale: {original_text}.\n\nEcco la risposta da valutare:\n\n{risposta}")
])
critique_question_chain = critique_question_prompt | structured_critic_llm