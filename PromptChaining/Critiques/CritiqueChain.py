from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from Shared.shared import llm

class CritiqueSchema(BaseModel):
    approvato: bool = Field(
        description="TRUE se la risposta è corretta e risolve il quesito. FALSE se contiene errori o allucinazioni."
    )
    punti_da_correggere: List[str] = Field(
        description="Se approvato è TRUE, questa lista DEVE essere tassativamente vuota []. "
                    "Se approvato è FALSE, inserisci qui l'elenco dei punti deboli riscontrati."
    )

structured_critic_llm = llm.with_structured_output(CritiqueSchema)

critique_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei l'Ispettore Controllo Qualità del sistema (Critico).\n"
        "Il tuo unico compito è validare se la risposta dell'agente è corretta e non inventata.\n\n"
        "REGOLA DI COERENZA TASSATIVA:\n"
        "- Se non trovi errori e la risposta va bene: imposta approvato = True e punti_da_correggere = [].\n"
        "- Se trovi errori o mancanze: imposta approvato = False e descrivi i problemi in punti_da_correggere.\n\n"
        "Nota: I messaggi di cortesia, saluti o presentazioni (es. 'Ciao, come va?') sono risposte VALIDE. Approvale sempre con True."
    )),
    ("user", "Richiesta utente: {original_text}\n\nRisposta agente: {risposta}")
])
critique_question_chain = critique_question_prompt | structured_critic_llm