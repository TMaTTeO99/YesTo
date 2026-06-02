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

        "REGOLE DI VALUTAZIONE:\n"
        "1. Messaggi di cortesia, saluti o chiacchiere -> approvato=True sempre.\n"
        "2. Risposte di conoscenza generale statica (capitali, definizioni, storia...) -> "
        "   approvato=True se la risposta è corretta, False se contiene errori fattuali.\n"
        "3. ATTENZIONE ALLE ALLUCINAZIONI SUL DATABASE: se la richiesta riguarda dati aziendali, "
        "   tabelle, ordini, clienti, listini o qualsiasi dato operativo, e l'agente fornisce "
        "   dati specifici SENZA che questi provengano da uno strumento reale (tool), "
        "   quella è un'allucinazione -> approvato=False, segnalalo in punti_da_correggere.\n"
        "4. REGOLA DI COERENZA: se approvato=False devi sempre fornire almeno un punto in "
        "   punti_da_correggere. Se non riesci a trovare errori concreti, imposta approvato=True.\n"
    )),
    ("user", "Richiesta utente: {original_text}\n\nRisposta agente: {risposta}")
])
critique_question_chain = critique_question_prompt | structured_critic_llm