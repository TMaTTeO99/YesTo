from typing import List, Union, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from Shared.shared import llm

class PlanSchema(BaseModel):
    sotto_task: List[str] = Field(
        description="Lista ordinata di sotto-task sequenziali necessari per rispondere alla domanda dell'utente. Ogni task deve essere atomico e chiaro."
    )

class RePlanningSchema(BaseModel): 
    stop: bool = Field(
        description="True se l'obiettivo iniziale dell'utente è stato pienamente raggiunto o se non è possibile andare avanti. " \
                    "False se ci sono ancora task da eseguire o se il piano va aggiornato."
    )
    new_plan: Optional[List[str]] = Field(
        default=None,
        description = "Se 'finito' è False, inserisci qui la lista AGGIORNATA dei sotto-task rimanenti. Puoi mantenere i vecchi o cambiarli in base ai risultati ottenuti."
    )
    

# Forziamo il modello a rispondere usando rigorosamente lo schema JSON
structured_planner_llm = llm.with_structured_output(PlanSchema)

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei il Capo Progetto di un sistema agentico avanzato.\n"
        "Il tuo compito è prendere la richiesta macro di un utente e scomporla in un piano di lavoro "
        "composto da sotto-task sequenziali e cronologici.\n\n"
        "Linee guida per i sotto-task:\n"
        "- Devono essere specifici e focalizzati su una singola azione alla volta.\n"
        "- Devono essere ordinati in modo logico (es. non puoi chiedere di riassumere un testo prima di averlo cercato).\n"
        "- Evita task generici, scrivi azioni chiare.\n\n"
        "Genera l'output strutturato richiesto."
    )),
    ("user", "Ecco la richiesta dell'utente da scomporre in passaggi:\n\n{original_text}")
])

# 3. La Catena del Planner
planner_chain = planner_prompt | structured_planner_llm


structured_replanner_llm = llm.with_structured_output(RePlanningSchema)

replanner_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei il Direttore di Gara di un sistema agentico avanzato (Re-Planner).\n"
        "Il tuo compito è analizzare la richiesta ORIGINALE dell'utente, guardare il 'Diario di bordo' "
        "con le azioni già completate e decidere se l'obiettivo finale è stato REALMENTE raggiunto.\n\n"
        "REGOLE TASSATIVE DI LOGICA:\n"
        "1. L'obiettivo è raggiunto SOLO se l'azione finale richiesta (es. creare la tabella, inserire dati) "
        "   è stata eseguita con successo sul database. Se hai solo fatto verifiche preliminari (es. elenco_tabelle_db), "
        "   l'obiettivo NON è raggiunto. Imposta 'stop' a False.\n"
        "2. Se l'obiettivo NON è raggiunto, devi guardare i task che erano stati pianificati e "
        "   restituire in 'new_plan' la lista dei task che mancano ancora per completare l'opera, "
        "   rimuovendo SOLO il task che è appena stato completato con successo.\n"
        "3. Se un task è fallito o ha generato un errore, NON arrenderti. Modifica il 'new_plan' "
        "   inserendo un task correttivo o una strategia alternativa per aggirare l'errore.\n"
        "4. Imposta 'stop' a True SOLO quando l'operazione finale è stata confermata dal database. "
        "Genera l'output strutturato richiesto rispettando maniacalmente queste regole."
    )),
    ("user", (
        "🎯 RICHIESTA ORIGINALE UTENTE: {original_text}\n\n"
        "📊 DIARIO DI BORDO (Task già eseguiti e relativi risultati):\n"
        "{past_steps_context}"
    ))
])

replanner_chain = replanner_prompt | structured_replanner_llm