from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm


class PlanSchema(BaseModel):
    
    ragionamento: str = Field(
        description="Ragionamento step-by-step prima di generare il piano: analisi della richiesta, strumenti rilevanti, numero minimo di task."
    )
    sotto_task: List[str] = Field(
        description="Lista ordinata di sotto-task sequenziali necessari per rispondere alla domanda dell'utente. Ogni task deve essere atomico e chiaro."
    )
    


class RePlanningSchema(BaseModel):
    stop: bool = Field(
        description="True se l'errore è insuperabile e non è possibile procedere in nessun modo. "
                    "False nella quasi totalità dei casi: devi sempre tentare un piano correttivo."
    )
    new_plan: Optional[List[str]] = Field(
        default=None,
        description="Lista AGGIORNATA dei sotto-task rimanenti per recuperare dall'errore. "
                    "Deve contenere un task correttivo o alternativo che risolva il problema riscontrato."
    )


_planner_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei il Capo Progetto di un sistema agentico avanzato.\n"
        "Il tuo compito è scomporre la richiesta dell'utente nel MINIMO numero di sotto-task "
        "strettamente necessari per rispondere, senza aggiungere passaggi accessori o di contorno.\n\n"
        "REGOLE TASSATIVE:\n"
        "1. MINIMALISMO: pianifica solo le azioni indispensabili. Se una singola ricerca web o "
        "   una singola query DB è sufficiente, il piano ha UN solo task.\n"
        "2. Non aggiungere task di verifica, raccolta dettagli aggiuntivi, o controlli "
        "   se non esplicitamente richiesti dall'utente.\n"
        "3. Ogni task deve essere atomico e indipendente: se due azioni possono essere "
        "   fuse in una sola query/ricerca, accorpale.\n"
        "REGOLE CoT:\n"
        "Prima di generare il piano, ragiona ad alta voce:\n"\
        "- Cosa sta chiedendo esattamente l'utente?\n" \
        "- Quali strumenti disponibili sono rilevanti?\n" \
        "- Qual è il numero MINIMO di task necessari?\n\n" \
        "Genera l'output strutturato rispettando rigorosamente queste regole."
    )),
    ("user", (
        "Richiesta dell'utente:\n\n{original_text}"
    ))
])

_replanner_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei il Gestore degli Errori di un sistema agentico avanzato (Re-Planner).\n"
        "Vieni chiamato SOLO quando un task ha generato un errore. Il tuo unico compito è "
        "produrre il piano correttivo MINIMO per riprendersi dal problema.\n\n"
        "REGOLE TASSATIVE:\n"
        "1. MINIMALISMO: inserisci nel 'new_plan' SOLO il task correttivo che risolve l'errore, "
        "   più gli eventuali task originali che non sono ancora stati eseguiti e sono ancora necessari. "
        "   Non aggiungere task di verifica o controllo aggiuntivi.\n"
        "2. Scrivi ogni task come una frase semplice e diretta. NON usare prefissi come 'Task:', "
        "   'Step:', numeri o qualsiasi altro prefisso. Esempio corretto: 'Clicca su Rifiuta tutto'.\n"
        "3. Quelli già completati con successo non vanno mai ripetuti.\n"
        "4. Non arrenderti mai: quasi sempre esiste una strategia alternativa. "
        "   Imposta 'stop' a False e fornisci un 'new_plan' correttivo.\n"
        "5. Imposta 'stop' a True ESCLUSIVAMENTE se l'errore è strutturalmente insuperabile "
        "   (es. permessi mancanti, risorsa inesistente e non creabile) e non esiste alcuna alternativa.\n"
        "Genera l'output strutturato rispettando rigorosamente queste regole."
    )),
    ("user", (
        "🎯 RICHIESTA ORIGINALE UTENTE: {original_text}\n\n"
        "📊 DIARIO DI BORDO (Task eseguiti e relativi risultati, incluso l'errore):\n"
        "{past_steps_context}"
    ))
])

planner_chain = _planner_prompt | llm.with_structured_output(PlanSchema)
replanner_chain = _replanner_prompt | llm.with_structured_output(RePlanningSchema)
