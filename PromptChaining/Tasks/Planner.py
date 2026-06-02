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
        description="True se l'errore è insuperabile e non è possibile procedere in nessun modo. "
                    "False nella quasi totalità dei casi: devi sempre tentare un piano correttivo."
    )
    new_plan: Optional[List[str]] = Field(
        default=None,
        description="Lista AGGIORNATA dei sotto-task rimanenti per recuperare dall'errore. "
                    "Deve contenere un task correttivo o alternativo che risolva il problema riscontrato."
    )
    

# Forziamo il modello a rispondere usando rigorosamente lo schema JSON
structured_planner_llm = llm.with_structured_output(PlanSchema)

planner_prompt = ChatPromptTemplate.from_messages([
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
        "4. Massimo 3 task salvo casi eccezionali con operazioni multi-step realmente distinte "
        "   (es. leggi schema DB → crea tabella → verifica creazione).\n"
        "5. Se la richiesta prevede di digitare testo in una pagina web (barra di ricerca, form, ecc.), "
        "   pianifica SEMPRE tre task separati:\n"
        "   (1) 'Apri la pagina <URL> con fetch'\n"
        "   (2) 'Individua gli elementi interattivi della pagina con get_inputs'\n"
        "   (3) 'Compila il campo di ricerca con <testo> e invia'\n"
        "   Non accorpare questi step e non sostituire il task (2) con un altro fetch.\n"
        "Genera l'output strutturato rispettando rigorosamente queste regole."
    )),
    ("user", "Richiesta dell'utente:\n\n{original_text}")
])

# 3. La Catena del Planner
planner_chain = planner_prompt | structured_planner_llm


structured_replanner_llm = llm.with_structured_output(RePlanningSchema)

replanner_prompt = ChatPromptTemplate.from_messages([
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
        "4. Se l'errore è un timeout su action='fill' (selector non trovato), il piano correttivo "
        "   DEVE iniziare con 'Individua gli elementi interattivi della pagina con get_inputs' "
        "   per trovare il selector corretto, seguito da un nuovo tentativo di fill con il selector giusto. "
        "   NON riproporre lo stesso selector che ha già fallito.\n"
        "5. Non arrenderti mai: quasi sempre esiste una strategia alternativa. "
        "   Imposta 'stop' a False e fornisci un 'new_plan' correttivo.\n"
        "4. Imposta 'stop' a True ESCLUSIVAMENTE se l'errore è strutturalmente insuperabile "
        "   (es. permessi mancanti, risorsa inesistente e non creabile) e non esiste alcuna alternativa.\n"
        "Genera l'output strutturato rispettando rigorosamente queste regole."
    )),
    ("user", (
        "🎯 RICHIESTA ORIGINALE UTENTE: {original_text}\n\n"
        "📊 DIARIO DI BORDO (Task eseguiti e relativi risultati, incluso l'errore):\n"
        "{past_steps_context}"
    ))
])

replanner_chain = replanner_prompt | structured_replanner_llm