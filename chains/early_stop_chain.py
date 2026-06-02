from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm


class EarlyStopSchema(BaseModel):
    obiettivo_raggiunto: bool = Field(
        description="TRUE se l'obiettivo finale dell'utente è già stato pienamente raggiunto "
                    "con i task già eseguiti, rendendo inutile proseguire con i task rimanenti. "
                    "FALSE se i task rimanenti sono ancora necessari per completare l'obiettivo."
    )
    motivazione: str = Field(
        description="Breve spiegazione del perché l'obiettivo è già raggiunto o perché è necessario continuare."
    )


_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei l'Ottimizzatore di un sistema agentico avanzato.\n"
        "Vieni chiamato dopo ogni task eseguito con successo per decidere se la RICHIESTA ORIGINALE "
        "dell'utente ha già ricevuto una risposta sufficiente, rendendo inutile proseguire.\n\n"
        "REGOLA FONDAMENTALE: ignora il piano. Il piano è una stima iniziale che può essere "
        "sovrastimata. Concentrati SOLO su: la domanda dell'utente ha già una risposta concreta "
        "e soddisfacente nei risultati ottenuti finora?\n\n"
        "ESEMPI:\n"
        "- Utente chiede 'quanto costa X?' → se nei risultati c'è già un prezzo → obiettivo_raggiunto=True\n"
        "- Utente chiede 'crea la tabella Y' → se la tabella non è ancora stata creata → False\n"
        "- Utente chiede 'cerca notizie su X' → se i risultati web contengono notizie su X → True\n\n"
        "REGOLE:\n"
        "1. Imposta obiettivo_raggiunto=True se la risposta alla domanda originale è già presente "
        "   nei risultati, anche parzialmente. Non serve raccogliere ogni dettaglio possibile.\n"
        "2. Imposta obiettivo_raggiunto=False SOLO se manca ancora l'informazione o l'azione "
        "   centrale richiesta dall'utente (non dettagli accessori).\n"
        "3. I task rimanenti nel piano NON sono un motivo sufficiente per continuare: il piano "
        "   può essere sovrastimato rispetto alla reale necessità dell'utente.\n"
        "4. ECCEZIONE CRITICA: se nei risultati è visibile un popup, dialog o schermata bloccante "
        "   (es. cookie consent, login wall, GDPR) che impedisce l'accesso al contenuto reale, "
        "   imposta obiettivo_raggiunto=False anche se la pagina è stata caricata. "
        "   Il contenuto non è accessibile finché il blocco non viene rimosso.\n"
        "Genera l'output strutturato rispettando rigorosamente queste regole."
    )),
    ("user", (
        "🎯 RICHIESTA ORIGINALE UTENTE: {original_text}\n\n"
        "✅ RISULTATI OTTENUTI FINORA:\n{past_steps_context}\n\n"
        "📋 TASK RIMANENTI NEL PIANO (potrebbero essere superflui):\n{remaining_tasks}"
    ))
])

early_stop_chain = _prompt | llm.with_structured_output(EarlyStopSchema)
