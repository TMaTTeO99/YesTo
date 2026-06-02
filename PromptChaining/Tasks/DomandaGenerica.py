from pydantic import BaseModel, Field
from Shared.shared import llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class DomandaResponse(BaseModel):
    needs_tools: bool = Field(
        description=(
            "TRUE se la richiesta richiede: accesso al database, dati aziendali specifici, "
            "creazione/modifica di tabelle, ricerche web in tempo reale, o qualsiasi azione operativa. "
            "FALSE se puoi rispondere con conoscenza generale statica o con la cronologia della chat."
        )
    )
    answer: str = Field(
        description=(
            "Se needs_tools è FALSE: la risposta completa alla domanda dell'utente. "
            "Se needs_tools è TRUE: lascia questo campo vuoto ('')."
        )
    )


structured_question_llm = llm.with_structured_output(DomandaResponse)

question_check_knowledge_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei un assistente aziendale interno. Decidi se puoi rispondere con la tua conoscenza "
        "generale/cronologia della chat, oppure se servono strumenti esterni.\n\n"

        "USA needs_tools=TRUE per:\n"
        "- Qualsiasi operazione sul database (elencare tabelle, creare tabelle, leggere dati)\n"
        "- Dati aziendali specifici (ordini, clienti, listini, report)\n"
        "- Ricerche web, notizie recenti, dati in tempo reale\n"
        "- Analisi di file o dati che non hai in memoria\n\n"

        "USA needs_tools=FALSE per:\n"
        "- Conoscenza generale statica (capitali, definizioni, storia, matematica...)\n"
        "- Saluti, cortesie, chiacchiere\n"
        "- Domande che puoi rispondere dalla chat_history\n\n"

        "ESEMPI:\n"
        "- 'Qual è la capitale della Francia?' -> needs_tools=False, answer='Parigi'\n"
        "- 'Controlla l'ultimo ordine di Matteo' -> needs_tools=True, answer=''\n"
        "- 'Mostrami le tabelle del database' -> needs_tools=True, answer=''\n"
        "- 'Ciao come stai?' -> needs_tools=False, answer='Ciao! Sto bene, grazie.'\n"
        "- 'Quali sono i prezzi del nostro listino?' -> needs_tools=True, answer=''"
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Domanda dell'utente: {original_text}")
])

parallel_question_chain = question_check_knowledge_prompt | structured_question_llm