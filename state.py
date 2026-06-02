from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # --- Conversazione ---
    # ID della sessione utente, usato per filtrare la memoria RAG nel vector store
    session_id: str
    # Testo originale dell'utente per il turno corrente; può essere arricchito con
    # chat history prima di entrare nel sotto-grafo di planning
    original_text: str
    # Lista completa dei messaggi (HumanMessage, AIMessage...). L'annotazione
    # add_messages dice a LangGraph di appendere i nuovi messaggi invece di
    # sovrascrivere l'intera lista ad ogni aggiornamento dello stato
    messages: Annotated[list[BaseMessage], add_messages]
    # Riassunto testuale delle conversazioni più vecchie, usato quando i messaggi
    # superano il buffer e vengono compressi per risparmiare token
    message_summary: str
    # Buffer temporaneo dei messaggi da riassumere prima di scartarli
    message_buffer: list[BaseMessage]

    # --- Loop di riflessione QA ---
    # Flag che indica se il nodo critica ha approvato la risposta generata
    critique_approvata: bool
    # Lista di punti di correzione restituiti dal critico; viene iniettata nel
    # prompt del nodo domanda al turno successivo se la risposta non è approvata
    critique_punti: list[str]
    # Contatore dei tentativi di risposta nel loop domanda→critica; il grafo
    # esce forzatamente dopo 5 tentativi per evitare loop infiniti
    threshold: int

    # --- Sotto-grafo di planning ---
    # Lista ordinata di sotto-task prodotta dal planner; l'execution node
    # consuma un task alla volta facendo pop dal primo elemento
    plan: List[str]
    # Storico dei task già eseguiti con il loro risultato, usato dal replanner
    # per decidere se continuare o correggere il piano
    past_steps: List[dict]
    # Risposta finale sintetizzata dopo che tutti i tool hanno restituito i
    # loro risultati; è quello che viene mostrato all'utente a fine planning
    response: str
