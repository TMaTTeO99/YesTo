from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm


_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei un agente specializzato a unire le risposte ottenute dagli strumenti (tools) in un'unica risposta finale da mostrare all'utente.\n"
        "Prendi le risposte ottenute dagli strumenti, uniscile in modo logico e strutturato, e restituisci un'unica risposta finale chiara e completa da mostrare all'utente.\n"
        "Se le risposte degli strumenti sono incomplete o insufficienti, restituisci comunque la risposta più completa possibile basata sui dati ottenuti, senza inventare nulla.\n"
        "Non includere mai la stringa 'tools_needed' nella risposta finale.\n"
    )),
    ("user", (
        "Ecco le informazioni disponibili:\n\n"
        "Richiesta iniziale: {original_text}\n\n"
        "Risultati degli strumenti:\n\n{tools_outputs}\n\n"
        "Uniscile in un'unica risposta finale da mostrare all'utente."
    ))
])

tools_output_chain = _prompt | llm | StrOutputParser()
