from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Shared.shared import llm

final_tools_output_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei un agente specializzato a unire le risposte ottenute dagli strumenti (tools) in un'unica risposta finale da mostrare all'utente.\n" 
               "Prendi le risposte ottenute dagli strumenti, uniscile in modo logico e strutturato, e restituisci un'unica risposta finale chiara e completa da mostrare all'utente.\n"
               "Se le risposte degli strumenti sono incomplete o insufficienti, restituisci comunque la risposta più completa possibile basata sui dati ottenuti, senza inventarere nulla.\n"),
    ("user", "Ecco le risposte ottenute dagli strumenti:\n\n{tools_outputs}\n\nUniscile in un'unica risposta finale da mostrare all'utente.")
])
final_tools_output_chain = final_tools_output_prompt | llm | StrOutputParser()
