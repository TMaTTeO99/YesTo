from Shared.shared import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# NOT USED

# Prompt to analyze the complaint and extract the emotional tone
feeling_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei un esperto psicologo. Analizza il tono emotivo del reclamo"
    "e sintetizza in massimo 2 righe il sentimenti e le emozioni del cliente."
    "All'interno della risposta finale NON devi mai inserire commenti riguardo le correzzioni che hai applicato in relazione"
                "alle critiche dell' agent dedicato alle critiche"),
    ("user", "Reclamo del cliente: {original_text}")
])
feeling_chain = feeling_prompt | llm | StrOutputParser()

# prompt to analyze the complaint and extract the key points to solve the problem
emotional_solution_promt = ChatPromptTemplate.from_messages([
    ("system", "Sei un esperto di analisi del testo. Analizza il reclamo del cliente e identifica quali"
    " sono i punti critici da affrontare per risolvere il problema in modo efficace."
    "Sintetizza in massimo 3 punti chiave i problemi principali da affrontare per risolvere il reclamo del cliente."
    "All'interno della risposta finale NON devi mai inserire commenti riguardo le correzzioni che hai applicato in relazione"
                "alle critiche dell' agent dedicato alle critiche"),
    ("user", "Reclamo del cliente: {original_text}")
])
emotional_solution_chain = emotional_solution_promt | llm | StrOutputParser()

# prompt to unify the two analyses in a single response to the customer
resume_emotional_promt = ChatPromptTemplate.from_messages([
    ("system", "Sei un esperto supervisore editoriale. Prendi queste due analisi indipendenti e uniscile per "
    "fornire una risposta completa ed efficace al cliente che ha presentato il reclamo."
    "Assicurati di affrontare sia gli aspetti emotivi che quelli pratici del reclamo, fornendo una risposta empatica e risolutiva."
    "All'interno della risposta finale NON devi mai inserire commenti riguardo le correzzioni che hai applicato in relazione"
                "alle critiche dell' agent dedicato alle critiche"),
    ("user", (
        "Ecco  le due analisi da unificare:\n\n"
        "- Analisi Emotiva: {feeling_result}\n"
        "- Soluzione Emotiva: {emotional_solution_result}"
    ))
])
resume_emotional_chain = resume_emotional_promt | llm | StrOutputParser()

# input map to take the original input and the chat history 
input_map = {
    "original_text": lambda x : x["original_text"],
}

# parallel block to execute the three chains in parallelo and then unify the results
parallel_block = {
    "feeling_result" : feeling_chain,
    "emotional_solution_result" : emotional_solution_chain,
}

parallel_emotional_chain = input_map | parallel_block | resume_emotional_chain | llm | StrOutputParser()