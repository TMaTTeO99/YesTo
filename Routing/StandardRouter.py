from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Shared.shared import llm

coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei un espeto di analisi del testo e devi analizzare l'input dell'utente."
                "Analizza il testo dell utente e verifica se è una domanda. Se è una domanda rispondi ESATTAMENTE con: 'domanda'"
                "Non aggiungere punteggiatura o altre parole."
                "Cerca di dare una sola opzione alla volta, non dare mai due opzioni ma SOLTANTO UNA"
                "Non inventare nulla, puoi dare come opzioni soltanto quelle che ti ho indicato"),
    ("user", "{original_text}")
])
coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()


'''
    Old routing pattern without memory managment
'''

# memory_checkpointer = MemorySaver()

# threads_database = {}

# def get_thread_history(thread_id: str) -> ChatMessageHistory :
#     if thread_id not in threads_database:
#         threads_database[thread_id] = ChatMessageHistory()
#     return threads_database[thread_id]


# def coordinator_agent(original_text: str, thread_id: str):
    
#     history_backend = get_thread_history(thread_id=thread_id)
#     chat_history = history_backend.messages

#     decision = coordinator_router_chain.invoke({"original_text" : original_text}).strip().lower()
#     print(f"DEBUG: decision -> [{decision}]\n")

#     match decision :
#         case "reclamo": 
#             output_text = parallel_emotional_chain.invoke({"original_text" : original_text, "chat_history": chat_history})
#         case "domanda":
#             output_text = parallel_question_chain.invoke({"original_text" : original_text, "chat_history": chat_history})
#         case _:
#             output_text = "Mi dispiace. Impossibile servire la richiesta."
        
#     history_backend.add_user_message(original_text)
#     history_backend.add_ai_message(output_text)

#     return output_text

# branches = {
#     "report" : RunnablePassthrough.assign(output = lambda x : final_chain.invoke({"original_text" : x["request"]})),
#     "reclamo" : RunnablePassthrough.assign(output = lambda x : complaint_chain.invoke({"original_text" : x["request"]})),
#     "canzone" : RunnablePassthrough.assign(output = lambda x : canzone_chain.invoke({"original_text" : x["request"]})),
#     "generico" : RunnablePassthrough.assign(output = lambda x: parallel_analysis_chain.invoke({"original_text" : x["request"]}))
# }

# delegation_branch = RunnableBranch(
#     (lambda x : x["decision"].strip().lower() == "report", branches["report"]),
#     (lambda x : x["decision"].strip().lower() == "canzone", branches["canzone"]),
#     (lambda x : x["decision"].strip().lower() == "reclamo", branches["reclamo"]),
#     branches["generico"]
# )

# coordinator_agent = {
#     "decision" : coordinator_router_chain,
#     "request" : RunnablePassthrough()
# } | delegation_branch | (lambda x : x["output"])