'''
    My first test: "Prompt Chaining" 
'''
from langchain_core.messages import HumanMessage
from Shared.shared import getDBAddress
from langgraph.checkpoint.postgres import PostgresSaver
from Routing.GraphRouter import uncompiled_workflow

if __name__ == "__main__":

    with PostgresSaver.from_conn_string(getDBAddress()) as memory:
        
        memory.setup()
        app_graph = uncompiled_workflow.compile(checkpointer=memory)

        id = input("Inserire Id Sessione:\n")
        while input("Inserire 1 per uscire o 0 per continuare: ") != "1":
            
            prompt = input("Inserire Prompt:\n")
            config = {"configurable": {"thread_id": id}}
        
            stato_iniziale_1 = {
                "original_text": prompt,
                "messages": [HumanMessage(content=prompt)]
            }
        
            output_grafo_1 = app_graph.invoke(stato_iniziale_1, config=config)
            print(f"Risposta:\n{output_grafo_1['messages'][-1].content}\n")
        
        