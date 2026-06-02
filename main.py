import os
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from config import get_db_address, init_rag
from graph import workflow


def _handle_ingest(arg: str):
    from memory.vector_store import ingest_pdf, ingest_text
    path = arg.strip()
    if not os.path.exists(path):
        print(f"[ERRORE] File non trovato: {path}")
        return
    if path.lower().endswith(".pdf"):
        n = ingest_pdf(path)
        print(f"[RAG] PDF indicizzato: {n} chunk da '{os.path.basename(path)}'")
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        n = ingest_text(text, source=os.path.basename(path))
        print(f"[RAG] Testo indicizzato: {n} chunk da '{os.path.basename(path)}'")


if __name__ == "__main__":
    init_rag()

    with PostgresSaver.from_conn_string(get_db_address()) as memory:
        memory.setup()
        app = workflow.compile(checkpointer=memory)

        session_id = input("Inserire Id Sessione:\n")
        while True:
            raw = input("Inserire 1 per uscire, /ingest <file> per caricare documenti, oppure il tuo prompt:\n")

            if raw.strip() == "1":
                break

            if raw.strip().startswith("/ingest "):
                _handle_ingest(raw.strip()[len("/ingest "):])
                continue

            prompt = raw
            config = {"configurable": {"thread_id": session_id}}
            state = {
                "session_id": session_id,
                "original_text": prompt,
                "messages": [HumanMessage(content=prompt)],
            }
            output = app.invoke(state, config=config)
            response = output.get("response") or output["messages"][-1].content
            print(f"Risposta:\n{response}\n")
