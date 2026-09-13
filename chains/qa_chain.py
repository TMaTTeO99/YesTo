from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import llm


class QAResponse(BaseModel):
    needs_tools: bool = Field(
        description=(
            "TRUE if the request requires: database access, specific company data, "
            "creating/modifying tables, real-time web searches, or any operational action. "
            "FALSE if you can answer with static general knowledge or the chat history."
        )
    )
    answer: str = Field(
        description=(
            "If needs_tools is FALSE: the complete answer to the user's question. "
            "If needs_tools is TRUE: leave this field empty ('')."
        )
    )


_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an internal company assistant. Decide whether you can answer with your "
        "general knowledge/chat history, or whether external tools are needed.\n\n"
        "USE needs_tools=TRUE for:\n"
        "- Any database operation (listing tables, creating tables, reading data)\n"
        "- Specific company data \n"
        "- Web searches, recent news, real-time data\n"
        "- Any question about the content of documents, PDFs, files, manuals, policies, or anything that "
        "  would live in an internal knowledge base — you have NO document content available here, only chat_history\n"
        "- ANY message that asks to retry, search again, use tools, search online/in the knowledge base, or "
        "  otherwise instructs you to look something up — even a short imperative like 'riprova', 'cerca dinuovo', "
        "  'cerca nella tua conoscenza' — MUST be needs_tools=True. Never answer these from memory/general knowledge, "
        "  and never say 'I already searched and found nothing' instead of actually triggering a new search.\n"
        "- If you don't know the answare to the user, never invent the answare\n"
        "- Analysis of files or data you don't have in memory\n\n"
        "USE needs_tools=FALSE for:\n"
        "- Static general knowledge (capitals, definitions, history, math...)\n"
        "- Greetings, courtesy, small talk\n"
        "- Questions you can answer purely from chat_history (this conversation's own prior turns), "
        "  NOT from any document or knowledge base\n\n"
        "EXAMPLES:\n"
        "- 'Qual è la capitale della Francia?' -> needs_tools=False, answer='Parigi'\n"
        "- 'Controlla l'ultimo ordine di Matteo' -> needs_tools=True, answer=''\n"
        "- 'Mostrami le tabelle del database' -> needs_tools=True, answer=''\n"
        "- 'Ciao come stai?' -> needs_tools=False, answer='Ciao! Sto bene, grazie.'\n"
        "- 'Quali sono i prezzi del nostro listino?' -> needs_tools=True, answer=''\n"
        "- 'Qual è l'indice del libro X?' / 'Cosa dice il documento che ho caricato?' -> needs_tools=True, answer=''\n"
        "- 'riprova' / 'cerca dinuovo' / 'cerca nella tua conoscenza' -> needs_tools=True, answer=''\n\n"
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "User question: {original_text}")
])

qa_chain = _prompt | llm.with_structured_output(QAResponse)
