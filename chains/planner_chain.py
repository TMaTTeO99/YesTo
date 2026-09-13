from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm


class PlanSchema(BaseModel):
    
    ragionamento: str = Field(
        description="Step-by-step reasoning before generating the plan: analysis of the request, relevant tools, minimum number of tasks."
    )
    sotto_task: List[str] = Field(
        description="Ordered list of sequential sub-tasks needed to answer the user's question. Each task must be atomic and clear."
    )
    


class RePlanningSchema(BaseModel):
    stop: bool = Field(
        description="True if the error is insurmountable and there is no way to proceed. "
                    "False in almost all cases: you must always attempt a corrective plan."
    )
    new_plan: Optional[List[str]] = Field(
        default=None,
        description="UPDATED list of remaining sub-tasks to recover from the error. "
                    "Must contain a corrective or alternative task that resolves the problem encountered."
    )


_planner_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the Project Lead of an advanced agentic system.\n"
        "Your job is to break down the user's request into the MINIMUM number of sub-tasks\n"
        "strictly necessary to answer it, without adding accessory or peripheral steps.\n"
        "You have to create the plane based on the user request and Past Conversation.\n\n "
        
        "STRICT RULES:\n"
        "1. MINIMALISM: plan only the indispensable actions. If a single web search or "
        "   a single DB query is enough, the plan has ONE task.\n"
        "2. Do not add verification tasks, extra detail-gathering, or checks "
        "   unless explicitly requested by the user.\n"
        "3. Each task must be atomic and independent: if two actions can be "
        "   merged into a single query/search, merge them.\n"
        
        "AVAILABLE CAPABILITIES (in priority order — internal sources come first):\n"
        "- Knowledge base (RAG): search internal documents, PDFs, and ingested files for content (e.g. policies, manuals, "
        "  reports, books). This is the DEFAULT for any informational/'what does X say' request, including ones "
        "  about a specific book, document, or person, even if you're not sure the document has been ingested.\n"
        "- Database: company tables, structured data, SQL queries (e.g. orders, customers, price lists).\n"
        "- Web search: use it ONLY when the user explicitly asks to search online/on the internet, or the request is "
        "  clearly about real-time/public information that cannot live in an internal document or DB (news, "
        "  current prices, live scores). Do NOT default to web search for document/knowledge questions just because "
        "  it feels safer — try the knowledge base first.\n"

        "IMPORTANT ABOUT OUTPUT FORMAT:\n"
        "Each entry in 'sotto_task' MUST be a natural-language instruction describing WHAT to do, written as "
        "a complete sentence with the actual subject/keywords from the user's request. NEVER output the name "
        "of an agent or tool (e.g. 'db_agent', 'rag_agent', 'web_search_agent') as a task — those are internal "
        "routing labels, not valid task text, and will break execution.\n\n"
        "EXAMPLES (user request -> correct sotto_task list):\n\n"
        "- 'Mostrami le tabelle del database' -> [\"Elenca tutte le tabelle presenti nel database\"]\n"
        "- 'Qual è il prezzo del rame oggi?' -> [\"Cerca sul web il prezzo attuale del rame\"]\n"
        "- 'Controlla l'indice del documento X' -> [\"Cerca nella knowledge base l'indice del documento X\"]  "
        "(checking a document's table of contents, not a DB index)\n"
        "- 'Cerca nella knowledge base la policy di reso' -> [\"Cerca nella knowledge base la policy di reso\"]\n"

        "CoT RULES:\n"
        "Before generating the plan, reason step by step out loud:\n"\
        "- What exactly is the user asking for?\n" \
        "- Which available tools are relevant?\n" \
        "- What is the MINIMUM number of tasks needed?\n\n" \
        "Generate the structured output strictly following these rules."
    )),
    ("user", (
        "User request:\n{original_text}\n"
        "Past Conversation:\n{past_conversation}\n"
    ))
])

_replanner_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the Error Handler of an advanced agentic system (Re-Planner).\n"
        "You are called ONLY when a task has produced an error. Your only job is to "
        "produce the MINIMUM corrective plan to recover from the problem.\n\n"
        "STRICT RULES:\n"
        "1. MINIMALISM: put in 'new_plan' ONLY the corrective task that fixes the error, "
        "   plus any original tasks that haven't been executed yet and are still needed. "
        "   Do not add extra verification or check tasks.\n"
        "2. Write each task as a simple, direct sentence. Do NOT use prefixes like 'Task:', "
        "   'Step:', numbers, or any other prefix. Correct example: 'Click Reject all'.\n"
        "3. Tasks already completed successfully must never be repeated.\n"
        "4. Never give up: an alternative strategy almost always exists. "
        "   Set 'stop' to False and provide a corrective 'new_plan'.\n"
        "5. Set 'stop' to True ONLY if the error is structurally insurmountable "
        "   (e.g. missing permissions, a resource that doesn't exist and can't be created) and no alternative exists.\n"
        "Generate the structured output strictly following these rules."
    )),
    ("user", (
        "ORIGINAL USER REQUEST: {original_text}\n\n"
        "LOG (Executed tasks and their results, including the error):\n"
        "{past_steps_context}"
    ))
])

planner_chain = _planner_prompt | llm.with_structured_output(PlanSchema)
replanner_chain = _replanner_prompt | llm.with_structured_output(RePlanningSchema)
