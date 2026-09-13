from langchain_core.prompts import ChatPromptTemplate
from config import llm
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


class SupervisorAgentCall(str, Enum):
    WEB_SEARCH_AGENT = "web_search_agent" 
    DB_AGENT = "db_agent"
    RAG_AGENT = "rag_agent"

class ToolsSupervisorSchema(BaseModel):

    selected_agent: SupervisorAgentCall = Field(
        description= "The sub-agent selected to execute the user's request, based on the conversation context and the request itself.\n" \
        "You must respond exclusively with the name of the selected sub-agent, without adding further explanations or comments.\n" \
        "Available sub-agents:\n" \
        "rag_agent: the DEFAULT choice for any informational request — knowledge base / internal documents, PDFs, "
        "files, manuals, reports, CVs, policies, books, or generally 'what does X say/contain'. If you are not sure "
        "where the information lives, prefer rag_agent: it's cheap to try and internal data always takes priority.\n" \
        "db_agent: If the user's request requires accessing company data or tables in the SQL database "
        "(orders, customers, structured records).\n" \
        "web_search_agent: use it ONLY when at least one of these is true:\n" \
        "  (a) the user explicitly asked to search online / on the web / on the internet, or\n" \
        "  (b) the request is clearly about real-time or public information that cannot live in an internal "
        "document or DB (news, current prices, live scores, general public facts), or\n" \
        "  (c) the LOG below already shows rag_agent and/or db_agent were tried for this same request and failed "
        "(success: False) or returned nothing relevant — in that case web_search_agent is an acceptable fallback.\n" \
        "NEVER pick web_search_agent as a first guess for a document/knowledge/informational request just because "
        "it hasn't been tried yet — try rag_agent first.\n" \
        "EXAMPLES:\n" \
        "- 'cerca l'indice del libro X' (no prior attempts) -> rag_agent\n" \
        "- 'hai informazioni sul cv di Mario Rossi?' (no prior attempts) -> rag_agent\n" \
        "- 'cerca l'indice del libro X' (LOG shows rag_agent already failed) -> web_search_agent\n" \
        "- 'qual è il prezzo del rame oggi?' -> web_search_agent (inherently real-time public info)\n" \
        "- 'cerca online il prezzo del libro X' -> web_search_agent (user explicitly said online)\n" \
        "- 'mostrami le tabelle del database' -> db_agent"
    )

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a supervisor agent specialized in choosing which sub-agent to run based on the user's request and the plan proposed by the planning agent.\n"
        "Internal sources (knowledge base, database) always take priority over the public web unless the rules below say otherwise.\n"
    )),
    ("user", (
        "Current user request: {original_text}\n\n"
        "Remaining plan: {plan}\n\n"
        "LOG (tools already tried for this request, in order):\n{past_steps_context}\n\n"
    ))
])

supervisor_chain = _prompt | llm.with_structured_output(ToolsSupervisorSchema)