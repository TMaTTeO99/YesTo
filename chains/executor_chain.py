from langchain_core.prompts import ChatPromptTemplate
from config import llm
from tools.db_tools import elenco_tabelle_db, create_table, find_table_info
from tools.rag_tools import search_knowledge_base
from tools.web_tools import web_search_tool
# from tools.browser_tools import browse_page


# _tools = [elenco_tabelle_db, create_table, find_table_info, web_search_tool, browse_page]
_db_tools = [elenco_tabelle_db, create_table, find_table_info]
_web_tools = [web_search_tool]
_rag_tools = [search_knowledge_base]


_web_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the specialized WEB Executor of an agentic system.\n"
        "Your job is to resolve a single SUB-TASK that requires information from the web.\n\n"
        "You have a web search tool available: use it to find up-to-date or external information.\n\n"
        "GENERAL RULES:\n"
        "- Only address the assigned SUB-TASK, nothing else.\n"
        "- Don't just report raw search results: synthesize only the information relevant to the sub-task.\n"
    )),
    ("user", (
        "ORIGINAL USER REQUEST (use it to figure out the concrete subject/keywords to search for):\n"
        "{original_text}\n\n"
        "LOG (Steps already done):\n"
        "{past_steps_context}\n\n"
        "CURRENT SUB-TASK TO RESOLVE NOW:\n"
        "{current_task}"
    ))
])

_db_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the specialized DATABASE Executor of an agentic system.\n"
        "Your job is to resolve a single SUB-TASK related to the company database (tables, data, schema).\n\n"
        "You have tools available to list tables, inspect their structure, and create new ones.\n\n"
        "GENERAL RULES:\n"
        "- Only address the assigned SUB-TASK, nothing else.\n"
        "- Before creating or modifying a table, check whether it already exists or whether you need information about its structure.\n"
        "- Never invent table or column names: only use ones confirmed by the tools.\n"
    )),
    ("user", (
        "ORIGINAL USER REQUEST (use it to figure out the concrete subject/keywords to search for):\n"
        "{original_text}\n\n"
        "LOG (Steps already done):\n"
        "{past_steps_context}\n\n"
        "CURRENT SUB-TASK TO RESOLVE NOW:\n"
        "{current_task}"
    ))
])

_rag_promt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the specialized RAG Executor of an agentic system.\n"
        "Your job is to resolve a single SUB-TASK related to the retreive information from documents stored inside your database.\n"
        "You have tools available in the list tables to retrieve informations"
        "GENERAL RULES:\n"
        "- Only address the assigned SUB-TASK, nothing else.\n"
        "- Never invent data about information requested by user.\n"
        "- When calling the search tool, build the 'query' argument from the concrete subject/keywords in the "
        "ORIGINAL USER REQUEST (e.g. names, topics, document titles). Never pass the sub-task label itself "
        "(like 'search the knowledge base') as the query.\n"
    )),
    ("user", (
        "ORIGINAL USER REQUEST (use it to figure out the concrete subject/keywords to search for):\n"
        "{original_text}\n\n"
        "LOG (Steps already done):\n"
        "{past_steps_context}\n\n"
        "CURRENT SUB-TASK TO RESOLVE NOW:\n"
        "{current_task}"
    ))
])


_observation_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You just executed a tool to resolve a sub-task. "
        "Analyze the REAL result received and write:\n"
        "OBSERVATION: [what you learned from the result]\n"
        "FINAL ANSWER: [the answer to the sub-task based on the result]\n"
    )),
    ("user", (
        "SUB-TASK: {current_task}\n\n"
        "TOOL RESULT:\n{tool_result}"
    ))
])

observation_chain = _observation_prompt | llm


rag_executor_chain = _rag_promt | llm.bind_tools(_rag_tools)
db_executor_chain = _db_prompt | llm.bind_tools(_db_tools)
web_executor_chain = _web_prompt | llm.bind_tools(_web_tools)


# Exposed for planning node tool dispatch
web_tools_map = {t.name: t for t in _web_tools}
db_tools_map = {t.name: t for t in _db_tools}
rag_tools_map = {t.name: t for t in _rag_tools}


