from langchain_core.prompts import ChatPromptTemplate
from config import llm
from tools.db_tools import elenco_tabelle_db, create_table, find_table_info
from tools.web_tools import web_search_tool
# from tools.browser_tools import browse_page


# _tools = [elenco_tabelle_db, create_table, find_table_info, web_search_tool, browse_page]
_tools = [elenco_tabelle_db, create_table, find_table_info, web_search_tool]

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei l'Esecutore specializzato di un sistema agentico.\n"
        "Il tuo compito è risolvere un singolo SOTTO-TASK specifico che ti viene assegnato.\n\n"
        "Per farlo, hai a disposizione degli STRUMENTI (Tools) che puoi invocare se necessario.\n"
        "Se il sotto-task richiede di guardare tabelle o cercare dati, usa il tool adatto\n\n"
        "REGOLE GENERALI:\n"
        "Se il sotto-task richiede un tool, richiamalo direttamente tramite il meccanismo di function calling "
        "(NON descrivere l'azione come testo, invocala realmente).\n"
        "Se non serve alcun tool, rispondi direttamente al sotto-task.\n"
    )),
    ("user", (
        "📊 DIARIO DI BORDO (Passi già fatti):\n"
        "{past_steps_context}\n\n"
        "🎯 SOTTO-TASK ATTUALE DA RISOLVERE ORA:\n"
        "{current_task}"
    ))
])

executor_chain = _prompt | llm.bind_tools(_tools)

_observation_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Hai appena eseguito un tool per risolvere un sotto-task. "
        "Analizza il risultato REALE ricevuto e scrivi:\n"
        "OSSERVAZIONE: [cosa hai imparato dal risultato]\n"
        "RISPOSTA FINALE: [la risposta al sotto-task basata sul risultato]\n"
    )),
    ("user", (
        "🎯 SOTTO-TASK: {current_task}\n\n"
        "📥 RISULTATO DEL TOOL:\n{tool_result}"
    ))
])

observation_chain = _observation_prompt | llm

# Exposed for planning node tool dispatch
tools_map = {t.name: t for t in _tools}


