from langchain_core.prompts import ChatPromptTemplate
from config import llm
from tools.db_tools import elenco_tabelle_db, create_table, find_table_info
from tools.web_tools import web_search_tool
# from tools.browser_tools import browse_page


# _tools = [elenco_tabelle_db, create_table, find_table_info, web_search_tool, browse_page]
_db_tools = [elenco_tabelle_db, create_table, find_table_info]
_web_tools = [web_search_tool]


_web_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei l'Esecutore WEB specializzato di un sistema agentico.\n"
        "Il tuo compito è risolvere un singolo SOTTO-TASK che richiede informazioni dal web.\n\n"
        "Hai a disposizione un tool di ricerca web: usalo per trovare informazioni aggiornate o esterne.\n\n"
        "REGOLE GENERALI:\n"
        "- Rispondi solo al SOTTO-TASK assegnato, non fare altro.\n"
        "- Non limitarti a riportare i risultati grezzi della ricerca: sintetizza solo le informazioni rilevanti al sotto-task.\n"
    )),
    ("user", (
        "📊 DIARIO DI BORDO (Passi già fatti):\n"
        "{past_steps_context}\n\n"
        "🎯 SOTTO-TASK ATTUALE DA RISOLVERE ORA:\n"
        "{current_task}"
    ))
])

_db_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei l'Esecutore DATABASE specializzato di un sistema agentico.\n"
        "Il tuo compito è risolvere un singolo SOTTO-TASK relativo al database aziendale (tabelle, dati, schema).\n\n"
        "Hai a disposizione tool per elencare tabelle, ispezionarne la struttura e crearne di nuove.\n\n"
        "REGOLE GENERALI:\n"
        "- Rispondi solo al SOTTO-TASK assegnato, non fare altro.\n"
        "- Prima di creare o modificare una tabella, verifica se esiste già o se ti servono informazioni sulla sua struttura.\n"
        "- Non inventare mai nomi di tabelle o colonne: usa solo quelli confermati dai tool.\n"
    )),
    ("user", (
        "📊 DIARIO DI BORDO (Passi già fatti):\n"
        "{past_steps_context}\n\n"
        "🎯 SOTTO-TASK ATTUALE DA RISOLVERE ORA:\n"
        "{current_task}"
    ))
])


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



db_executor_chain = _db_prompt | llm.bind_tools(_db_tools)
web_executor_chain = _web_prompt | llm.bind_tools(_web_tools)


# Exposed for planning node tool dispatch
web_tools_map = {t.name: t for t in _web_tools}
db_tools_map = {t.name: t for t in _db_tools}


