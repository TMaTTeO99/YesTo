from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Shared.shared import llm
from Tools.Tools import Find_table_info, elenco_tabelle_db, Create_table


tools_list = [elenco_tabelle_db, Create_table, Find_table_info]
llm_with_tools = llm.bind_tools(tools_list)

executor_prompt = ChatPromptTemplate.from_messages([
    # ("system", (
    #     "Sei l'Esecutore specializzato di un sistema agentico.\n"
    #     "Il tuo compito è risolvere un singolo SOTTO-TASK specifico che ti viene assegnato.\n"
    #     "Per farlo in modo accurato, hai a disposizione il 'Diario di bordo' che contiene "
    #     "i risultati dei sotto-task che sono già stati eseguiti prima di questo.\n\n"
    #     "Regole:\n"
    #     "- Concentrati SOLO sul sotto-task attuale.\n"
    #     "- Usa le informazioni dei passi passati se sono utili per risolvere il task attuale.\n"
    #     "- Rispondi in modo dettagliato, preciso e professionale, fornendo il risultato dell'azione."
    # )),
    # ("user", (
    #     "📊 DIARIO DI BORDO (Passi già fatti in precedenza):\n"
    #     "{past_steps_context}\n\n"
    #     "🎯 SOTTO-TASK ATTUALE DA RISOLVERE ORA:\n"
    #     "{current_task}"
    # ))
    ("system", (
        "Sei l'Esecutore specializzato di un sistema agentico.\n"
        "Il tuo compito è risolvere un singolo SOTTO-TASK specifico che ti viene assegnato.\n\n"
        "Per farlo, hai a disposizione degli STRUMENTI (Tools) che puoi invocare se necessario.\n"
        "Se il sotto-task richiede di guardare tabelle o cercare bug, usa il tool adatto "
        "invece di inventare i dati.\n\n"
        "Rispondi descrivendo l'azione fatta o mostrando i dati reali ottenuti dagli strumenti."
    )),
    ("user", (
        "📊 DIARIO DI BORDO (Passi già fatti):\n"
        "{past_steps_context}\n\n"
        "🎯 SOTTO-TASK ATTUALE DA RISOLVERE ORA:\n"
        "{current_task}"
    ))
])

executor_chain = executor_prompt | llm_with_tools