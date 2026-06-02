from langchain_core.prompts import ChatPromptTemplate
from Shared.shared import llm
from Tools.Tools import Find_table_info, elenco_tabelle_db, Create_table, Cerca_su_Web, Interagisci_con_Pagina_Web


tools_list = [elenco_tabelle_db, Create_table, Find_table_info, Cerca_su_Web, Interagisci_con_Pagina_Web]
llm_with_tools = llm.bind_tools(tools_list)

executor_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei l'Esecutore specializzato di un sistema agentico.\n"
        "Il tuo compito è risolvere un singolo SOTTO-TASK specifico che ti viene assegnato.\n\n"
        "Per farlo, hai a disposizione degli STRUMENTI (Tools) che puoi invocare se necessario.\n"
        "Se il sotto-task richiede di guardare tabelle o cercare dati, usa il tool adatto "
        "invece di inventare i dati.\n\n"
        "REGOLE PER Interagisci_con_Pagina_Web:\n"
        "- Per cliccare un bottone di cui non conosci il CSS selector, usa il testo visibile: "
        "  selector='text=Rifiuta tutto' oppure selector='text=Accetta tutto'.\n"
        "- Preferisci SEMPRE selettori testuali (text=...) rispetto a selettori CSS inventati.\n"
        "- Se la pagina mostra un popup o dialog bloccante (es. cookie consent), devi gestirlo "
        "  PRIMA di procedere con il task principale.\n\n"
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