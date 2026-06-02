from langchain_core.prompts import ChatPromptTemplate
from config import llm
from tools.db_tools import elenco_tabelle_db, create_table, find_table_info
from tools.web_tools import web_search_tool
from tools.browser_tools import browse_page


_tools = [elenco_tabelle_db, create_table, find_table_info, web_search_tool, browse_page]

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei l'Esecutore specializzato di un sistema agentico.\n"
        "Il tuo compito è risolvere un singolo SOTTO-TASK specifico che ti viene assegnato.\n\n"
        "Per farlo, hai a disposizione degli STRUMENTI (Tools) che puoi invocare se necessario.\n"
        "Se il sotto-task richiede di guardare tabelle o cercare dati, usa il tool adatto "
        "invece di inventare i dati.\n\n"
        "REGOLE PER browse_page:\n"
        "- Per cliccare un bottone di cui non conosci il CSS selector, usa il testo visibile: "
        "  selector='text=Rifiuta tutto' oppure selector='text=Accetta tutto'.\n"
        "- Preferisci SEMPRE selettori testuali (text=...) rispetto a selettori CSS inventati.\n"
        "- Se la pagina mostra un popup o dialog bloccante (es. cookie consent), devi gestirlo "
        "  PRIMA di procedere con il task principale.\n"
        "- Per digitare testo in un campo di ricerca o form, devi SEMPRE seguire questo flusso:\n"
        "  1. Usa action='get_inputs' per ottenere la lista degli elementi interattivi e i loro CSS selector.\n"
        "  2. Usa action='fill' con il selector ESATTO trovato nel risultato di get_inputs.\n"
        "  NON indovinare il selector: usa SOLO selettori restituiti da get_inputs o visibili nel testo della pagina.\n"
        "  NON usare action='fetch' per eseguire una ricerca: fetch legge solo la pagina, non digita nulla.\n\n"
        "Rispondi descrivendo l'azione fatta o mostrando i dati reali ottenuti dagli strumenti."
    )),
    ("user", (
        "📊 DIARIO DI BORDO (Passi già fatti):\n"
        "{past_steps_context}\n\n"
        "🎯 SOTTO-TASK ATTUALE DA RISOLVERE ORA:\n"
        "{current_task}"
    ))
])

executor_chain = _prompt | llm.bind_tools(_tools)

# Exposed for planning node tool dispatch
tools_map = {t.name: t for t in _tools}
