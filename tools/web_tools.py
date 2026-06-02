import json
from langchain_core.tools import tool
from config import get_web_search, debug_print


def _make_tool_response(success: bool, receipt: str, summary: str, details: str) -> str:
    return json.dumps(
        {"success": success, "receipt": receipt, "summary": summary, "details": details},
        ensure_ascii=False,
    )


@tool
def web_search_tool(query: str) -> str:
    """Usa questo strumento SOLO quando l'utente chiede informazioni in tempo reale,
    notizie recenti, fatti di attualità o concetti generali non presenti nel database aziendale.
    Il parametro 'query' deve essere la stringa testuale esatta da inviare al motore di ricerca (es. 'Meteo Milano oggi')."""
    debug_print(f"🌐 [TOOL WEB] web_search_tool(query='{query}')")

    try:
        results = get_web_search().invoke({"query": query})
        if not results:
            msg = "La ricerca online non ha prodotto nessun risultato utile per questa query."
            return _make_tool_response(False, msg, msg, msg)

        details = f"Risultati estratti dal Web per '{query}':\n\n{results}"
        summary = f"Ricerca web completata con successo per '{query}'."
        return _make_tool_response(True, f"Ricerca web eseguita per '{query}'.", summary, details)

    except Exception as e:
        msg = f"Errore durante la ricerca web: {e}"
        return _make_tool_response(False, msg, msg, msg)
