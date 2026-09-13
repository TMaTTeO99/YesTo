import json
from langchain_core.tools import tool
from config import get_web_search, debug_print
from tools.utils import _make_tool_response

@tool
def web_search_tool(query: str) -> str:
    """Use this tool ONLY when the user asks for real-time information,
    recent news, current events, or general concepts not present in the company database.
    The 'query' parameter must be the exact text string to send to the search engine (e.g. 'Meteo Milano oggi')."""
    debug_print(f"🌐 [TOOL WEB] web_search_tool(query='{query}')")

    try:
        results = get_web_search().invoke({"query": query})
        if not results:
            msg = "The web search doesn't produce result for the query."
            return _make_tool_response(False, msg, msg, msg)

        details = f"Extracted result from web '{query}':\n\n{results}"
        summary = f"Web search completed '{query}'."
        return _make_tool_response(True, f"Web search about '{query}'.", summary, details)

    except Exception as e:
        msg = f"Error in web search: {e}"
        return _make_tool_response(False, msg, msg, msg)
