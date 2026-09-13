from langchain_core.tools import tool
from memory.vector_store import search_documents
from tools.utils import _make_tool_response
from config import debug_print

@tool
def search_knowledge_base(query: str, k: int = 3):
    
    """
        Use this tool to search information from documents in your knowledge base.
        PARAMETERS:
            - query: The user query to search information from from documents
            - k: The max number of result that must be searched         
    """
    debug_print(f"[SEARCH KNOLADGE TOOL]: query: {query}")

    search_result = search_documents(query=query, k=k)
    if not search_result:
        msg = "The knowledge base search didn't produce any result for the query."
        return _make_tool_response(False, msg, msg, msg)

    formatted_chunks = []
    for doc in search_result:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
        content = " ".join(doc.page_content.split())
        formatted_chunks.append(f"[{source}, pag. {page}]\n{content}")

    details = f"Extracted results for '{query}':\n\n" + "\n\n---\n\n".join(formatted_chunks)
    summary = f"Research about '{query}'."
    return _make_tool_response(True, f"Search done about '{query}'.", summary, details)
