import json
from langchain_core.tools import tool
from Shared.shared import engine, debug_print
from sqlalchemy import inspect, text
from Shared.shared import web_search
from typing import List, Dict, Any


def _make_tool_response(success: bool, receipt: str, summary: str, details: str) -> dict:
    return json.dumps({
        "success": success,
        "receipt": receipt,
        "summary": summary,
        "details": details
    }, ensure_ascii=False)

@tool
def elenco_tabelle_db() -> dict:
    """Usa questo strumento per ottenere l'elenco di tutte le tabelle presenti nel database.
    QUESTO STRUMENTO NON ACCETTA ALCUN PARAMETRO DI INPUT (lasciare gli argomenti vuoti {})."""
    debug_print("🔌 [TOOL PYTHON] Esecuzione di: elenco_tabelle_db()...")
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if not tables:
            receipt = "Nessuna tabella trovata nel database."
            return _make_tool_response(False, receipt, receipt, receipt)

        details = f"Tabelle trovate nel database: {tables}"
        summary = f"Sono presenti {len(tables)} tabelle: {tables}."
        receipt = "Elenco tabelle ottenuto correttamente."
        return _make_tool_response(True, receipt, summary, details)

    except Exception as e:
        error_message = f"Errore critico durante la lettura delle tabelle: {str(e)}"
        return _make_tool_response(False, error_message, error_message, error_message)

@tool
def Create_table(table_name: str, columns: List[Dict[str, Any]]) -> dict:
    """Usa questo strumento per creare una nuova tabella nel database.
    
    PARAMETRI:
    - table_name: Il nome della tabella (es. 'utenti').
    - columns: Una lista di dizionari Python contenente le colonne. 
      Ogni dizionario deve avere le chiavi chiare 'name' e 'type' (es. 'constraint' opzionale).
      Esempio: [{"name": "id", "type": "INTEGER", "constraint": "PRIMARY KEY"}]
    """
    debug_print(f"🔌 [TOOL PYTHON] Esecuzione di: Create_table(table_name: {table_name})...")
    
    try:
        if not isinstance(columns, list) or not columns:
            receipt = "Il parametro 'columns' deve essere una lista valida e non vuota di colonne."
            return _make_tool_response(False, receipt, receipt, receipt)

        clm = []
        for col in columns:
            if not isinstance(col, dict):
                receipt = "Ogni elemento di 'columns' deve essere un dizionario valido."
                return _make_tool_response(False, receipt, receipt, receipt)
            name = col.get("name")
            col_type = col.get("type")
            constraint = col.get("constraint", "")
            
            if not name or not col_type:
                receipt = "Ogni colonna deve contenere obbligatoriamente i campi 'name' e 'type'."
                return _make_tool_response(False, receipt, receipt, receipt)
                
            clm.append(f"{name} {col_type} {constraint}".strip())

        column_definitions = ", ".join(clm)

        with engine.connect() as connection:
            final_query = f"CREATE TABLE {table_name} ({column_definitions});"
            debug_print(f"   [LOG TOOL] Query SQL generata: {final_query}\n")

            with connection.begin():
                connection.execute(text(final_query))

        summary = f"Creata tabella '{table_name}' con colonne: {', '.join([col['name'] for col in columns])}."
        details = f"Eseguita query SQL: {final_query}"
        receipt = f"Tabella '{table_name}' creata con successo."
        return _make_tool_response(True, receipt, summary, details)
        
    except Exception as e:    
        error_message = f"Errore critico durante la creazione della tabella: {str(e)}"
        return _make_tool_response(False, error_message, error_message, error_message)

    
@tool
def Find_table_info(table_name: str) -> dict:
    """Usa questo strumento per ottenere la struttura dettagliata di una tabella specifica (colonne, tipi di dato e vincoli)."""
    debug_print(f"🔌 [TOOL PYTHON] Esecuzione di: Find_table_info(table_name: {table_name})...")

    try:
        with engine.connect() as connection: 
            inspector = inspect(connection)
            
            existing_tables = inspector.get_table_names()
            if table_name not in existing_tables:
                receipt = f"Tabella '{table_name}' non trovata nel database."
                details = f"Tabelle disponibili: {existing_tables}"
                return _make_tool_response(False, receipt, receipt, details)

            columns_info = inspector.get_columns(table_name)
            if not columns_info:
                receipt = f"Nessuna colonna trovata per la tabella '{table_name}'."
                return _make_tool_response(False, receipt, receipt, receipt)

            details = f"Informazioni sulla tabella '{table_name}':\n"
            for col in columns_info:
                nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                details += f"- Campo: {col['name']}, Tipo: {col['type']}, Opzioni: {nullable}\n"
            summary = f"La tabella '{table_name}' contiene {len(columns_info)} colonne."
            receipt = f"Struttura della tabella '{table_name}' recuperata correttamente."
            return _make_tool_response(True, receipt, summary, details)
            
    except Exception as e:
        error_message = f"Errore critico durante la lettura della struttura della tabella: {str(e)}"
        return _make_tool_response(False, error_message, error_message, error_message)
    
@tool
def Cerca_su_Web(query: str) -> dict:
    """Usa questo strumento SOLO quando l'utente chiede informazioni in tempo reale, 
    notizie recenti, fatti di attualità o concetti generali non presenti nel database aziendale.
    Il parametro 'query' deve essere la stringa testuale esatta da inviare al motore di ricerca (es. 'Meteo Milano oggi')."""
    debug_print(f"🌐 [TOOL WEB] Ricerca online in corso per la query: '{query}'...")
    
    try:
        risultati = web_search.invoke({"query": query})
        
        if not risultati:
            receipt = "La ricerca online non ha prodotto nessun risultato utile per questa query."
            return _make_tool_response(False, receipt, receipt, receipt)
            
        details = f"Risultati estratti dal Web per '{query}':\n\n{risultati}"
        summary = f"Ricerca web completata per '{query}'."
        receipt = "Ricerca web eseguita con successo."
        return _make_tool_response(True, receipt, summary, details)
        
    except Exception as e:
        error_message = f"Errore durante la ricerca web: {str(e)}"
        return _make_tool_response(False, error_message, error_message, error_message)
