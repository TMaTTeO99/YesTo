import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from sqlalchemy import inspect, text
from config import get_engine, debug_print


def _make_tool_response(success: bool, receipt: str, summary: str, details: str) -> str:
    return json.dumps(
        {"success": success, "receipt": receipt, "summary": summary, "details": details},
        ensure_ascii=False,
    )


@tool
def elenco_tabelle_db() -> str:
    """Usa questo strumento per ottenere l'elenco di tutte le tabelle presenti nel database.
    QUESTO STRUMENTO NON ACCETTA ALCUN PARAMETRO DI INPUT (lasciare gli argomenti vuoti {})."""
    debug_print("🔌 [TOOL DB] elenco_tabelle_db()")

    try:
        inspector = inspect(get_engine())
        tables = inspector.get_table_names()
        if not tables:
            msg = "Nessuna tabella trovata nel database."
            return _make_tool_response(False, msg, msg, msg)

        details = f"Tabelle trovate nel database: {tables}"
        summary = f"Sono presenti {len(tables)} tabelle: {tables}."
        return _make_tool_response(True, "Elenco tabelle ottenuto correttamente.", summary, details)

    except Exception as e:
        msg = f"Errore critico durante la lettura delle tabelle: {e}"
        return _make_tool_response(False, msg, msg, msg)


@tool
def create_table(table_name: str, columns: List[Dict[str, Any]]) -> str:
    """Usa questo strumento per creare una nuova tabella nel database.

    PARAMETRI:
    - table_name: Il nome della tabella (es. 'utenti').
    - columns: Lista di dizionari con chiavi 'name', 'type', e opzionalmente 'constraint'.
      Esempio: [{"name": "id", "type": "INTEGER", "constraint": "PRIMARY KEY"}]
    """
    debug_print(f"🔌 [TOOL DB] create_table(table_name={table_name})")

    try:
        if not isinstance(columns, list) or not columns:
            msg = "Il parametro 'columns' deve essere una lista valida e non vuota."
            return _make_tool_response(False, msg, msg, msg)

        col_defs = []
        for col in columns:
            if not isinstance(col, dict):
                msg = "Ogni elemento di 'columns' deve essere un dizionario."
                return _make_tool_response(False, msg, msg, msg)
            name, col_type = col.get("name"), col.get("type")
            if not name or not col_type:
                msg = "Ogni colonna deve avere i campi 'name' e 'type'."
                return _make_tool_response(False, msg, msg, msg)
            col_defs.append(f"{name} {col_type} {col.get('constraint', '')}".strip())

        query = f"CREATE TABLE {table_name} ({', '.join(col_defs)});"
        debug_print(f"   [LOG DB] Query: {query}")

        with get_engine().connect() as conn:
            with conn.begin():
                conn.execute(text(query))

        summary = f"Creata tabella '{table_name}' con colonne: {', '.join(c['name'] for c in columns)}."
        return _make_tool_response(True, f"Tabella '{table_name}' creata con successo.", summary, f"Eseguita query: {query}")

    except Exception as e:
        msg = f"Errore critico durante la creazione della tabella: {e}"
        return _make_tool_response(False, msg, msg, msg)


@tool
def find_table_info(table_name: str) -> str:
    """Usa questo strumento per ottenere la struttura dettagliata di una tabella (colonne, tipi, vincoli)."""
    debug_print(f"🔌 [TOOL DB] find_table_info(table_name={table_name})")

    try:
        with get_engine().connect() as conn:
            inspector = inspect(conn)
            if table_name not in inspector.get_table_names():
                receipt = f"Tabella '{table_name}' non trovata nel database."
                details = f"Tabelle disponibili: {inspector.get_table_names()}"
                return _make_tool_response(False, receipt, receipt, details)

            cols = inspector.get_columns(table_name)
            if not cols:
                msg = f"Nessuna colonna trovata per la tabella '{table_name}'."
                return _make_tool_response(False, msg, msg, msg)

            details = f"Informazioni sulla tabella '{table_name}':\n"
            for col in cols:
                nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
                details += f"- Campo: {col['name']}, Tipo: {col['type']}, Opzioni: {nullable}\n"

            summary = f"La tabella '{table_name}' contiene {len(cols)} colonne."
            return _make_tool_response(True, f"Struttura di '{table_name}' recuperata.", summary, details)

    except Exception as e:
        msg = f"Errore critico durante la lettura della struttura della tabella: {e}"
        return _make_tool_response(False, msg, msg, msg)
