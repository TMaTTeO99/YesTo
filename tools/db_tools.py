import json
import re
from typing import List, Dict, Any
from langchain_core.tools import tool
from sqlalchemy import inspect, text
from config import get_engine, debug_print
from tools.utils import _make_tool_response

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_CONSTRAINTS = {
    "", "PRIMARY KEY", "NOT NULL", "UNIQUE", "PRIMARY KEY NOT NULL", "NOT NULL UNIQUE",
}
_ALLOWED_TYPES = {
    "INTEGER", "BIGINT", "SMALLINT", "SERIAL", "BIGSERIAL", "TEXT", "VARCHAR", "CHAR",
    "BOOLEAN", "DATE", "TIMESTAMP", "TIMESTAMPTZ", "NUMERIC", "REAL", "DOUBLE PRECISION",
    "UUID", "JSON", "JSONB",
}


def _validate_identifier(value: str) -> bool:
    return bool(_IDENTIFIER_RE.match(value or ""))


def _validate_type(col_type: str) -> bool:
    base = re.match(r"^[A-Za-z ]+", col_type or "")
    if not base:
        return False
    normalized = base.group(0).strip().upper()
    if normalized not in _ALLOWED_TYPES:
        return False
    # allow a trailing size spec like VARCHAR(255)
    remainder = (col_type or "")[base.end():]
    return remainder == "" or bool(re.match(r"^\(\d+(,\d+)?\)$", remainder))

@tool
def elenco_tabelle_db() -> str:
    """Use this tool to get the list of all tables present in the database.
    THIS TOOL DOES NOT ACCEPT ANY INPUT PARAMETER (leave the arguments empty {})."""
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
    """Use this tool to create a new table in the database.

    PARAMETERS:
    - table_name: The name of the table (e.g. 'utenti').
    - columns: List of dictionaries with keys 'name', 'type', and optionally 'constraint'.
      Example: [{"name": "id", "type": "INTEGER", "constraint": "PRIMARY KEY"}]
    """
    debug_print(f"🔌 [TOOL DB] create_table(table_name={table_name})")

    try:
        if not _validate_identifier(table_name):
            msg = f"Nome tabella non valido: '{table_name}'. Sono ammessi solo lettere, numeri e underscore."
            return _make_tool_response(False, msg, msg, msg)

        if not isinstance(columns, list) or not columns:
            msg = "Il parametro 'columns' deve essere una lista valida e non vuota."
            return _make_tool_response(False, msg, msg, msg)

        col_defs = []
        for col in columns:
            if not isinstance(col, dict):
                msg = "Ogni elemento di 'columns' deve essere un dizionario."
                return _make_tool_response(False, msg, msg, msg)
            name, col_type = col.get("name"), col.get("type")
            constraint = col.get("constraint", "") or ""
            if not name or not col_type:
                msg = "Ogni colonna deve avere i campi 'name' e 'type'."
                return _make_tool_response(False, msg, msg, msg)
            if not _validate_identifier(name):
                msg = f"Nome colonna non valido: '{name}'. Sono ammessi solo lettere, numeri e underscore."
                return _make_tool_response(False, msg, msg, msg)
            if not _validate_type(col_type):
                msg = f"Tipo colonna non ammesso: '{col_type}'."
                return _make_tool_response(False, msg, msg, msg)
            if constraint.strip().upper() not in _ALLOWED_CONSTRAINTS:
                msg = f"Vincolo non ammesso: '{constraint}'."
                return _make_tool_response(False, msg, msg, msg)
            col_defs.append(f"{name} {col_type} {constraint}".strip())

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
    """Use this tool to get the detailed structure of a table (columns, types, constraints)."""
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
