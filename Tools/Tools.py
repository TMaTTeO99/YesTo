import json
from langchain_core.tools import tool
from Shared.shared import engine, debug_print
from sqlalchemy import inspect, text

@tool
def elenco_tabelle_db() -> str:
    """Usa questo strumento per ottenere l'elenco di tutte le tabelle presenti nel database.
    QUESTO STRUMENTO NON ACCETTA ALCUN PARAMETRO DI INPUT (lasciare gli argomenti vuoti {})."""
    debug_print("🔌 [TOOL PYTHON] Esecuzione di: elenco_tabelle_db()...")
    
    try:
        # Ora engine è l'oggetto reale erogato dal Singleton
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if not tables:
            return "Nessuna tabella trouvata nel database."
        return f"Tabelle trovate nel database: {tables}"

    except Exception as e:
        return f"[ERROR]: Errore critico durante la lettura delle tabelle: {str(e)}"

    
@tool
def Create_table(table_name: str, columns_json: str) -> str:
    """Usa questo strumento per creare una nuova tabella nel database.
    
    PARAMETRI:
    - table_name: Il nome della tabella (es. 'utenti').
    - columns_json: Una stringa in formato JSON standard che rappresenta una lista di dizionari...
    """
    debug_print(f"🔌 [TOOL PYTHON] Esecuzione di: Create_table(table_name: {table_name})...")
    
    try:
        try:
            columns_list = json.loads(columns_json)
        except Exception as json_err:
            return f"[ERROR]: Il parametro 'columns_json' non è un JSON valido. Dettaglio: {str(json_err)}"

        clm = []
        for col in columns_list:
            name = col.get("name")
            col_type = col.get("type")
            constraint = col.get("constraint", "")
            
            if not name or not col_type:
                return "[ERROR]: Ogni colonna nel JSON deve avere obbligatoriamente i campi 'name' e 'type'."
                
            clm.append(f"{name} {col_type} {constraint}".strip())

        column_definitions = ", ".join(clm)

        with engine.connect() as connection:
            final_query = f"CREATE TABLE {table_name} ({column_definitions});"
            debug_print(f"   [LOG TOOL] Query SQL generata: {final_query}\n")

            with connection.begin():
                connection.execute(text(final_query))

        return f"Tabella '{table_name}' creata con successo."
        
    except Exception as e:    
        return f"[ERROR]: Errore critico durante la creazione della tabella: {str(e)}"

    
@tool
def Find_table_info(table_name: str) -> str:
    """Usa questo strumento per ottenere la struttura dettagliata di una tabella specifica (colonne, tipi di dato e vincoli)."""
    debug_print(f"🔌 [TOOL PYTHON] Esecuzione di: Find_table_info(table_name: {table_name})...")

    try:
        with engine.connect() as connection: 
            inspector = inspect(connection)
            
            existing_tables = inspector.get_table_names()
            if table_name not in existing_tables:
                return f"[ERROR]: Tabella '{table_name}' non trovata nel database. Tabelle disponibili: {existing_tables}"

            columns_info = inspector.get_columns(table_name)
            if not columns_info:
                return f"Nessuna colonna trovata per la tabella '{table_name}'."

            info_str = f"Informazioni sulla tabella '{table_name}':\n"
            for col in columns_info:
                nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                info_str += f"- Campo: {col['name']}, Tipo: {col['type']}, Opzioni: {nullable}\n"
            return info_str
            
    except Exception as e:
        return f"[ERROR]: Errore critico durante la lettura della struttura della tabella: {str(e)}"