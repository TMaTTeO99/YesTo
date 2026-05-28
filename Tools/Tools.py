from langchain_core.tools import tool
from pydantic import BaseModel, Field
from Shared.shared import getDBAddress
from sqlalchemy import create_engine, inspect
from sqlalchemy import text

engine = create_engine(getDBAddress())

class ColumnDefinition(BaseModel):
    name: str = Field(description="Il nome della colonna da creare. Ad esempio: 'id', 'nome', 'data_creazione', ...")
    type: str = Field(description="Il tipo di dato della colonna. Ad esempio: 'Integer', 'VARCHAR(255)', ...")
    constraint: str = Field(description="Eventuali vincoli per la colonna. Ad esempio: 'PRIMARY KEY', 'NOT NULL', 'UNIQUE', ... Se non ci sono vincoli, lasciare vuoto.")

@tool
def elenco_tabelle_db() -> str:
    """Usa questo strumento per ottenere l'elenco di tutte le tabelle presenti nel database aziendale."""
    print("🔌 [TOOL PYTHON] Esecuzione di: elenco_tabelle_db()...")
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if not tables:
            return "Nessuna tabella trovata nel database."
        return f"Tabelle trovate nel database: {tables}"

    except Exception as e:
        return f"Errore critico durante la connessione al database: {str(e)}"
    
@tool
def Create_table(table_name: str, columns: list[ColumnDefinition]) -> str:
                 
    """Usa questo strumento per creare una tabella all'interno del database con le specifiche date. 
    Il parametro 'columns' è una lista di oggetti colonna, dove ogni elemento definisce 
    il nome della colonna, il tipo di dato SQL e gli eventuali vincoli (es. NOT NULL, PRIMARY KEY)."""
    print("🔌 [TOOL PYTHON] Esecuzione di: Create_table(table_name: {table_name})...")
    
    try:

        clm = []
        for col in columns:
            clm.append(f"{col.name} {col.type} {col.constraint if col.constraint else ''}".strip())

        column_definitions = ", ".join(clm)

        with engine.connect() as connection:

            final_query = f"CREATE TABLE {table_name} ({column_definitions});"
            print(f"   [LOG TOOL] Query SQL generata: {final_query}\n")

            with connection.begin():
                connection.execute(text(final_query))

        return f"Tabella '{table_name}' creata con successo con le colonne: {columns}"
    except Exception as e:    
        return f"Errore critico durante la connessione al database: {str(e)}"
    
@tool
def Find_table_info(table_name: str) -> str:
    """ Usa questo strumento per ottenere informazioni dettagliate su una tabella specifica del database, inclusi i nomi delle colonne, i tipi di dati e eventuali vincoli. """
    print(f"🔌 [TOOL PYTHON] Esecuzione di: Find_table_info(table_name: {table_name})...")

    try:
        with engine.connect() as connection: 
            inspector = inspect(connection)
            if table_name not in inspector.get_table_names():
                return f"Tabella '{table_name}' non trovata nel database."

            columns_info = inspector.get_columns(table_name)
            if not columns_info:
                return f"Nessuna colonna trovata per la tabella '{table_name}'."

            info_str = f"Informazioni sulla tabella '{table_name}':\n"
            for col in columns_info:
                info_str += f"- Colonna: {col['name']}, Tipo: {col['type']}, Nullable: {col['nullable']}\n"
            return info_str
    except Exception as e:
        return f"Errore critico durante la connessione al database: {str(e)}"