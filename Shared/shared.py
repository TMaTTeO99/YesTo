import os
import threading
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from sqlalchemy import create_engine

load_dotenv()

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class AgentResourcesSingleton:
    _instance = None
    _lock = threading.Lock()  

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AgentResourcesSingleton, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        debug_print("🧠 [SINGLETON] Inizializzazione della risorsa LLM...")
        self._llm = ChatOllama(model="llama3.1", temperature=0)
        
        # Configurazione parametri DB
        db_user = os.getenv("DB_USER", "postgres")
        db_passwd = os.getenv("DB_PASSWD", "")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT")
        if not db_port or db_port == "None":
            db_port = "5432"
        db_name = os.getenv("DB_NAME", "postgres")
        
        self._db_address = f"postgresql://{db_user}:{db_passwd}@{db_host}:{db_port}/{db_name}"
        
        self._engine = None
        
        self._initialized = True
        debug_print("✅ [SINGLETON] Risorse inizializzate con successo.")

    @property
    def llm(self):
        return self._llm

    @property
    def db_address(self):
        return self._db_address

    @property
    def engine(self):
        """Punto di accesso globale all'engine del database, creato in modo Lazy."""
        if self._engine is None:
            debug_print("🔌 [SINGLETON] Creazione del pool di connessioni (SQLAlchemy Engine)...")
            self._engine = create_engine(self._db_address)
        return self._engine


_resources = AgentResourcesSingleton()

llm = _resources.llm
def getDBAddress() -> str:
    return _resources.db_address

engine = _resources.engine