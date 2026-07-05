import os
import threading
import whisper
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from sqlalchemy import create_engine

load_dotenv()

DEBUG = os.getenv("DEBUG", "False").lower() == "true"


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class _AgentResources:
    """Thread-safe singleton holding all shared expensive resources."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        debug_print("🧠 [SINGLETON] Inizializzazione della risorsa LLM...")

        # it can be substituted with any other LLM.
        self._llm = ChatOllama(model="llama3.1")
        
        # model to record and transcribe audio.
        self._whisper_model = whisper.load_model("base")
        
        self._web_search = None
        self._engine = None

        db_user = os.getenv("DB_USER", "postgres")
        db_passwd = os.getenv("DB_PASSWD", "")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT") or "5432"
        db_name = os.getenv("DB_NAME", "postgres")
        self._db_address = f"postgresql://{db_user}:{db_passwd}@{db_host}:{db_port}/{db_name}"

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
        if self._engine is None:
            debug_print("🔌 [SINGLETON] Creazione del pool di connessioni (SQLAlchemy Engine)...")
            self._engine = create_engine(self._db_address)
        return self._engine

    @property
    def web_search(self):
        if self._web_search is None:
            debug_print("🌐 [SINGLETON] Istanziazione del client DuckDuckGo Search...")
            self._web_search = DuckDuckGoSearchRun()
        return self._web_search
    
    @property
    def whisper_model(self):
        return self._whisper_model


_resources = _AgentResources()

# Module-level shortcuts
llm = _resources.llm
whisper_model = _resources.whisper_model

def get_engine():
    return _resources.engine


def get_web_search():
    return _resources.web_search


def get_db_address() -> str:
    return _resources.db_address


def init_rag():
    """Inizializza il vector store RAG. Da chiamare dopo load_dotenv()."""
    from memory.vector_store import init_vector_stores
    init_vector_stores(get_db_address())
