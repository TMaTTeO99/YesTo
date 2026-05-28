from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from Shared.shared import llm

class RouterClassification(str, Enum):
    DOMANDA = "domanda"
    RECLAMO = "reclamo"
    SALUTO = "saluto"
    INCOMPRENSIBILE = "incomprensibile"

class RouterSchema(BaseModel):
    classification: RouterClassification = Field(
        description="La classificazione della richiesta dell'utente. Scegli 'domanda' per quesiti, ordini o comandi operativi."
    )
    justification: str = Field(
        description="Una brevissima spiegazione (una frase) del perché hai scelto questa classificazione. Aiuta il modello a ragionare."
    )

structured_router_llm = llm.with_structured_output(RouterSchema)

coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei il Router Principale di un sistema aziendale avanzato. Il tuo unico scopo è classificare l'input dell'utente.\n\n"
        "CATEGORIE AMMESSE:\n"
        "- 'domanda': Richieste di dati, creazione tabelle, analisi, query, compiti operativi o domande di cultura generale.\n"
        "- 'saluto': Cortesia, saluti, ringraziamenti.\n"
        "- 'reclamo': Espressioni di insoddisfazione, rabbia, bug segnalati con frustrazione.\n"
        "- 'incomprensibile': Testo sconnesso, lettere a caso.\n\n"
        
        "ESEMPI DI CLASSIFICAZIONE (FEW-SHOT):\n"
        "1. User: 'Ciao, come stai?' -> classification: 'saluto'\n"
        "2. User: 'Crea una tabella chiamata fornitori con id e nome' -> classification: 'domanda'\n"
        "3. User: 'Il sistema fa schifo, ieri ho perso tutti i dati' -> classification: 'reclamo'\n"
        "4. User: 'Mostrami le tabelle del db' -> classification: 'domanda'\n"
        "5. User: 'asdffg123' -> classification: 'incomprensibile'\n\n"
        
        "Analizza l'input dell'utente, compila la giustificazione e seleziona la classificazione corretta."
    )),
    ("user", "Input utente da classificare: {original_text}")
])

coordinator_router_chain = coordinator_router_prompt | structured_router_llm