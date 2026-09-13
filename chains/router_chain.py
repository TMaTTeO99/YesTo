from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from typing import Literal


class RouterClassification(str, Enum):
    DOMANDA = "domanda"
    INCOMPRENSIBILE = "incomprensibile"


class RouterSchema(BaseModel):
    classification: RouterClassification = Field(
        description="The classification of the user's request.\n" \
        "Available options:\n" \
        "domanda: If the user ask something\n"\
        "incomprensibile: If the user typed something that seems with no sense\n"
    )
    justification: str = Field(
        description="A very brief explanation (one sentence) of why you chose this classification."
    )


_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the Main Router of an advanced company system. Your only purpose is to classify the user's input.\n"
        "Also use previous conversation messages to better understand the context.\n\n"
        "ALLOWED CATEGORIES:\n"
        "- 'domanda': Requests for data, table creation, analysis, queries, operational tasks, general-knowledge "
        "- 'incomprensibile': ONLY disconnected/meaningless text (random letters, gibberish) with no coherent intent "
        "  and no connection to the previous conversation.\n\n"
        "CLASSIFICATION EXAMPLES (FEW-SHOT):\n"
        "2. User: 'Crea una tabella chiamata fornitori con id e nome' -> classification: 'domanda'\n"
        "4. User: 'Mostrami le tabelle del db' -> classification: 'domanda'\n"
        "5. User: 'asdffg123' -> classification: 'incomprensibile'\n"
        "6. User: 'Non cercare online, cerca direttamente nella tua base knowledge' -> classification: 'domanda' "
        "  (it's a correction to a previous request, not nonsense)\n\n"
        "Analyze the user's input, fill in the justification, and select the correct classification.\n\n"
        "BEHAVIOUR RULES:\n"
        "1. Instead to invent something use your tools to answare the question\n"
        "2. If the user tell you to retry to do something i must do it so probably the request needs a tool call\n"

    )),
    ("user", "User input to classify: {original_text}\n"
             "Previous messages: {message_summary}\n")
])

router_chain = _prompt | llm.with_structured_output(RouterSchema)
