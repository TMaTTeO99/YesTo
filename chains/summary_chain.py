from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm


_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei un assistente di sintesi molto efficiente. Riceverai un riassunto storico esistente e un nuovo blocco di testo da comprimere.\n"
        "Il tuo compito è restituire un solo paragrafo brevissimo, denso e fedele al contenuto, mantenendo solo le informazioni rilevanti, gli obiettivi e gli elementi azione.\n"
        "Non aggiungere spiegazioni superflue, non inventare informazioni e non inserire date o altri metadati."
    )),
    ("user", (
        "Riassunto storico precedente:\n{existing_summary}\n\n"
        "Testo da sintetizzare:\n{input_text}\n\n"
        "Restituisci un paragrafo compatto che sintetizza il materiale fornito e conserva il senso principale della conversazione o dei passi completati."
    ))
])

summary_chain = _prompt | llm | StrOutputParser()
