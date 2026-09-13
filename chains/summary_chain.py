from langchain_core.prompts import ChatPromptTemplate
from config import llm
from pydantic import BaseModel, Field

class SummarySchema(BaseModel):
    summary: str = Field(
        description="Summarized text of the conversation between the user and the agent. Must be concise, clear, and preserve the essential context.\n"
    )


_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a highly efficient summarization assistant. You will receive an existing historical summary and a new block of text to compress.\n"
        "Your job is to return a single very short paragraph, dense and faithful to the content, keeping only relevant information, goals, and action items.\n"
        "Do not add superfluous explanations, do not invent information, and do not insert dates or other metadata."
    )),
    ("user", (
        "Previous historical summary:\n{existing_summary}\n\n"
        "Text to summarize:\n{input_text}\n\n"
        "Return a compact paragraph that summarizes the provided material and preserves the main sense of the conversation or completed steps."
    ))
])

summary_chain = _prompt | llm.with_structured_output(SummarySchema)
