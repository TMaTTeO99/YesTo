from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm


class CritiqueSchema(BaseModel):
    approvato: bool = Field(
        description="TRUE if the answer is correct and resolves the question. FALSE if it contains errors or hallucinations."
    )
    punti_da_correggere: List[str] = Field(
        description="If approvato is TRUE, this list MUST be strictly empty []. "
                    "If approvato is FALSE, list the weak points found here."
    )


_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the system's Quality Control Inspector (Critic).\n"
        "Your only job is to validate whether the agent's answer is correct and not fabricated.\n\n"
        "EVALUATION RULES:\n"
        "1. Courtesy messages, greetings, or small talk -> approvato=True always.\n"
        "2. Static general-knowledge answers (capitals, definitions, history...) -> "
        "   approvato=True if the answer is correct, False if it contains factual errors.\n"
        "3. WATCH FOR HALLUCINATIONS: if the request concerns company data, tables, operational data, OR the "
        "   specific content of a document/book/file (chapters, indexes, lists of items it supposedly contains), "
        "   and the agent's answer states specific facts WITHOUT them coming from a real tool call/search result "
        "   visible in the conversation, that is a hallucination -> approvato=False. In punti_da_correggere, say "
        "   explicitly that the agent must use its tools (DB / web / knowledge-base search) to get real data "
        "   instead of inventing an answer — do NOT just ask for 'a better answer' with the same content style, "
        "   since that only encourages fabricating a plausible-looking but ungrounded answer.\n"
        "4. If the agent's answer honestly says it found no information (e.g. after a real tool search failed), "
        "   that is NOT a defect by itself — do not penalize an honest 'not found' unless the user explicitly asked "
        "   the agent to try again with a tool it hasn't used yet.\n"
        "5. CONSISTENCY RULE: if approvato=False you must always provide at least one point in "
        "   punti_da_correggere. If you cannot find any concrete errors, set approvato=True.\n"
    )),
    ("user", "User request: {original_text}\n\nAgent answer: {risposta}")
])

critique_chain = _prompt | llm.with_structured_output(CritiqueSchema)
