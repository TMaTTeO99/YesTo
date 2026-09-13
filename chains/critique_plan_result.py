from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm


class CritiqueSchema(BaseModel):
    approvato: bool = Field(
        description="True if the data produced by the agent genuinely provides the information requested by the user. False otherwise."
    )

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert text analyst. Your job is to evaluate whether the data produced by the agent genuinely provides the information the user requested.\n"
        "If the data produced by the agent is sufficient and answers the user's request, set approvato=True. If the data produced by the agent is insufficient or does not answer the user's request, set approvato=False.\n"
        "RULES:\n"
        "- The data must be complete and accurate.\n"
        "- The data must be relevant to the user's request.\n"
    )),
    ("user", (
        "User request: {original_text}\n\n"
        "Data produced by the agent: {past_steps_context}\n\n"
        "Evaluate whether the data produced by the agent genuinely answers the user's request and set the 'approvato' field accordingly."
    ))
])

critique_plan_result_chain = _prompt | llm.with_structured_output(CritiqueSchema)
