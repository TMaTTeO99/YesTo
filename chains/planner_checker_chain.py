from langchain_core.prompts import ChatPromptTemplate
from config import llm
from pydantic import BaseModel, Field

class PlannerCheckerSchema(BaseModel):

    result: bool = Field(description="The classification of the plan." 
                        "True if and only if the plan procuded by planner is coherent with user request.\n"\
                        "False if the plan produced by planner has nothing to do with user request.\n")

_prompt = ChatPromptTemplate.from_messages([
    
    ("system", (
        "You are a specialized plan checker. Your aim is to check if the plan created by planner is coherent with user request.\n"
        "RULES:\n"
        "1. The plan must be coherent with user request, if the plan try to resolve other things your result is 'False'\n"
        "2. The result bust has a value. True or False"
    )),
    ("user", (
        "User request: {original_text}\n\n"
        "Plan: {plan}"
    ))
])

planner_checker_chain = _prompt | llm.with_structured_output(PlannerCheckerSchema)