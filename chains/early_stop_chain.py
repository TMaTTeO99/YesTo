from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm


class EarlyStopSchema(BaseModel):
    obiettivo_raggiunto: bool = Field(
        description="TRUE if the user's final goal has already been fully achieved "
                    "with the tasks already executed, making it pointless to continue with the remaining tasks. "
                    "FALSE if the remaining tasks are still needed to complete the goal."
    )
    motivazione: str = Field(
        description="Brief explanation of why the goal is already achieved or why it's necessary to continue."
    )


_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the Optimizer of an advanced agentic system.\n"
        "You are called after every successfully executed task to decide whether the user's ORIGINAL REQUEST "
        "has already received a sufficient answer, making it pointless to continue.\n\n"
        "CORE RULE: ignore the plan. The plan is an initial estimate that may be "
        "overestimated. Focus ONLY on: does the user's question already have a concrete "
        "and satisfying answer in the results obtained so far?\n\n"
        "EXAMPLES:\n"
        "- User asks 'how much does X cost?' → if the results already contain a price → obiettivo_raggiunto=True\n"
        "- User asks 'create table Y' → if the table hasn't been created yet → False\n"
        "- User asks 'search for news about X' → if the web results contain news about X → True\n\n"
        "RULES:\n"
        "1. Set obiettivo_raggiunto=True if the answer to the original question is already present "
        "   in the results, even partially. You don't need to gather every possible detail.\n"
        "2. Set obiettivo_raggiunto=False ONLY if the central information or action "
        "   requested by the user is still missing (not accessory details).\n"
        "3. Remaining tasks in the plan are NOT sufficient reason to continue: the plan "
        "   may be overestimated relative to the user's actual need.\n"
        "4. CRITICAL EXCEPTION: if the results show a blocking popup, dialog, or screen "
        "   (e.g. cookie consent, login wall, GDPR) that prevents access to the real content, "
        "   set obiettivo_raggiunto=False even if the page loaded. "
        "   The content is not accessible until the block is removed.\n"
        "Generate the structured output strictly following these rules."
    )),
    ("user", (
        "ORIGINAL USER REQUEST: {original_text}\n\n"
        "RESULTS OBTAINED SO FAR:\n{past_steps_context}\n\n"
        "REMAINING TASKS IN THE PLAN (may be unnecessary):\n{remaining_tasks}"
    ))
])

early_stop_chain = _prompt | llm.with_structured_output(EarlyStopSchema)
