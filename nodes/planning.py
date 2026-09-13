from typing import Literal
from state import PlanningState
from chains.planner_chain import planner_chain
from chains.summary_chain import summary_chain
from chains.early_stop_chain import early_stop_chain
from chains.tools_output_chain import tools_output_chain
from chains.critique_plan_result import critique_plan_result_chain
from config import debug_print
from nodes.node_utils import _call_replanner_node
from chains.planner_checker_chain import planner_checker_chain
import json

MAX_PAST_STEP_BUFFER = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compact_past_steps(past_steps: list[dict], old_steps_summary: str = None) -> list[dict]:
    """Summarize old steps when the buffer exceeds MAX_PAST_STEP_BUFFER to save context."""
    if len(past_steps) <= MAX_PAST_STEP_BUFFER:
        return past_steps

    recent = past_steps[-MAX_PAST_STEP_BUFFER:]
    old = past_steps[:-MAX_PAST_STEP_BUFFER]
    summarized = summary_chain.invoke({
        "existing_summary": old_steps_summary or "",
        "input_text": "\n".join(f"Task: {s['task']}\nRisultato: {s['receipt']}" for s in old),
    })
    return [{"task": "Passi precedenti compressi", "receipt": summarized, "summary": summarized, "details": summarized}] + recent


def _build_steps_context(past_steps: list[dict], old_steps_summary: str = None) -> str:
    compacted = _compact_past_steps(past_steps, old_steps_summary)
    return "\n".join(f"Task: {s['task']}\nRicevuta: {s['details']}" for s in compacted)


def _build_final_context(past_steps: list[dict]) -> str:
    return "\n---\n".join(
        f"Operazione pianificata: {s['task']}\nSintesi: {s['summary']}\nDettagli: {s['details']}"
        for s in past_steps
    )


def _normalize_tool_result(tool_name: str, raw) -> dict:
    try:
        data = json.loads(str(raw))
        if isinstance(data, dict):
            return {
                "tool_name": tool_name,
                "success": bool(data.get("success", True)),
                "receipt": str(data.get("receipt", "")),
                "summary": str(data.get("summary", "")),
                "details": str(data.get("details", "")),
            }
    except Exception:
        pass
    text = str(raw)
    return {"tool_name": tool_name, "success": True, "receipt": text, "summary": text, "details": text}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planning_init_node(state: PlanningState):

    debug_print("📋 [PLANNING] Nodo INIT - Generazione del piano di lavoro...")
    result = planner_chain.invoke({"original_text": state["original_text"], "past_conversation" : state["past_conversation"]})
    debug_print(f"   [LOG PLANNER] Task pianificati: {result.sotto_task}")
    debug_print(f"   [LOG PLANNER] Raggionamento: {result.ragionamento}")
    return {"plan": result.sotto_task, "past_steps": [], "response": ""}



def replanner_node(state: PlanningState):
    debug_print("🧠 [PLANNING] Nodo RE-PLANNER - Valutazione dello stato del piano...")

    current_plan = list(state.get("plan", []))
    past_steps = state.get("past_steps", [])

    if not past_steps:
        return {}

    last_step = past_steps[-1]
    last_failed = not last_step.get("success", True)
    debug_print(f"   [LOG RE-PLANNER] Ultimo risultato: {str(last_step['details'])}")

    if not last_failed and current_plan:
        debug_print(f"   [LOG RE-PLANNER] Tutto procede bene. Task rimanenti: {len(current_plan)}")
        context = _build_steps_context(past_steps, state.get("old_steps_summary"))
        remaining = "\n".join(f"- {t}" for t in current_plan)

        early = early_stop_chain.invoke({
            "original_text": state["original_text"],
            "past_steps_context": context,
            "remaining_tasks": remaining,
        })
        debug_print(f"[LOG EARLY-STOP] obiettivo_raggiunto={early.obiettivo_raggiunto} | {early.motivazione}")

        if early.obiettivo_raggiunto:
            debug_print("   🛑 [EARLY STOP] Obiettivo già raggiunto. Salto i task rimanenti.")
            return {"plan": []}

        return {"old_steps_summary" : context}


    if not last_failed and not current_plan:

        context = _build_steps_context(past_steps, state.get("old_steps_summary"))
        result_critique = critique_plan_result_chain.invoke({ "original_text" : state.get("original_text"), "past_steps_context" : context})

        if result_critique.approvato:
            debug_print("   🛑 [GUARDRAIL] Obiettivo raggiunto e piano esaurito. Uscita forzata.")
            return {"plan": []}
        else:
            debug_print("   ⚠️ [LOG RE-PLANNER] Errore rilevato. Interpello LLM per ri-pianificare...")
            return _call_replanner_node(state, context)

    if last_failed:

        debug_print("   ⚠️ [LOG RE-PLANNER] Errore rilevato. Interpello LLM per ri-pianificare...")
        context = _build_steps_context(past_steps, state.get("old_steps_summary"))

        return _call_replanner_node(state, context)

def planner_checker(state: PlanningState):

    debug_print(f"[PLANNING CHECKER]: Check the plan produced by planner")
    resutl_plan_checker = planner_checker_chain.invoke({"original_text" : state["original_text"], "plan" : state["plan"]})

    if resutl_plan_checker.result:
        return {"past_plan" : "good_plan"}
    
    return {"past_plan" : "bad_plan"}

def planner_checker_result_router(state: PlanningState):
    
    return state["past_plan"]

def merge_tools_output_node(state: PlanningState):
    debug_print("🎯 [PLANNING] Nodo MERGE - Confezionamento risposta finale...")
    past_steps = state.get("past_steps", [])

    if not past_steps:
        msg = "Mi dispiace, si è verificato un problema durante l'esecuzione delle operazioni."
        return {"response": msg}

    context = _build_final_context(past_steps)
    final = tools_output_chain.invoke({"tools_outputs": context, "original_text": state["original_text"]})
    debug_print("   [LOG MERGE] Risposta completata con successo.")
    return {"plan": [], "response": final}


def planning_routing_logic(state: PlanningState) -> Literal["continue_execution", "go_to_final_response"]:
    return "continue_execution" if state.get("plan") else "go_to_final_response"
