from typing import Literal
from state import PlanningState
from chains.planner_chain import planner_chain, replanner_chain
from chains.executor_chain import executor_chain, tools_map
from chains.summary_chain import summary_chain
from chains.early_stop_chain import early_stop_chain
from chains.tools_output_chain import tools_output_chain
from config import debug_print
import json

MAX_PAST_STEP_BUFFER = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compact_past_steps(past_steps: list[dict]) -> list[dict]:
    """Summarize old steps when the buffer exceeds MAX_PAST_STEP_BUFFER to save context."""
    if len(past_steps) <= MAX_PAST_STEP_BUFFER:
        return past_steps

    recent = past_steps[-MAX_PAST_STEP_BUFFER:]
    old = past_steps[:-MAX_PAST_STEP_BUFFER]
    summarized = summary_chain.invoke({
        "existing_summary": "",
        "input_text": "\n".join(f"Task: {s['task']}\nRisultato: {s['details']}" for s in old),
    })
    return [{"task": "Passi precedenti compressi", "receipt": summarized, "summary": summarized, "details": summarized}] + recent


def _build_steps_context(past_steps: list[dict]) -> str:
    compacted = _compact_past_steps(past_steps)
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
                "receipt": str(data.get("receipt", "")),
                "summary": str(data.get("summary", "")),
                "details": str(data.get("details", "")),
            }
    except Exception:
        pass
    text = str(raw)
    return {"tool_name": tool_name, "receipt": text, "summary": text, "details": text}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planning_init_node(state: PlanningState):
    debug_print("📋 [PLANNING] Nodo INIT - Generazione del piano di lavoro...")
    result = planner_chain.invoke({"original_text": state["original_text"]})
    debug_print(f"   [LOG PLANNER] Task pianificati: {result.sotto_task}")
    debug_print(f"   [LOG PLANNER] Raggionamento: {result.ragionamento}")
    return {"plan": result.sotto_task, "past_steps": [], "response": ""}


def execution_node(state: PlanningState):
    current_plan = list(state.get("plan", []))
    if not current_plan:
        return {"plan": []}

    task = current_plan.pop(0)
    debug_print(f"🎯 [PLANNING] Nodo EXECUTION - Task attuale: '{task}'")

    past_steps = state.get("past_steps", [])
    context = "".join(f"- Task: {s['task']} -> Risultato: {s['details']}\n" for s in past_steps) or "Nessun task eseguito in precedenza."

    result = executor_chain.invoke({"current_task": task, "past_steps_context": context})

    new_steps = list(past_steps)

    if hasattr(result, "tool_calls") and result.tool_calls:
        outputs = []
        for call in result.tool_calls:
            name = call["name"]
            args = call.get("args", {})
            debug_print(f"   [LOG EXECUTOR] Chiamata tool: {name} con args {args}")

            if name in tools_map:
                raw = tools_map[name].invoke(args)
                normalized = _normalize_tool_result(name, raw)
                debug_print(f"   [LOG EXECUTOR] Risultato '{name}': {normalized['receipt']}")
                outputs.append(f"[{name}]: {normalized['details']}")
                new_steps.append({"task": task, **normalized})
            else:
                err = f"[ERROR]: tool '{name}' non riconosciuto."
                outputs.append(err)
                new_steps.append({"task": task, "tool_name": name, "receipt": err, "summary": err, "details": err})
    else:
        text = result.content if hasattr(result, "content") else str(result)
        new_steps.append({"task": task, "tool_name": "no_tool", "receipt": text, "summary": text, "details": text})

    debug_print("   [LOG EXECUTOR] Task completato.")
    return {"plan": current_plan, "past_steps": new_steps}


def replanner_node(state: PlanningState):
    debug_print("🧠 [PLANNING] Nodo RE-PLANNER - Valutazione dello stato del piano...")

    current_plan = list(state.get("plan", []))
    past_steps = state.get("past_steps", [])

    if not past_steps:
        return {}

    last_receipt = str(past_steps[-1]["receipt"])
    debug_print(f"   [LOG RE-PLANNER] Ultimo risultato: {str(past_steps[-1]['details'])}")

    if "[ERROR]" not in last_receipt and not current_plan:
        debug_print("   🛑 [GUARDRAIL] Obiettivo raggiunto e piano esaurito. Uscita forzata.")
        return {"plan": []}

    if "[ERROR]" not in last_receipt and current_plan:
        debug_print(f"   [LOG RE-PLANNER] Tutto procede bene. Task rimanenti: {len(current_plan)}")
        context = _build_steps_context(past_steps)
        remaining = "\n".join(f"- {t}" for t in current_plan)

        early = early_stop_chain.invoke({
            "original_text": state["original_text"],
            "past_steps_context": context,
            "remaining_tasks": remaining,
        })
        debug_print(f"   [LOG EARLY-STOP] obiettivo_raggiunto={early.obiettivo_raggiunto} | {early.motivazione}")

        if early.obiettivo_raggiunto:
            debug_print("   🛑 [EARLY STOP] Obiettivo già raggiunto. Salto i task rimanenti.")
            return {"plan": []}

        return {}

    debug_print("   ⚠️ [LOG RE-PLANNER] Errore rilevato. Interpello LLM per ri-pianificare...")
    context = _build_steps_context(past_steps)
    res = replanner_chain.invoke({"original_text": state["original_text"], "past_steps_context": context})

    if res.stop:
        debug_print("   [LOG RE-PLANNER] LLM ha confermato di interrompere.")
        return {"plan": []}

    new_plan = res.new_plan or []
    debug_print(f"   [LOG RE-PLANNER] Nuovo piano: {new_plan}")
    return {"plan": new_plan}


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
