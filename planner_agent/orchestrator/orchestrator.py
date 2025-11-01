# orchestrator.py
# ---------- UPDATED (AGENTIC FLOW) ----------
# Orchestrates the 4-stage workflow and loops between Transport Retrieval and Plan Validation/Repair.
import logging
import time
from typing import Dict, Any
from planner_agent.agent.transport import TransportAdapter, attach_transport_options
from planner_agent.planner_core.core import score_candidates, shortlist, assign_to_days, explain, _safe_get, \
    _pace_minutes, _minutes_for_item  # your core heuristics
from planner_agent.tools.config import MAX_AGENT_ITERATIONS, Retrieval_Agent_Folder
from planner_agent.tools.helper import aggregate_budget_range, _lunch_minutes
from planner_agent.agent.planner_agent import PlannerAgent, CrewAIAdapter as PlannerCrewAdapter
from planner_agent.agent.final_agent import FinalAgent, CrewAIAdapterForFinal
from planner_agent.tools.s3io import update_json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configurable params
TRANSPORT_ADAPTER = TransportAdapter()  # swap with real adapter if you have one
PLANNER_CREW_ADAPTER = PlannerCrewAdapter()  # replace with PlannerCrewAdapter() wired to CrewAI
FINAL_CREW_ADAPTER = CrewAIAdapterForFinal()  # replace with CrewAIAdapterForFinal()

def validate_itinerary(itinerary: Dict[str, Any], metrics: Dict[str, Any], payload: dict) -> Dict[str, Any]:
    """
    Validate gates (budget, coverage, pace) and return gates dict.
    This is intentionally deterministic; agentic decisions only occur in PlannerAgent.
    """
    req = payload.get("requirements", {})
    budget_cap = float(req.get("budget"))
    gates = {"budget_ok": True, "coverage_ok": True, "pace_ok": True, "uncertainty_escalate": False}

    # Build scheduled items list for budget aggregation
    scheduled_items = []
    for date, plan in itinerary.items():
        for slot in ("morning", "afternoon", "lunch"):
            item = plan.get(slot, {}).get("item")
            if item:
                scheduled_items.append({"item": item})
    number_adult = _safe_get(req, ["optional", "adult"]) or 1
    number_child = _safe_get(req, ["optional", "child"]) or 0
    number_senior = _safe_get(req, ["optional", "senior"]) or 0
    group_counts = req.get("group_counts", {"adult":number_adult,"child":number_child,"senior":number_senior})
    agg = aggregate_budget_range(scheduled_items, medians_by_type=None, group_counts=group_counts)
    total_transport_cost = sum(plan.get("metrics", {}).get("total_travel_cost_sgd", 0.0) for plan in itinerary.values())
    expected_total = agg["expected"] + total_transport_cost
    max_total = agg["max"] + total_transport_cost

    if agg["unknown_frac"] >= 0.30 or agg["uncertainty_ratio"] >= 0.10:
        gates["uncertainty_escalate"] = True

    gates["budget_ok"] = max_total <= budget_cap

    # coverage gate (simple): require some interest terms covered if requested
    if req.get("optional", {}).get("interests"):
        # metrics may include interest_terms_covered; fallback to True if unknown
        gates["coverage_ok"] = bool(metrics.get("interest_terms_covered"))

    # pace/time: ensure each day activity_time + travel_time <= pace_limit
    pace_minutes = _pace_minutes(req.get("pace"))
    for date, plan in itinerary.items():
        activity_minutes = 0
        for slot in ("morning", "afternoon"):
            it = plan.get(slot, {}).get("item")
            if it:
                activity_minutes += _minutes_for_item({"item": it})
        lunch_item = plan.get("lunch", {}).get("item")
        if lunch_item:
            activity_minutes += _lunch_minutes({"item": lunch_item})
        travel_min = plan.get("metrics", {}).get("total_travel_min", 0)
        if activity_minutes + travel_min > pace_minutes:
            gates["pace_ok"] = False
            break

    gates["all_ok"] = gates["budget_ok"] and gates["coverage_ok"] and gates["pace_ok"] and not gates["uncertainty_escalate"]
    gates["expected_total_spend_sgd"] = round(expected_total, 2)
    gates["max_total_spend_sgd"] = round(max_total, 2)
    gates["unknown_frac"] = agg["unknown_frac"]
    gates["uncertainty_ratio"] = agg["uncertainty_ratio"]
    return gates

def plan_itinerary(payload: dict, fileName: str) -> Dict[str, Any]:
    """
    Orchestrator main flow (agentic-ready).
    1) Heuristic Plan (core)
    2) Transport Retrieval (attach transport)
    3) Validate -> if fail call planner agent -> reattach transport -> revalidate (loop)
    4) FinalAgent formatting
    """
    t0 = time.time()
    # Stage 1: Heuristic plan
    scored = score_candidates(payload)
    sl = shortlist(payload, scored)
    itinerary, metrics = assign_to_days(payload, sl)

    # Add itinerary to payload and upload to S3
    payload["itinerary"] = itinerary
    # Upload to Retrieval Agent bucket
    update_json(Retrieval_Agent_Folder+"/"+ fileName, payload)

    # Stage 2: Transport Retrieval (attach transport options) TO BE REMOVE
    transport_adapter =TRANSPORT_ADAPTER
    itinerary = attach_transport_options(itinerary, transport_adapter)

    # Stage 3: Validation & possible repair loop
    gates = validate_itinerary(itinerary, metrics, payload)
    planner_agent = PlannerAgent(crew_adapter=PLANNER_CREW_ADAPTER)
    iterations = 0
    while not gates.get("all_ok") and iterations < MAX_AGENT_ITERATIONS:
        iterations += 1
        # Agentic repair: pass current itinerary, gates, metrics, shortlist
        new_it, new_metrics = planner_agent.run(itinerary, metrics, sl, payload, gates)
        if not new_it:
            # planner couldn't repair -> break and return best-effort
            break
        # Stage 2 again: reattach transport options in case structure changed (planner may have swapped items)
        new_it = attach_transport_options(new_it, transport_adapter) # TO BE REMOVE
        itinerary = new_it
        payload["itinerary"] = itinerary
        # Upload to Retrieval Agent bucket
        update_json(Retrieval_Agent_Folder+"/"+ fileName, payload)
        metrics = new_metrics or metrics
        gates = validate_itinerary(itinerary, metrics, payload)
        # loop continues until gates pass or max iterations exhausted

    # Stage 4: Final formatting using final agent
    final_agent = FinalAgent(crew_adapter=FINAL_CREW_ADAPTER)
    requirements = payload.get("requirements", {})
    explanation = explain(requirements, itinerary, metrics)
    final_payload = final_agent.run(itinerary, metrics, explanation, gates, payload)

    elapsed_ms = int((time.time() - t0) * 1000)
    return {
        "scored": scored,
        "shortlist": sl,
        "itinerary": itinerary,
        "metrics": metrics,
        "explanation": explanation,
        "gates": gates,
        "planner_iterations": iterations,
        "final_output": final_payload,
        "elapsed_ms": elapsed_ms
    }
    #return final_payload
