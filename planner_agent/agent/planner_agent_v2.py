# planner_agent.py
# ---------- NEW (AGENTIC) ----------
# PlannerAgent is agentic: it can either use CrewAI (via CrewAIAdapter) or run a local heuristic.
# It accepts the failed gates and attempts to repair the plan repeatedly until gates pass or max iterations.
import logging
from typing import Dict, Any, Tuple, Optional
import copy
import json
import time
import os

from planner_agent.agent.transport import attach_transport_options
from planner_agent.tools.helper import impute_price

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Try to import the Crew class from your CrewAI SDK; adjust the import as your SDK requires.
try:
    from crewai import Crew, Agent, Task  # type: ignore
except Exception:
    # SDK not installed or import failed; wrap gracefully
    Crew = None
    Agent = None
    Task = None

LLM_MODEL = os.getenv("CREW_LLM_MODEL", "gpt-4o-mini")


def _parse_crew_output(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Robust parser for Crew raw responses.
    Accepts: dict, JSON string, list (of strings or dicts).
    Returns the first reasonable dict or None.
    ### FIXED: implemented defensive parsing so adapter doesn't crash.
    """
    if raw is None:
        return None
    # If already a dict, return it
    if isinstance(raw, dict):
        return raw
    # If it's a string, try to JSON-decode
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            # not JSON; ignore
            return None
    # If it's a list, try each element
    if isinstance(raw, list):
        for elt in raw:
            if isinstance(elt, dict):
                return elt
            if isinstance(elt, str):
                try:
                    parsed = json.loads(elt)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue
    # Unknown shape
    return None


class CrewAIAdapter:
    """
    CrewAI adapter to run agentic tasks synchronously.

    Usage:
      adapter = CrewAIAdapter(max_retries=2, timeout_seconds=30)
      response = adapter.run(prompt="Repair the itinerary...", context={...})

    Expected response: a dict (parsed JSON) with keys like 'itinerary' and 'metrics'.
    """

    def __init__(self, max_retries: int = 1, timeout_seconds: int = 30, verbose: bool = False):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose

    def _build_agent_spec(self, prompt: str, task_description: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Build a minimal agent spec and expected output schema for the Task.
        Returns (agent_obj_or_descriptor, expected_output_dict)
        ### FIXED: provide expected_output as a dict (not a plain string).
        """
        GOAL = (
            "Repair the provided itinerary so it meets the validation gates (budget, pace/day, interest coverage, and uncertainty thresholds). "
            "Return a JSON object with keys: itinerary, metrics, edits, status, notes. "
            "If not fully solvable, return a best-effort plan and clearly mark unsatisfied gates and reasons."
        )

        BACKSTORY = (
            "You are an experienced travel operations engineer and itinerary optimizer. "
            "You prefer conservative, verifiable edits that reduce cost, travel time, or uncertainty. "
            "Always include a short rationale and numeric impact estimate for each edit. "
            "Minimize changes: prefer removals of expensive or distant items first, then replacements using the shortlist."
        )
        # Use a lightweight agent descriptor if Agent class isn't available
        if Agent is not None:
            agent = Agent(
                name="planner_repair_agent",
                role="Plan Repair",
                goal=GOAL,
                backstory=BACKSTORY,
                allow_delegation=False,
                verbose=self.verbose,
                llm=LLM_MODEL,
            )
        else:
            # SDK not present; provide a dict describing the agent (some SDK variants accept this)
            agent = {
                "name": "planner_repair_agent",
                "role": "Plan Repair",
                "goal": GOAL,
                "backstory": BACKSTORY,
                "llm": LLM_MODEL,
            }

        # expected_output: use a strict JSON schema-style dict (SDKs often accept "expected_output" as dict)
        expected_output = {
            "type": "object",
            "required": ["itinerary"],
            "properties": {
                "itinerary": {"type": ["object", "null"], "description": "Full itinerary JSON (days -> slots -> places)."},
                "metrics": {"type": ["object", "null"], "description": "Optional metrics (costs, travel_time, etc.)."},
                "edits": {"type": "array", "description": "List of edits performed or recommended."},
                "status": {"type": "string", "description": "ok|partial|fail"},
                "notes": {"type": ["string", "null"], "description": "Human readable notes"},
                "human_summary": {"type": ["string", "null"], "description": "Short HTML or text summary"}
            },
            "additionalProperties": True
        }

        return agent, expected_output

    def run(self, prompt: str, task_description: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        New signature to match planner usage:
          run(prompt=str(prompt), task_description=str(task_description))
        Builds a Task (with expected_output) from the provided strings and optional context,
        calls the Crew SDK with the Task object (not stringified), and returns a parsed dict.
        ### FIXED: unified adapter.run implementation, robust parsing, tries multiple SDK call signatures.
        """
        if Crew is None:
            if self.verbose:
                logger.info("[CrewAIAdapter] Crew SDK not available; skipping agentic run.")
            return None

        prompt_text = prompt or ""
        task_desc = task_description or prompt_text
        ctx = context or {}

        # Build agent spec and expected_output
        agent, expected_output = self._build_agent_spec(prompt_text, task_desc)

        # Try to construct a Task - many SDKs accept different param names; try likely variants
        task_obj = None
        if Task is not None:
            try:
                task_obj = Task(description=task_desc, agent=agent, context=ctx, expected_output=expected_output)
            except Exception as e:
                # try alternative keyword
                try:
                    task_obj = Task(description=task_desc, agent=agent, context=ctx, expected_output_schema=expected_output)
                except Exception:
                    # final fallback: try the minimal Task with no expected_output and rely on string instructions in description
                    try:
                        task_obj = Task(description=task_desc, agent=agent, context=ctx)
                    except Exception as ee:
                        if self.verbose:
                            logger.exception("[CrewAIAdapter] Failed to construct Task object: %s", ee)
                        task_obj = None

        # If Task class not available or construction failed, use a dict payload
        if task_obj is None:
            task_payload = {
                "description": task_desc,
                "agent": agent,
                "context": ctx,
                "expected_output": expected_output
            }
        else:
            task_payload = task_obj

        attempt = 0
        last_exc = None
        # instantiate Crew once per call
        try:
            crew = Crew(agents=[agent], verbose=self.verbose)
        except Exception as e:
            logger.exception("[CrewAIAdapter] Failed to initialize Crew: %s", e)
            return None

        while attempt <= self.max_retries:
            attempt += 1
            try:
                if self.verbose:
                    logger.info(f"[CrewAIAdapter] kickoff attempt {attempt}")

                raw = None
                # Try common SDK invocation signatures defensively
                try:
                    # some SDKs accept tasks=[task_payload]
                    raw = crew.kickoff(tasks=[task_payload])
                except TypeError:
                    try:
                        raw = crew.kickoff(task=task_payload)
                    except TypeError:
                        try:
                            raw = crew.run(task_payload)
                        except Exception:
                            try:
                                raw = crew.kickoff()
                            except Exception as e:
                                if self.verbose:
                                    logger.info("[CrewAIAdapter] All kickoff signatures failed on this attempt: %s", e)
                                raise

                if self.verbose:
                    logger.info("[CrewAIAdapter] raw response type: %s", type(raw))

                parsed = _parse_crew_output(raw)
                if parsed:
                    return parsed

                # if raw is a string that contains JSON somewhere, attempt to extract
                if isinstance(raw, str):
                    try:
                        # try to find JSON substring
                        start = raw.find("{")
                        if start >= 0:
                            candidate = raw[start:]
                            parsed2 = json.loads(candidate)
                            if isinstance(parsed2, dict):
                                return parsed2
                    except Exception:
                        pass

                last_exc = RuntimeError("unparsable crew output")
            except Exception as exc:
                last_exc = exc
                logger.exception("[CrewAIAdapter] exception during kickoff attempt %d: %s", attempt, exc)
                time.sleep(min(0.5 * attempt, 3.0))
                continue

        logger.info("[CrewAIAdapter] all attempts failed: %s", last_exc)
        return None


class PlannerAgent:
    def __init__(self, crew_adapter: Optional[CrewAIAdapter] = None, max_iterations: int = 3):
        """
        crew_adapter: if provided, PlannerAgent will run agentic tasks via CrewAI; else uses local heuristics.
        """
        self.crew_adapter = crew_adapter
        self.max_iterations = max_iterations

    def run(self,
            requirements: Dict[str, Any],
            shortlist: Dict[str, Any],
            itinerary: Dict[str, Any],
            transport: Dict[str, Any],
            metrics: Dict[str, Any],
            gates: Dict[str, Any],
            force_review: bool = False
            ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Attempt to repair or REVIEW the itinerary.

        Returns: (new_itinerary_or_none, new_metrics_or_none)
        Always returns tuples (no other types).
        ### FIXED: simplified gates check and consistent returns.
        """
        # Prefer explicit 'all_ok' key; fallback to checking boolean keys
        gates_ok = bool(gates.get("all_ok")) if isinstance(gates, dict) else False

        needs_agent = force_review or not gates_ok

        if self.crew_adapter and needs_agent:
            return self._run_with_crew(itinerary, transport, metrics, shortlist, requirements, gates,
                                       review_only=(force_review and gates_ok))

        # No Crew adapter or not needed -> local behavior
        if force_review:
            explained = self.local_explain_and_recommend(requirements, itinerary, transport, metrics, gates, shortlist)
            # normalized dict shape
            it_suggested = explained.get("itinerary_suggested", itinerary) if isinstance(explained, dict) else itinerary
            new_metrics = explained.get("metrics", metrics) if isinstance(explained, dict) else metrics
            # attach agent_summary if present
            agent_summary = explained.get("agent_summary") if isinstance(explained, dict) else None
            if agent_summary:
                new_metrics = dict(new_metrics)
                new_metrics["_agent_review"] = agent_summary
            return it_suggested, new_metrics

        # default fallback: local repair if gates failing, otherwise return original
        if not gates_ok:
            return self._local_repair(itinerary, metrics, shortlist, requirements, gates)
        return itinerary, metrics

    def _run_with_crew(self, itinerary, transport, metrics, shortlist, requirements, gates, review_only: bool = False):
        """
        AGENTIC: call Crew to REVIEW or REPAIR the itinerary.
        review_only=True -> agent should NOT make large structural changes, only produce a 'review' and 'recommended edits'
        review_only=False -> agent should attempt repair if gates failing.
        Agent is asked to return a strict JSON contract described below.

        ### FIXED: robust call to crew_adapter.run with context and schema in task_description/context.
        """
        mode = "REVIEW_ONLY" if review_only else "REPAIR"
        prompt = f"{'Review' if review_only else 'Repair or review'} the itinerary to satisfy these gates. MODE: {mode}."

        expected_schema = {
            "itinerary": "object or null",
            "metrics": "object or null",
            "human_summary": "string or null",
            "per_day_timeline": "array",
            "edits": "array",
            "status": "ok|partial|fail",
            "notes": "string"
        }
        schema_json = json.dumps(expected_schema, indent=2)

        task_description = f"""
TRAVEL PLANNER {mode} TASK

USER REQUIREMENTS:
{json.dumps(requirements, indent=2)}

CURRENT ITINERARY:
{json.dumps(itinerary, indent=2)}

CURRENT TRANSPORT OPTIONS:
{json.dumps(transport, indent=2)}

PERFORMANCE METRICS:
{json.dumps(metrics, indent=2)}

GATE RESULTS:
{json.dumps(gates, indent=2)}

AVAILABLE SWAP CANDIDATES:
{json.dumps(shortlist, indent=2)}

INSTRUCTIONS (do not include internal chain-of-thought; provide auditable rationales):
1) Produce a JSON object matching the 'expected_response_schema' below.
2) If MODE == REVIEW_ONLY: Do NOT perform large edits. Provide human_summary, per_day_timeline, and recommended_edits (0..3).
3) If MODE == REPAIR and gates failing: propose concrete edits and a repaired itinerary JSON.
4) ALWAYS include 'edits' (empty list if none), 'status', and 'notes' for any remaining unsatisfied gates.
5) Keep output minimal and JSON-only.

EXPECTED_RESPONSE_SCHEMA:
{schema_json}
"""

        context = {
            "requirements": requirements,
            "itinerary": itinerary,
            "transport": transport,
            "metrics": metrics,
            "gates": gates,
            "shortlist": shortlist,
            "mode": mode
        }

        # Run the crew adapter
        try:
            response = None
            if self.crew_adapter:
                response = self.crew_adapter.run(prompt=prompt, task_description=task_description, context=context)
            # If no response from crew, fallback deterministically
            if not response:
                if review_only:
                    explained = self.local_explain_and_recommend(requirements, itinerary, transport, metrics, gates, shortlist)
                    m = dict(metrics)
                    m["_agent_review"] = explained.get("agent_summary", {})
                    return explained.get("itinerary_suggested", itinerary), m
                else:
                    return self._local_repair(itinerary, metrics, shortlist, requirements, gates)

            # parse/validate response (response may already be a dict)
            if isinstance(response, dict):
                new_it = response.get("itinerary") or itinerary
                new_metrics = response.get("metrics") or metrics
                agent_summary = {
                    "human_summary": response.get("human_summary"),
                    "edits": response.get("edits", []),
                    "status": response.get("status"),
                    "notes": response.get("notes")
                }
                new_metrics = dict(new_metrics)
                new_metrics["_agent_review"] = agent_summary
                return new_it, new_metrics

            # otherwise try to parse raw via helper
            parsed = _parse_crew_output(response)
            if parsed and isinstance(parsed, dict):
                new_it = parsed.get("itinerary") or itinerary
                new_metrics = parsed.get("metrics") or metrics
                agent_summary = {
                    "human_summary": parsed.get("human_summary"),
                    "edits": parsed.get("edits", []),
                    "status": parsed.get("status"),
                    "notes": parsed.get("notes")
                }
                new_metrics = dict(new_metrics)
                new_metrics["_agent_review"] = agent_summary
                return new_it, new_metrics

        except Exception as e:
            logger.exception("[PlannerAgent] crew_adapter.run failed with exception: %s", e)
            # Fallback behavior: deterministic explanation or repair
            if review_only:
                explained = self.local_explain_and_recommend(requirements, itinerary, transport, metrics, gates, shortlist)
                m = dict(metrics)
                m["_agent_review"] = explained.get("agent_summary", {})
                return explained.get("itinerary_suggested", itinerary), m
            return self._local_repair(itinerary, metrics, shortlist, requirements, gates)

        # If we reach here, parsing failed; fallback
        if review_only:
            explained = self.local_explain_and_recommend(requirements, itinerary, transport, metrics, gates, shortlist)
            m = dict(metrics)
            m["_agent_review"] = explained.get("agent_summary", {})
            return explained.get("itinerary_suggested", itinerary), m
        return self._local_repair(itinerary, metrics, shortlist, requirements, gates)

    def local_explain_and_recommend(self, requirements, itinerary, transport, metrics, gates, shortlist):
        """
        Lightweight deterministic fallback when Crew isn't available or when we need a quick
        auditable review without calling external LLM:
          - builds human_summary text,
          - lists up to 3 recommended edits (with simple numeric heuristics),
          - returns a suggested itinerary (may be same as input).

        ### FIXED: returns a dict with fixed keys for caller compatibility.
        """
        # Simple summary
        days = len(itinerary.keys()) if isinstance(itinerary, dict) else 0
        human_summary = f"Auto-review: {days} day(s). Budget target: {requirements.get('budget')}. Estimated cost: {metrics.get('estimated_cost', 'unknown')}."
        edits = []

        # If budget exceeded, recommend dropping the single most expensive scheduled item
        if not gates.get("budget_ok"):
            scheduled = []
            for date, plan in (itinerary.items() if isinstance(itinerary, dict) else []):
                for slot in ("morning", "afternoon", "lunch"):
                    item = plan.get(slot, {}).get("item") if isinstance(plan, dict) else None
                    if item:
                        price = impute_price(item).get("adults", 0)
                        scheduled.append({"date": date, "slot": slot, "item": item, "price": price})
            scheduled.sort(key=lambda x: x["price"], reverse=True)
            if scheduled:
                top = scheduled[0]
                edits.append({
                    "type": "remove",
                    "target": {"date": top["date"], "slot": top["slot"], "place_id": top["item"].get("place_id")},
                    "suggested": None,
                    "impact": {"cost_sgd_delta": -top["price"], "travel_minutes_delta": 0, "carbon_kg_delta": 0.0},
                    "rationale": f"Remove highest-cost item ({top['item'].get('name', 'unknown')}) to reduce budget pressure."
                })

        agent_summary = {
            "human_summary": human_summary,
            "edits": edits,
            "status": "partial" if edits else "ok",
            "notes": "Local heuristic review"
        }

        # normalized return contract
        return {
            "itinerary_suggested": itinerary,
            "metrics": metrics,
            "agent_summary": agent_summary
        }

    def _local_repair(self, itinerary, metrics, shortlist, requirements, gates):
        """
        Local heuristic repair:
         - If budget violated: iteratively remove the scheduled item with highest imputed price (descending),
         - If pace violated: drop afternoon slots first,
         - Re-attach transport and re-evaluate gates (orchestrator will re-run validate after receiving returned itinerary).
        This is conservative and deterministic (no external calls).
        """
        it = copy.deepcopy(itinerary)
        for iteration in range(self.max_iterations):
            scheduled = []
            for date, plan in (it.items() if isinstance(it, dict) else []):
                for slot in ("morning", "afternoon"):
                    item = plan.get(slot, {}).get("item") if isinstance(plan, dict) else None
                    if item:
                        price = impute_price(item).get("adults", 0)
                        scheduled.append({"date": date, "slot": slot, "item": item, "estimated_price": price})
            scheduled.sort(key=lambda x: x["estimated_price"], reverse=True)
            if not scheduled:
                break
            # Attempt removal based on gate type
            if not gates.get("budget_ok"):
                to_remove = scheduled[0]
                it[to_remove["date"]][to_remove["slot"]]["item"] = None
            elif not gates.get("pace_ok"):
                removed = False
                for s in scheduled:
                    if s["slot"] == "afternoon":
                        it[s["date"]][s["slot"]]["item"] = None
                        removed = True
                        break
                if not removed:
                    it[scheduled[0]["date"]][scheduled[0]["slot"]]["item"] = None
            else:
                # coverage/uncertainty heuristics
                removed_any = False
                if shortlist and isinstance(shortlist, dict):
                    for cat in shortlist.values():
                        for candidate in (cat if isinstance(cat, list) else []):
                            pid = candidate.get("place_id") or candidate.get("id")
                            for date, plan in it.items():
                                for slot in ("morning", "afternoon", "lunch"):
                                    item = plan.get(slot, {}).get("item")
                                    if item and (item.get("place_id") == pid or item.get("id") == pid):
                                        plan[slot]["item"] = None
                                        removed_any = True
                                        break
                                if removed_any:
                                    break
                            if removed_any:
                                break
                        if removed_any:
                            break
                if not removed_any:
                    it[scheduled[0]["date"]][scheduled[0]["slot"]]["item"] = None

            # Return modified itinerary; orchestrator is responsible for re-attaching transport and re-validating gates.
            return it, metrics

        # If cannot repair, return None pair
        return None, None
