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

from planner_agent.tools.config import LLM_MODEL, OPENAI_API_KEY
from planner_agent.tools.helper import impute_price

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    from crewai import Crew, Agent, Task
except Exception:
    # SDK not installed or import failed; wrap gracefully
    Crew = None
    Agent = None
    Task = None
    LLM = None
# planner_agent.py (inside CrewAIAdapter)

def _parse_crew_output(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Robust parser for Crew raw responses.
    Accepts: dict, JSON string, list, and CrewOutput (new).
    Returns the first reasonable dict or None.
    """
    if raw is None:
        return None

    # NEW FIX: Handle CrewOutput objects directly
    if hasattr(raw, 'raw') and isinstance(raw.raw, str):
        raw = raw.raw  # Extract the underlying string content
    elif hasattr(raw, 'result') and isinstance(raw.result, str):
        raw = raw.result  # Alternative extraction method for older SDKs/variants

    # If already a dict, return it
    if isinstance(raw, dict):
        return raw
    # If it's a string, try to JSON-decode
    if isinstance(raw, str):
        try:
            # First attempt: treat the whole string as JSON
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            # Second attempt: try to find a JSON substring (robust extraction)
            try:
                # Use regex or simple find/split to isolate the JSON block
                # Looking for standard markdown JSON fences (`json` or `)
                if '```json' in raw:
                    raw = raw.split('```json', 1)[-1].split('```', 1)[0].strip()
                elif '```' in raw:
                    raw = raw.split('```', 1)[-1].split('```', 1)[0].strip()

                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                # not JSON; ignore
                return None
    # If it's a list, try each element (unchanged)
    if isinstance(raw, list):
        for elt in raw:
            # Recursive call to handle lists of strings or dicts
            parsed = _parse_crew_output(elt)
            if parsed:
                return parsed
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

    def __init__(self, max_retries: int = 1, timeout_seconds: int = 30, verbose: bool = True):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose

    def _build_agent_spec(self, prompt: str, task_description: str):
        """
        Build a minimal agent spec and expected output schema for the Task.
        Returns (agent_obj_or_descriptor, expected_output_dict)
        ### FIXED: provide expected_output as a dict (not a plain string).
        """
        GOAL = (
            "Repair the provided itinerary so it meets the validation gates (budget, pace/day, interest coverage, and uncertainty thresholds). "
            "Swap or remove items from the itinerary as needed to satisfy the gates."
            "CRITICAL RULE: Lunch stop validation now has TWO conditions for replacement: "
            "1) If the lunch item's **geo_cluster_id does NOT match** the cluster ID of the day's attractions, "
            "OR 2) If the lunch stop is determined to be too far (transport gate violation) from its preceding or succeeding attractions. "
            "If EITHER condition is true, you MUST remove the existing lunch item and replace it with an option from the 'AVAILABLE SWAP CANDIDATES DINING' list that satisfies BOTH cluster and proximity."
             "Return a JSON object with keys: itinerary, metrics, edits, status, notes. "
            "If not fully solvable, return a best-effort plan and clearly mark unsatisfied gates and reasons."
        )

        BACKSTORY = (
            "You are an experienced travel operations engineer and itinerary optimizer. "
            "You prefer conservative, verifiable edits that reduce cost, travel time, or uncertainty. "
            "Always include a short rationale and numeric impact estimate for each edit. "
            "Minimize changes: prefer removals of expensive or distant items first, then replacements using the shortlist."
        )
        LLM_CONFIG = {
            "api_key": OPENAI_API_KEY,
            "request_timeout": 60,
            "temperature": 0.2,
            # This is the standard way to force JSON output using LiteLLM/OpenAI config
            "response_format": {"type": "json_object"}
        }
        """LLM_MODEL = "apac.anthropic.claude-3-sonnet-20240229-v1:0"
        LLM_CONFIG = {
            # LiteLLM uses the 'model' parameter to specify the full provider and model name.
            # The format is typically "<provider>/<model_name>"
             "model": f"bedrock/{LLM_MODEL}",
            "request_timeout": 60,
            "temperature": 0.2,
        }"""
        # Use a lightweight agent descriptor if Agent class isn't available
        if Agent is not None:
            agent = Agent(
                role="Plan Repair",
                goal=GOAL,
                backstory=BACKSTORY,
                allow_delegation=False,
                verbose=self.verbose,
                llm=LLM_MODEL,
                config=LLM_CONFIG
            )
        else:
            # SDK not present; provide a dict describing the agent (some SDK variants accept this)
            agent = {
                "name": "planner_repair_agent",
                "role": "Plan Repair",
                "goal": GOAL,
                "backstory": BACKSTORY,
                "llm": LLM_MODEL,
                #"llm": LLM_CONFIG.get("model"),
                "config":LLM_CONFIG
            }

        # expected_output: Use a descriptive STRING (required by Pydantic Task validation)
        expected_output_str = (
            "A single, strict JSON object matching the schema provided in the task description. "
            "It must contain 'itinerary', 'metrics', 'edits', 'status', and 'human_summary' keys."
        )

        return agent, expected_output_str

    def run(self, prompt: str, task_description: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        New signature to match planner usage:
          run(prompt=str(prompt), task_description=str(task_description))
        Builds a Task (with expected_output) from the provided strings and optional context,
        calls the Crew SDK with the Task object (not stringified), and returns a parsed dict.
        ### FIX: Removed 'context=ctx' from Task constructors and task_payload to resolve Pydantic error.
        """
        if Crew is None:
            if self.verbose:
                logger.info("[CrewAIAdapter] Crew SDK not available; skipping agentic run.")
            return None

        prompt_text = prompt or ""
        task_desc = task_description or prompt_text
        ctx = context or {}

        # Build agent spec and expected_output (now a string)
        agent, expected_output = self._build_agent_spec(prompt_text, task_desc)

        # Try to construct a Task - many SDKs accept different param names; try likely variants
        task_obj = None
        if Task is not None:
            try:
                # FIX 1: Removed context=ctx
                task_obj = Task(description=task_desc, agent=agent, expected_output=expected_output)
            except Exception as e:
                # try alternative keyword
                try:
                    # FIX 2: Removed context=ctx
                    task_obj = Task(description=task_desc, agent=agent, expected_output_schema=expected_output)
                except Exception:
                    # final fallback: try the minimal Task with no expected_output and rely on string instructions in description
                    try:
                        # FIX 3: Removed context=ctx
                        task_obj = Task(description=task_desc, agent=agent)
                    except Exception as ee:
                        if self.verbose:
                            logger.exception("[CrewAIAdapter] Failed to construct Task object: %s", ee)
                        task_obj = None

        # If Task class not available or construction failed, use a dict payload
        if task_obj is None:
            task_payload = {
                "description": task_desc,
                "agent": agent,
                # FIX 4: Removed "context": ctx,
                "expected_output": expected_output
            }
        else:
            task_payload = task_obj

        attempt = 0
        last_exc = None
        # instantiate Crew once per call
        try:
            # Instantiate Crew with the task_payload
            crew = Crew(agents=[agent], tasks=[task_payload], verbose=self.verbose)
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
                    # Primary: call kickoff with no args, relying on pre-loaded tasks
                    raw = crew.kickoff()
                except Exception:
                    try:
                        # Fallback 1: some SDKs accept tasks=[task_payload]
                        raw = crew.kickoff(tasks=[task_payload])
                    except TypeError:
                        try:
                            # Fallback 2: some SDKs accept task=task_payload
                            raw = crew.kickoff(task=task_payload)
                        except Exception:
                            # Fallback 3: try the run method
                            try:
                                raw = crew.run(task_payload)
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
            attractions: Dict[str, Any],
            shortlist: Dict[str, Any],
            dining: Dict[str, Any],
            itinerary: Dict[str, Any],
            transport: Dict[str, Any],
            metrics: Dict[str, Any],
            gates: Dict[str, Any],
            force_review: bool = True
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
            return self._run_with_crew(itinerary, transport, metrics, shortlist, attractions, dining, requirements, gates,
                                       review_only=(force_review and gates_ok))

        # No Crew adapter or not needed -> local behavior
        if force_review:
            explained = self.local_explain_and_recommend(requirements, itinerary, transport, metrics, gates, attractions, dining)
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
            return self._local_repair(itinerary, metrics, shortlist, gates)
        return itinerary, metrics

    def _run_with_crew(self, itinerary, transport, metrics,
            shortlist, attractions, dining, requirements, gates, review_only: bool = False):
        """
        AGENTIC: call Crew to REVIEW or REPAIR the itinerary.
        review_only=True -> agent should NOT make large structural changes, only produce a 'review' and 'recommended edits'
        review_only=False -> agent should attempt repair if gates failing.
        Agent is asked to return a strict JSON contract described below.

        ### FIXED: robust call to crew_adapter.run with context and schema in task_description/context.
        """
        mode = "REVIEW_ONLY" if review_only else "REPAIR"

        HEURISTIC_INSTRUCTION = (
            "CRITICAL RULE (Lunch replacement): Lunch stop validation now has TWO conditions for replacement: "
            "1) If the lunch item's geo_cluster_id does NOT match the cluster ID of the day's attractions, "
            "OR 2) If the lunch stop is determined to be too far (transport gate violation) from its preceding or succeeding attractions. "
            "If EITHER condition is true, you MUST replace it with an option from the AVAILABLE SWAP CANDIDATES DINING list that satisfies BOTH cluster and proximity when possible. "
            "Prefer dining options that are stroller-friendly / accessible if the user's requirements indicate accessibility constraints. "
            "If no exact cluster/proximity match is available, select the closest acceptable dining option and clearly document the tradeoff in the audit_trace."
        )
        DAILY_COUNT_RULES = (
            "HARD CONSTRAINTS (Per-day composition): For every day in the itinerary you MUST ensure: "
            "A) At least ONE lunch stop is present. If missing, insert a lunch stop from AVAILABLE SWAP CANDIDATES DINING following the 'Lunch replacement' rule above. "
            "B) At least ONE attraction/place is scheduled in the Morning slot and at least ONE attraction/place in the Afternoon slot. "
            "If a Morning or Afternoon slot is empty, insert a place from AVAILABLE SWAP CANDIDATES ATTRACTIONS preferring the day's cluster and preserving pace/accessibility. "
            "When inserting, ensure the insertion does not create a transport gate violation (or, if unavoidable, document the issue in 'notes' and mark 'status' accordingly)."
        )
        UNIQUENESS_RULE = (
            "UNIQUENESS CONSTRAINT (NEW & REQUIRED): Under NO CIRCUMSTANCE may the repair algorithm reuse a place_id that already appears in the current itinerary or that has been inserted earlier during this repair session. "
            "All inserted or replacement items MUST have place_id values distinct from every other place_id in the final repaired itinerary (the only exception is the accommodation/place of stay which may be referenced multiple times). "
            "If the available swap pools lack distinct candidates, document this limitation in 'notes', include the best possible distinct candidate, and set 'status' = 'partial'."
        )

        prompt = f"""
            --- AGENT INSTRUCTIONS ---
            You are a Plan Repair Agent. Your task is to review the FAILED ITINERARY and use the available dining options to create a REPAIRED ITINERARY.
             HIGH-LEVEL GOAL: {'Review' if review_only else 'Repair or review'} the itinerary to satisfy these gates. MODE: {mode}."""

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
            --- AGENT INSTRUCTIONS ---
            You are a Plan Repair Agent. Your task is to review the FAILED ITINERARY and use the available dining and attraction swap candidates to create a REPAIRED ITINERARY.
            HIGH-LEVEL GOAL: {'Review' if review_only else 'Repair or review'} the itinerary to satisfy these gates. MODE: {mode}.

            {HEURISTIC_INSTRUCTION}

            {DAILY_COUNT_RULES}

            {UNIQUENESS_RULE}

            Important practical rules for replacements/insertions:
            - When replacing or inserting a lunch stop, prefer items that share the same geo_cluster_id as the day's attractions and that keep travel time between adjacent stops small.
            - When inserting a missing Morning/Afternoon place, prefer attractions from the same geo_cluster as other stops that day; prefer attractions that are stroller-friendly / accessible if user requires accessibility.
            - Do not create more than 1 new full-day rearrangement per day in REPAIR mode; prefer local swaps (replace or insert within same day).
            - Every edit MUST be auditable: include the original place(s), the replacement(s), the reason (gate violated or missing slot), estimated cost delta (if known), estimated transport impact (minutes), and a one-line `audit_trace`.
            - Maintain the UNIQUENESS CONSTRAINT: do not reuse any place_id present in the input itinerary or already chosen during repair. If no distinct candidate is available, document and mark 'partial'.

            INSTRUCTIONS (do not include internal chain-of-thought; provide auditable rationales):
            1) Produce a JSON object matching the 'expected_response_schema' below.
            2) If MODE == REVIEW_ONLY: Do NOT perform large edits. Provide human_summary, per_day_timeline, and recommended_edits (0..3). You MUST still flag missing slots and explicitly recommend which dining/attraction candidates could satisfy them (but do NOT apply the edits in REVIEW_ONLY).
            3) If MODE == REPAIR and gates failing: propose concrete edits and a repaired itinerary JSON that:
                 - ensures each day has at least one Morning place, one Lunch stop, and one Afternoon place (unless impossible).
                 - enforces the UNIQUENESS CONSTRAINT so no place_id is reused (except accommodation references).
                 - keeps replacements local to the day where possible and preserves pace/accessibility preferences.
            4) ALWAYS include 'edits' (empty list if none), 'status' (ok|partial|fail), and 'notes' explaining any remaining unsatisfied hard constraints.
            5) For every day, ALWAYS return a per_day_timeline entry that includes keys: 'date' (or label), 'morning' (array), 'lunch' (array, min length 1), 'afternoon' (array), 'evening' (array, optional).
            6) Keep output minimal and JSON-only.

            EXPECTED_RESPONSE_SCHEMA:
            {schema_json}
        """
        # Run the crew adapter
        try:
            response = None
            if self.crew_adapter:
                # The LLM context is now entirely within task_description
                response = self.crew_adapter.run(prompt=prompt, task_description=task_description)
            # If no response from crew, fallback deterministically
            if not response:
                if review_only:
                    explained = self.local_explain_and_recommend(requirements, itinerary, transport, metrics, gates, attractions, dining)
                    m = dict(metrics)
                    m["_agent_review"] = explained.get("agent_summary", {})
                    return explained.get("itinerary_suggested", itinerary), m
                else:
                    return self._local_repair(itinerary, metrics, shortlist , gates)

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
                explained = self.local_explain_and_recommend(requirements, itinerary, transport, metrics, gates, attractions, dining)
                m = dict(metrics)
                m["_agent_review"] = explained.get("agent_summary", {})
                return explained.get("itinerary_suggested", itinerary), m
            return self._local_repair(itinerary, metrics, shortlist, gates)

        # If we reach here, parsing failed; fallback
        if review_only:
            explained = self.local_explain_and_recommend(requirements, itinerary, transport, metrics, gates, attractions, dining)
            m = dict(metrics)
            m["_agent_review"] = explained.get("agent_summary", {})
            return explained.get("itinerary_suggested", itinerary), m
        return self._local_repair(itinerary, metrics, shortlist, gates)

    def local_explain_and_recommend(self, requirements, itinerary, transport, metrics, gates, attractions, dining):
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

    def _local_repair(self, itinerary, metrics, shortlist, gates):
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