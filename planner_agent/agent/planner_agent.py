# planner_agent.py
# ---------- NEW (AGENTIC) ----------
# PlannerAgent is agentic: it can either use CrewAI (via CrewAIAdapter) or run a local heuristic.
# It accepts the failed gates and attempts to repair the plan repeatedly until gates pass or max iterations.

from typing import Dict, Any, Tuple, Optional
import copy
import json
import time
from typing import Dict, Any, Optional
from crewai import Agent, Task
import os

from planner_agent.agent.transport import attach_transport_options
from planner_agent.tools.helper import impute_price

# Try to import the Crew class from your CrewAI SDK; adjust the import as your SDK requires.
try:
    from crewai import Crew
except Exception:
    Crew = None  # we'll handle missing SDK gracefully


LLM_MODEL = os.getenv("CREW_LLM_MODEL", "gpt-4o-mini")


def _parse_crew_output(result):
    pass


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

    def _build_agent_spec(self, prompt: str, task_description: str) -> Tuple[Agent, Task]:
        """
        Build the minimal agent/task spec for Crew from a human-readable prompt + structured context.
        Many Crew configurations accept a single agent with 'instruction' + 'context'.
        Adjust this to match your CrewAI SDK's expected schema.
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
        agent = Agent(
            name="planner_repair_agent",
            role="Plan Repair",
            goal=GOAL, # "Repair itinerary to meet gates and produce JSON",
            backstory=BACKSTORY,
            allow_delegation=False,
            verbose=False,
            llm=LLM_MODEL,
        )

        task = Task(
            description=task_description,  # your natural-language instruction
            agent=agent,  # which agent performs it
            #context=context,  # structured data you already build
            expected_output=  "JSON with keys 'itinerary' and 'metrics'"

        )
        return agent, task


    def run(self, prompt: str, task_description: str) -> Dict[str, Any]:
        """
        Synchronously run Crew with the provided prompt/context and return parsed JSON dict.
        On error or unparsable result, return None.
        """
        if Crew is None:
            raise RuntimeError("Crew SDK not available. Please install the CrewAI SDK or provide an adapter implementation.")

        agent, task = self._build_agent_spec(prompt, task_description)
        last_exc = None
        attempt = 0
        while attempt <= self.max_retries:
            attempt += 1
            try:
                if self.verbose:
                    print(f"[CrewAIAdapter] kickoff attempt {attempt}/{self.max_retries+1}")
                # Build Crew object -- your SDK signature may differ: update as needed.
                crew = Crew(agents=[agent], tasks=[task], verbose=False)

                # kickoff and wait (synchronous)
                start_time = time.time()
                result = crew.kickoff()
                elapsed = time.time() - start_time
                if self.verbose:
                    print(f"[CrewAIAdapter] crew.kickoff completed in {elapsed:.1f}s, raw result type: {type(result)}")

                # parse result
                parsed = _parse_crew_output(result)
                if parsed:
                    # Basic validation: require itinerary key (caller expects this)
                    if "itinerary" in parsed or ("final_itinerary" in parsed):
                        # normalize: prefer 'itinerary'
                        if "final_itinerary" in parsed and "itinerary" not in parsed:
                            parsed["itinerary"] = parsed.pop("final_itinerary")
                        return parsed
                    # If parsed dict doesn't contain itinerary, still return it to allow flexible usage
                    return parsed

                # If parsing failed, record and maybe retry
                last_exc = RuntimeError("Crew returned unparsable output")
                if self.verbose:
                    print("[CrewAIAdapter] parse failed; raw result:", result)
                # sleep briefly before retrying
                time.sleep(0.5)
            except Exception as e:
                last_exc = e
                if self.verbose:
                    print(f"[CrewAIAdapter] exception during kickoff: {e}")
                # exponential backoff small sleep
                time.sleep(min(1.0 * attempt, 5.0))
                continue

        # all retries exhausted
        # option: raise or return None; caller (PlannerAgent) expects None to fallback to deterministic repair
        if last_exc:
            # include a helpful message
            if self.verbose:
                print(f"[CrewAIAdapter] all attempts failed: {last_exc}")
        return None


class PlannerAgent:
    def __init__(self, crew_adapter: Optional[CrewAIAdapter] = None, max_iterations: int = 3):
        """
        crew_adapter: if provided, PlannerAgent will run agentic tasks via CrewAI; else uses local heuristics.
        """
        self.crew_adapter = crew_adapter
        self.max_iterations = max_iterations

    def run(self, itinerary: Dict[str, Any], metrics: Dict[str, Any], shortlist: Dict[str, Any], payload: dict, gates: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Attempt to repair the itinerary. Returns (new_itinerary, new_metrics) or (None, None) if unable.
        This method loops (agentic) between proposing fixes and re-checking gates.
        """
        # If crew adapter provided, prefer agentic flow
        if self.crew_adapter:
            return self._run_with_crew(itinerary, metrics, shortlist, payload, gates)

        # Fallback local deterministic repair (heuristic)
        return self._local_repair(itinerary, metrics, shortlist, payload, gates)

    def _run_with_crew(self, itinerary, metrics, shortlist, payload, gates):
        """
        AGENTIC: use CrewAI to propose a repaired plan. The CrewAI agent should be given:
          - the itinerary (with transport),
          - the failed gates,
          - the requirements -> ask it to propose concrete edits (remove/replace times).
        CrewAI should return repaired itinerary JSON that conforms to schema.
        """
        prompt = "Repair the itinerary to satisfy these gates. Return full itinerary JSON with keys: itinerary, metrics (optional)."

        requirements = payload.get("requirements", {})
        # Create a comprehensive description that includes all data
        task_description = f"""
           TRAVEL PLANNER CRITIQUE TASK

           USER REQUIREMENTS:
           {json.dumps(requirements, indent=2)}

           CURRENT ITINERARY:
           {json.dumps(itinerary, indent=2)}

           PERFORMANCE METRICS:
           {json.dumps(metrics, indent=2)}

           GATE RESULTS (FAILED GATES NEED FIXING):
           {json.dumps(gates, indent=2)}

           AVAILABLE SWAP CANDIDATES:
           {json.dumps(shortlist, indent=2)}

           INSTRUCTIONS:
           1. Analyze the current itinerary against user requirements and gate results
           2. Identify key issues causing gate failures or suboptimal experiences
           3. Suggest up to 3 concrete swaps using ONLY the available swap candidates
           4. Focus on: budget alignment, interest coverage, accessibility, and geographic efficiency
           5. MUST avoid: zoo activities, must ensure wheelchair accessibility, prefer vegan options

           CRITICAL USER PRIORITIES:
           - Group: {requirements.get('optional', {}).get('group_type')}
           - Interests: {', '.join(requirements.get('optional', {}).get('interests', []))}
           - Avoid: {', '.join(requirements.get('optional', {}).get('uninterests', []))}
           - Budget: ${requirements.get('budget')} total
           - Accessibility: {requirements.get('optional', {}).get('accessibility_needs')}
           - Dietary: {requirements.get('optional', {}).get('dietary_preferences')}
           """
        context = [{
            "gates": gates,
            "requirements": payload.get("requirements"),
            "itinerary": itinerary,
            "metrics": metrics,
            "shortlist": shortlist
        }]
        try:
            response = self.crew_adapter.run(prompt=str(prompt), task_description=str(task_description))
            # Expect response to contain 'itinerary' and optionally 'metrics'
            new_it = response.get("itinerary")
            new_metrics = response.get("metrics", metrics)
            return new_it, new_metrics
        except Exception as e:
            # Agent failed, fallback to local repair
            return self._local_repair(itinerary, metrics, shortlist, payload, gates)

    def _local_repair(self, itinerary, metrics, shortlist, payload, gates):
        """
        Local heuristic repair:
         - If budget violated: iteratively remove the scheduled item with highest imputed price (descending),
         - If pace violated: drop afternoon slots first,
         - Re-attach transport and re-evaluate gates.
        This is conservative and deterministic (no external calls).
        """
        it = copy.deepcopy(itinerary)
        # iterative attempts
        for iteration in range(self.max_iterations):
            # recompute scheduled items list
            scheduled = []
            for date, plan in it.items():
                for slot in ("morning", "afternoon"):
                    item = plan.get(slot, {}).get("item")
                    if item:
                        price = impute_price(item)
                        scheduled.append({"date": date, "slot": slot, "item": item, "estimated_price": price["adult"]})
            # sort by estimated_price desc (drop expensive first)
            scheduled.sort(key=lambda x: x["estimated_price"], reverse=True)
            if not scheduled:
                break
            # Attempt removal based on gate type
            if not gates.get("budget_ok"):
                # remove the most expensive scheduled item
                to_remove = scheduled[0]
                it[to_remove["date"]][to_remove["slot"]]["item"] = None
            elif not gates.get("pace_ok"):
                # try to remove afternoon items first
                removed = False
                for s in scheduled:
                    if s["slot"] == "afternoon":
                        it[s["date"]][s["slot"]]["item"] = None
                        removed = True
                        break
                if not removed:
                    # fallback to removing most expensive
                    it[scheduled[0]["date"]][scheduled[0]["slot"]]["item"] = None
            else:
                # if only uncertainty_escalate or coverage, remove low-score items (if identifiable)
                if shortlist:
                    # remove lowest scored in shortlist that is currently scheduled
                    removed_any = False
                    # shortlist structure may vary; we try to find items by id
                    for cat in shortlist.values():
                        for candidate in cat:
                            pid = candidate.get("place_id") or candidate.get("id")
                            for date, plan in it.items():
                                for slot in ("morning", "afternoon", "lunch"):
                                    item = plan.get(slot, {}).get("item")
                                    if item and (item.get("place_id") == pid or item.get("id") == pid):
                                        # remove
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
                        # last resort remove most expensive
                        it[scheduled[0]["date"]][scheduled[0]["slot"]]["item"] = None
                else:
                    it[scheduled[0]["date"]][scheduled[0]["slot"]]["item"] = None

            # re-attach transport and compute new gates via orchestrator.validate_itinerary (import locally to avoid cycle)
            it = attach_transport_options(it)
            # We can't import orchestrator at module load due to circularity; orchestrator will call PlannerAgent and check gates itself.
            # Return and let orchestrator re-run validate_itinerary on modified itinerary.
            return it, metrics
        # If cannot repair, return None
        return None, None
