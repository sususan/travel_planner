import re
from _sha2 import sha256
from typing import Any, Dict, Optional, Callable, Tuple, List, MutableMapping
import json
import logging
import time
import traceback
import os

from planner_agent.grafana.dashboard import start_metrics_server, PIPELINE_STEP_COUNT, ITINERARIES_CREATED, \
    GATE_FAILURES, AVG_CO2, LAST_RUN_DURATION, LAST_RUN_GAUGE
from planner_agent.tools.helper import convertCrewOutputToJson
from planner_agent.tools.planner_agent_helper import apply_edits_to_itinerary, parse_planner_repair_response, \
    detect_prompt_injection, sanitize_text_field
from planner_agent.tools.s3io import update_json_data, get_json_data

os.environ["OPENAI_MODEL_NAME"] = "gpt-4.1-nano-2025-04-14"#"gpt-4o-mini"
os.environ.setdefault("CREWAI_TELEMETRY_ENABLED", "False")
# Also disable common OpenTelemetry exporters to be safe if SDK uses them:
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "")
os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "")
logger = logging.getLogger("planner_agent_crewtools")

# Reduce noisy telemetry log spam
logging.getLogger("crewai.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("crewai").setLevel(logging.WARNING)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] planner_agent_crewtools - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

import crewai
try:
    # best-effort: disable telemetry API if SDK exposes it
    if hasattr(crewai, "telemetry"):
        crewai.telemetry.enabled = False
        # and replace any send function with a no-op
        if hasattr(crewai.telemetry, "send_trace"):
            crewai.telemetry.send_trace = lambda *a, **kw: None
except Exception:
    pass

# Try to import crew SDK & tool decorator
try:
    # Agent, Crew, Task are core components for v1.4.1 Agentic Flow
    from crewai import Crew, Task, Agent, CrewOutput  # noqa: F401
    from crewai.tools import tool  # decorate tool functions

    CREW_AVAILABLE = True
except Exception:
    CREW_AVAILABLE = False

    # Provide a noop decorator replacement so code can run in local-only mode
    def tool(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

# NOTE: Assuming these local implementations exist and are imported correctly
from planner_agent.tools.planner_agent_tools import score_candidates as _local_score_candidates
from planner_agent.tools.planner_agent_tools import shortlist as _local_shortlist
from planner_agent.tools.planner_agent_tools import assign_to_days as _local_assign_to_days
from planner_agent.tools.planner_agent_tools import call_transport_agent_api as _local_call_transport_agent_api
from planner_agent.tools.planner_agent_tools import validate_itinerary as _local_validate_itinerary

# -------------------------
# Crew @tool wrappers (FIXED for v1.4.1)
# -------------------------


# -------------------------
# CrewToolInvoker: robust calls to Crew (Simplified for v1.4.1)
# -------------------------
class CrewToolInvoker:
    def __init__(self, crew_agent_descriptor: Optional[dict] = None, timeout_seconds: int = 60, verbose: bool = True):
        self.crew_agent_descriptor = {"role": "PlannerAgent","goal": "Execute planning tool calls."}
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose
        try:
            start_metrics_server(port=int(os.getenv("PLANNER_METRICS_PORT", "8000")))
        except Exception:
            logger.exception("Failed to start metrics HTTP server")

    def call(self, tool_name: str, inputs: Dict[str, Any]) -> str:

        if CREW_AVAILABLE is False:
            raise ImportError("crewai SDK not available in this environment")

        # 1. Get the correct tool object
        tool_obj = next((t for t in ALL_TOOLS if getattr(t, 'name', '') == tool_name), None)
        if not tool_obj:
            return f"Tool {tool_name} not found in ALL_TOOLS list."

        # 2. Prepare Task description/prompt
        task_prompt = f"Call the tool '{tool_name}' with the following input arguments:\n{json.dumps(inputs, indent=2)}"

        # 3. Instantiate Agent object
        LLM_CONFIG = {
            "max_tokens": 500,
            "request_timeout": 60,
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        agent = Agent(
            role=self.crew_agent_descriptor.get('role', 'Tool Executor'),
            goal=self.crew_agent_descriptor.get('goal', 'Execute the assigned planning task.'),
            backstory=f"A dedicated agent responsible for executing the '{tool_name}' tool.",
            tools=[tool_obj],
            verbose=True, # Use Agent verbose if needed
            llm = "gpt-4o-mini",
            config=LLM_CONFIG
        )

        # 4. Create Task object (v1.4.1 standard)
        task = Task(
            description=task_prompt,
            expected_output="The output dictionary from the tool call.",
            tools=[tool_obj]
        )

        # FIX: Assign the Agent to the Task for Sequential Process compatibility
        task.agent = agent

        # 5. Instantiate Crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=self.verbose,
            full_output=True  # Ensures detailed output is available
            # process=Process.sequential # Default
        )

        try:
            # 6. Kickoff the Crew (v1.4.1 standard synchronous run)
            raw_result = crew.kickoff()
            return raw_result.raw

        except Exception as e:
            logger.exception(f"Crew tool {tool_name} kickoff failed: %s", e)
            #return {"success": False, "parsed": None, "raw": None, "error": str(e)}
            return ""


@tool("planner_repair")
def planner_repair_tool(bucket: str, key: str, max_edits: int = 4) -> dict:
    """
    Calls an LLM (via Crew agent) to propose minimal repairs.
    Returns a dict matching the OUTPUT SCHEMA:
    {
      "edits": [...],
      "explain_summary": "...",
      "confidence": "<low|medium|high>",
      "_raw": "<raw agent output for audit>"
    }
    """
    print("!! LLM guided planner_repair_tool!!")  # allowed AFTER docstring

    payload = get_json_data(bucket, key)
    requirements = payload.get("requirements", {})
    itinerary = payload.get("itinerary", {})
    gates = payload.get("gates", {})
    shortlist = payload.get("shortlist", {})

    # Prompt-injection defenses
    for k, v in requirements.items():
        if isinstance(v, str):
            if detect_prompt_injection(v):
                logger.warning("Prompt injection detected in requirements[%s]; neutralizing.", k)
                requirements[k] = "[REDACTED_SUSPICIOUS]"
            else:
                requirements[k] = sanitize_text_field(v)

    # sanitize shortlist names and descriptions
    for pool in ("attractions", "dining"):
        items = shortlist.get(pool) or []
        for i in items:
            if "name" in i: i["name"] = sanitize_text_field(i["name"])
            if "description" in i:
                if detect_prompt_injection(i["description"]):
                    i["description"] = "[REDACTED_SUSPICIOUS]"
                else:
                    i["description"] = sanitize_text_field(i["description"])
    output_schema = """
    {
      "edits": [
        {
          "action": "swap" | "remove" | "add",
          "day_index": <0-based day index>,
          "remove_place_id": "<place_id>",        // required for swap/remove
          "add_place_id": "<place_id>",           // required for swap/add
          "type": "dinning | attraction",
          "insert_after_place_id": "<place_id>" | null, // optional for add: where to insert
          "reason": "<one-line rationale>"
        }
        ...
      ],
      "explain_summary": "<one-paragraph summary of why edits should fix failing gates>",
      "confidence": "<low|medium|high>"
    }
    """

    task_prompt = f"""
    You are PlannerRepairAgent.  MUST RETURN EXACTLY ONE JSON OBJECT and NOTHING ELSE.

    CONTEXT:
    - USER REQUIREMENTS:
    {json.dumps(requirements, indent=2)}

    - CURRENT ITINERARY (one object):
    {json.dumps(itinerary, indent=2)}

    - VALIDATION GATES (failing gates highlighted):
    {json.dumps(gates, indent=2)}

    - ATTRACTIONS CANDIDATE POOL (attractions shortlist results; include place_id, name, price_estimate, category, cluster_id, reliability_score):
    {json.dumps(shortlist.get("attractions"), indent=2)}
    
    - DINING CANDIDATE POOL (dining shortlist results; include place_id, name, price_estimate, category, cluster_id, reliability_score):
    {json.dumps(shortlist.get("dining"), indent=2)}

    GOAL:
    Propose a minimal set of edits (swap, remove, or add) so the itinerary satisfies the failing gates.
    Constraints:
    1. You MAY ONLY use places that appear in the Candidate Pool.
    2. Do not invent new places.
    3. Make minimal edits: prefer removing or swapping single items rather than re-building the whole day.
    4. Provide a short rationale for each edit (1 sentence).
    5. If no reasonable repair is possible using candidates, return edits: [] and a short "cannot_repair" reason.
    6. Lunch stop must be included on all days.Never remove lunch stop instead of swapping it from DINING CANDIDATE POOL.
    7. Always swap the same type of place for the same day. E.g, if remove dining, swap with dining, if remove attraction, swap with attraction.
        
    OUTPUT SCHEMA (exact JSON shape — must follow; DO NOT change punctuation):
    {output_schema}
    """

    # Agent config
    LLM_CONFIG = {
        "max_tokens": 500,
        "request_timeout": 60,
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    # Create agent (use whichever variant you prefer from earlier options)
    agent = Agent(
        role="PlannerRepairAgent",
        goal="Propose minimal edits (swap/remove/add) from the provided candidate pool to satisfy failing validation gates.",
        backstory=(
            "A planner assistant that proposes compact itinerary edits drawn only from the provided shortlist. "
            "Focus on minimal, valid changes with a one-line rationale for each edit."
        ),
        verbose=False,
        llm="gpt-4o-mini",
        config=LLM_CONFIG
    )
    expected_output = """
    {
            "type": "object",
            "properties": {
                "edits": {"type": "array"},
                "explain_summary": {"type": "string"},
                "confidence": {"type": "string"}
            },
            "required": ["edits", "explain_summary", "confidence"],
            "additionalProperties": True
        }
    """
    task = Task(
        description=task_prompt,
        # give a minimal expected_output schema object so Crew/Pydantic is happy
        expected_output=expected_output,
    )

    # Attach the agent to the task (this is the important bit)
    task.agent = agent

    # Now build Crew with the single agent+task
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    raw_output = None
    try:
        kickoff = crew.kickoff()
        # `kickoff()` return shape depends on Crew version; adapt as necessary:
        # some versions: kickoff().raw, others: kickoff().output or kickoff()
        raw_output = getattr(kickoff, "raw", kickoff)
        raw_text = raw_output if isinstance(raw_output, str) else json.dumps(raw_output)
    except Exception as e:
        logger.exception("planner_repair: Crew kickoff failed")
        # safe failure response
        return {"edits": [], "explain_summary": "crew_kickoff_failed", "confidence": "low", "_raw": str(e)}

    # Try to parse JSON robustly
    parsed = None
    try:
        # If raw_output is a dict already (Crew may return parsed object), use it
        if isinstance(raw_output, dict):
            parsed = raw_output
        else:
            parsed = json.loads(raw_text)
    except Exception:
        # fallback: extract first {...} block
        m = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw_text, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None

    if not parsed:
        logger.warning("planner_repair: failed to parse agent output, returning safe fallback")
        return {"edits": [], "explain_summary": "parse_error", "confidence": "low", "_raw": raw_text}

    # Enforce schema sanity: cap edits, ensure ids are strings and that add/swap targets exist in shortlist
    shortlist_by_id = {p.get("place_id"): p for p in (shortlist.get("places") or [])}
    valid_edits = []
    for e in parsed.get("edits", [])[:max_edits]:
        action = e.get("action")
        if action not in ("swap", "remove", "add"):
            continue
        day_index = e.get("day_index")
        if not isinstance(day_index, int) or day_index < 0:
            continue
        # check presence of place ids for actions
        if action in ("swap", "remove") and not e.get("remove_place_id"):
            continue
        if action in ("swap", "add") and not e.get("add_place_id"):
            continue
        # ensure add_place_id exists in shortlist when provided
        add_id = e.get("add_place_id")
        if add_id and add_id not in shortlist_by_id:
            continue
        valid_edits.append({
            "action": action,
            "day_index": day_index,
            "remove_place_id": e.get("remove_place_id"),
            "add_place_id": e.get("add_place_id"),
            "insert_after_place_id": e.get("insert_after_place_id"),
            "reason": e.get("reason", "")[:200]
        })

    parsed["edits"] = valid_edits
    parsed["_raw"] = raw_text
    return parsed

@tool("score_candidates")
def score_candidates_tool(bucket: str, key: str) -> dict:
    """Returns a mapping place_id -> item info (same as local implementation)."""
    print("score_candidates_tool")
    try:
        if not callable(_local_score_candidates):
            raise RuntimeError("Local score_candidates implementation not available")
        return _local_score_candidates(bucket, key, None)
        #print(f"score_candidates_tool: out ={out}")
    except Exception as e:
        logger.exception("score_candidates_tool failed: %s", e)
        return {}

@tool("shortlist")
def shortlist_tool(bucket: str, key: str) -> dict:
    """Returns dict {'attractions': [...], 'dining': [...], 'reasons': {...}}."""
    print("shortlist_tool")
    try:
        if not callable(_local_shortlist):
            raise RuntimeError("Local shortlist implementation not available")
        return _local_shortlist(bucket, key)

    except Exception as e:
        logger.exception("shortlist_tool failed: %s", e)
        return {}

@tool("assign_to_days")
def assign_to_days_tool(bucket: str, key: str) -> dict:
    """Returns {'itinerary': {...}, 'metrics': {...}, 'assign_reasons': {...}}."""
    print("assign_to_days_tool")

    try:
        if not callable(_local_assign_to_days):
            raise RuntimeError("Local assign_to_days implementation not available")
        return _local_assign_to_days(bucket, key)

        #return {"itinerary": itinerary, "metrics": metrics}

    except Exception as e:
        logger.exception("assign_to_days_tool failed: %s", e)
        return {"error": str(e)}

@tool("call_transport_agent_api")
def call_transport_agent_api_tool(bucket: str, key: str, sender_agent: str, session: str) -> Dict[str, Any]:
    """Returns parsed JSON or minimal dict from the transport agent API."""
    print("call_transport_agent_api_tool")
    try:
        if not callable(_local_call_transport_agent_api):
            raise RuntimeError("Local call_transport_agent_api implementation not available")
        resp = _local_call_transport_agent_api(bucket, key, sender_agent, session)
        print(f"call_transport_agent_api_tool: resp ={resp}")
        return resp
    except Exception as e:
        logger.exception("call_transport_agent_api_tool failed: %s", e)
        return {"error": str(e)}


@tool("validate_itinerary")
def validate_itinerary_tool(bucket: str, key: str) -> Dict[
    str, Any]:
    """Run deterministic gate validation on the final itinerary."""
    print("validate_itinerary_tool")
    try:
        if not callable(_local_validate_itinerary):
            raise RuntimeError("Local validate_itinerary implementation not available")
        out = _local_validate_itinerary(bucket, key)
        return out if isinstance(out, dict) else {"result": out}
    except Exception as e:
        logger.exception("validate_itinerary_tool failed: %s", e)
        return {"error": str(e)}


# List of all tool objects for easy retrieval
ALL_TOOLS = [score_candidates_tool,shortlist_tool,assign_to_days_tool, call_transport_agent_api_tool,
             validate_itinerary_tool,planner_repair_tool]

TOOL_MAP: Dict[str, Callable] = {getattr(t, "name", ""): t for t in ALL_TOOLS}
ALLOWED_TOOL_NAMES = frozenset([n for n in TOOL_MAP.keys() if n])  # remove empty names

def _make_diag_id(entry: MutableMapping) -> str:
    """Create a short id for diagnostics storage."""
    return sha256(repr(entry).encode()).hexdigest()[:12]

def record_tool_incident(kind: str, details: dict, bucket: Optional[str] = None, key: Optional[str] = None) -> dict:
    """
    Record a structured incident for auditing. If bucket/key provided, append to that S3 JSON file's diagnostics list.
    Returns the persisted diagnostic entry (with id and timestamp).
    """
    print(f"record_tool_incident {bucket} : {key}")
    ts = int(time.time())
    entry = {
        "id": _make_diag_id({"kind": kind, "ts": ts, "details": details}),
        "kind": kind,
        "timestamp": ts,
        "details": details,
    }
    logger.warning("Tool incident: %s %s", kind, details)

    # Best-effort persistence: if bucket/key provided, append to payload.diagnostics.incidents
    try:
        if bucket and key:
            payload = get_json_data(bucket, key) or {}
            diagnostics = payload.get("diagnostics", {})
            incidents = diagnostics.get("incidents", [])
            # Avoid storing large sensitive content; truncate inputs preview
            if "inputs" in details and isinstance(details["inputs"], str):
                details["inputs_preview"] = details["inputs"][:1000]
                details.pop("inputs", None)
            incidents.append(entry)
            diagnostics["incidents"] = incidents
            payload["diagnostics"] = diagnostics
            print(f"record_tool_incident {bucket} : {key}")
            update_json_data(bucket, key, payload)
    except Exception:
        # Do not raise on persistence failure; we already logged the warning above
        logger.exception("Failed to persist incident to S3 (non-fatal)")

    return entry

# -------------------------
# Helpers
# -------------------------
def _parse_crew_result(raw: Any) -> Any:
    """Normalize likely Crew return shapes."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    if isinstance(raw, dict):
        # Look for standard output keys in case the result is wrapped
        for k in ("outputs", "result", "data", "choices", "parsed"):
            if k in raw:
                return raw[k]
        return raw
    return raw

# -------------------------
# PlannerAgent (owns tool calling via Crew or local)
# -------------------------
class PlannerAgent:
    def __init__(self, use_crew: bool = True, crew_agent_descriptor: Optional[dict] = None, crew_timeout: int = 180):
        """
        use_crew: if True, tool calls are executed via Crew and @tool-wrapped functions.
                  if False, local python functions are called directly.
        crew_agent_descriptor: Agent descriptor passed to Crew (e.g. {"role":"PlannerAgent","goal":"..."})
        """
        self.use_crew = use_crew and CREW_AVAILABLE
        self.crew_agent_descriptor = crew_agent_descriptor or {"role": "PlannerAgent",
                                                               "goal": "Execute the planning pipeline."}
        self.crew_timeout = crew_timeout

        if self.use_crew:
            # Setting verbose to True for debugging in invoker is often useful, but sticking to previous default
            self.invoker = CrewToolInvoker(self.crew_agent_descriptor, timeout_seconds=crew_timeout, verbose=True)
        else:
            self.invoker = None

    def _call_tool(self, tool_name: str, *args, **kwargs):
        """
        Unified tool call. If use_crew -> call via invoker.call(tool_name, inputs)
        For Crew, inputs are packaged as a dict where positional args are mapped to function parameters.
        """
        if self.use_crew:
            # bundle inputs into a serializable dict for CrewToolInvoker
            inputs = kwargs.pop("inputs", {})  # allow direct dict
            # Examples of positional mapping — keep minimal & explicit
            if not inputs:
                if tool_name in ("score_candidates", "shortlist", "validate_itinerary"):
                    if len(args) >= 2:
                        inputs = {"bucket": args[0], "key": args[1]}
                elif tool_name == "assign_to_days":
                    if len(args) >= 2:
                        inputs = {"bucket": args[0], "key": args[1]}
                elif tool_name == "call_transport_agent_api":
                    if len(args) >= 4:
                        inputs = {"bucket": args[0], "key": args[1], "caller": args[2], "session_id": args[3]}
                elif tool_name == "planner_repair":
                    if len(args) >= 3:
                        inputs = {"bucket": args[0], "key": args[1], "max_edits": args[2]}
                else:
                    caller = inputs.get("caller") if isinstance(inputs, dict) else None
                    tool_obj = TOOL_MAP.get(tool_name)
                    if not tool_obj:
                        diag = {
                            "attempted_tool": tool_name,
                            "allowed_tools": list(ALLOWED_TOOL_NAMES),
                            "caller": caller or "unknown",
                            "inputs": str(inputs)[:2000],
                        }
                        record_tool_incident("disallowed_tool_lookup", diag,
                                             bucket=args[0] if isinstance(inputs, dict) else None,
                                             key=args[1] if isinstance(inputs, dict) else None)
                        # Fail closed: raise so orchestrator handles the error path
                        raise RuntimeError("Requested tool is not available or not permitted. Incident logged.")

            if "caller" not in inputs:
                inputs["caller"] = "planner_agent"
            resp = self.invoker.call(tool_name, inputs)
            return resp
        else:
            # local mode - call local functions directly
            if tool_name == "score_candidates":
                return _local_score_candidates(*args, **kwargs)
            if tool_name == "shortlist":
                return _local_shortlist(*args, **kwargs)
            if tool_name == "assign_to_days":
                return _local_assign_to_days(*args, **kwargs)
            if tool_name == "call_transport_agent_api":
                return _local_call_transport_agent_api(*args, **kwargs)
            if tool_name == "validate_itinerary":
                return _local_validate_itinerary(*args, **kwargs)
            if tool_name == "planner_repair":
                return planner_repair_tool(*args, **kwargs)
            raise RuntimeError(f"Unknown tool: {tool_name}")

    def run(self, payload: Dict[str, Any], bucket: str, key: str, session_id: str) -> Dict[str, Any]:
        """
        Run full deterministic pipeline using internal tool-calling:
          1) score_candidates
          2) shortlist
          3) assign_to_days
          4) call_transport_agent_api
          5) validate_itinerary

        Returns the same structure asked in your spec, includes 'explanation'.
        """
        start = time.time()
        diagnostics = {"run_id": session_id, "steps": []}
        try:
            # 1) score
            diagnostics["steps"].append("score_candidates")
            PIPELINE_STEP_COUNT.labels(step="score_candidates", status="started").inc()
            score_ret = self._call_tool("score_candidates", bucket, key)
            PIPELINE_STEP_COUNT.labels(step="score_candidates", status="success").inc()
            print(f"score_ret ={score_ret}")

            # 2) shortlist
            diagnostics["steps"].append("shortlist")
            PIPELINE_STEP_COUNT.labels(step="shortlist", status="started").inc()
            shortlist_ret = self._call_tool("shortlist", bucket, key)
            PIPELINE_STEP_COUNT.labels(step="shortlist", status="success").inc()
            #json_data = convertCrewOutputToJson(score_ret)
            #print(f"shortlist_ret ={shortlist_ret}")

            # 3) assign to days
            diagnostics["steps"].append("assign_to_days")
            PIPELINE_STEP_COUNT.labels(step="assign_to_days", status="started").inc()
            ret = self._call_tool("assign_to_days", bucket, key)
            PIPELINE_STEP_COUNT.labels(step="assign_to_days", status="success").inc()
            #json_data = convertCrewOutputToJson(ret)
            #itinerary = json_data.get("itinerary")
            #metrics = json_data.get("metrics")

            # 4) transport agent API
            diagnostics["steps"].append("call_transport_agent_api")
            PIPELINE_STEP_COUNT.labels(step="call_transport_agent_api", status="started").inc()
            transport_result = None

            # --- START FIX: Add Retries for ConnectionResetError ---
            MAX_RETRIES = 3
            RETRY_DELAY = 1.0  # Initial delay in seconds

            for attempt in range(MAX_RETRIES):
                try:
                    # Pass required positional arguments
                    transport_result = self._call_tool("call_transport_agent_api", bucket, key, "planner_agent",
                                                           session_id)
                    PIPELINE_STEP_COUNT.labels(step="call_transport_agent_api", status="success").inc()
                    print(f"transport_result ={transport_result}")
                    # If successful, break the loop
                    break
                except ConnectionResetError as e:
                    logger.warning(f"Transport tool connection failed (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    if attempt + 1 == MAX_RETRIES:
                         raise  # Re-raise error if all attempts fail

                    # Exponential backoff: 1s, 2s, 4s, ...
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                except Exception as e:
                    # Break on non-ConnectionReset errors (like missing argument, logic error, etc.)
                     logger.exception("Transport tool failed with non-network error: %s", e)
                     transport_result = {"error": str(e)}
                     break

             # --- END FIX ---

            # 5) validate itinerary
            diagnostics["steps"].append("validate_itinerary")
            PIPELINE_STEP_COUNT.labels(step="validate_itinerary", status="started").inc()
            ret = self._call_tool("validate_itinerary", bucket, key)
            gates = convertCrewOutputToJson(ret)
            PIPELINE_STEP_COUNT.labels(step="validate_itinerary", status="success").inc()
            # record gate failures (per-gate counter)
            if isinstance(gates, dict):
                for gate_name, gate_val in gates.items():
                    # assume gate boolean or structured dict with 'ok' field
                    try:
                        if gate_name == "all_ok":
                            continue
                        ok = False
                        if isinstance(gate_val, dict):
                            ok = gate_val.get("ok", gate_val.get("result", False))
                        else:
                            ok = bool(gate_val)
                        if not ok:
                            GATE_FAILURES.labels(gate=gate_name).inc()
                    except Exception:
                        # safe fallback
                        pass
            # success path
            run_success = True
            ITINERARIES_CREATED.labels(status="success").inc()

            if not gates.get("all_ok"):
                #call planner_repair (LLM) with current state
                diagnostics["steps"].append("planner_repair")
                ret = self._call_tool("planner_repair", bucket, key,max_edits=1)
                planner_repair_resp = parse_planner_repair_response(ret)
                print(f"planner_repair_resp ={planner_repair_resp}")
                edits = planner_repair_resp.get("edits", [])

                if not edits:
                    # no repair possible by LLM using current candidates -> fall back to deterministic tweaks or return failure
                    diagnostics["repair"] = {"status": "cannot_repair", "reason": planner_repair_resp.get("explain_summary")}
                else:
                    # 3) apply edits deterministically
                    # build map
                    payload = get_json_data(bucket, key)
                    shortlist = payload.get("shortlist", {})
                    itinerary = payload.get("itinerary", {})
                    shortlist_by_id = {p["place_id"]: p for p in shortlist.get("places", [])}

                    new_itinerary = apply_edits_to_itinerary(itinerary, edits, shortlist_by_id)
                    print(f"new_itinerary ={new_itinerary}")
                    # 5) re-run transport & validate
                    transport_result = self._call_tool("call_transport_agent_api", bucket, key, "planner_agent",
                                                       session_id)
                    new_gates = self._call_tool("validate_itinerary", bucket, key)

                    # 6) report repair result
                    diagnostics["repair"] = {
                        "planner_repair_resp": planner_repair_resp,
                        "transport_result": transport_result,
                        "gates_after": new_gates
                    }
                    itinerary = new_itinerary
                    gates = new_gates

                    # 4) persist or overwrite file used by transport/validation tools (so subsequent calls read updated itinerary)
                    payload["itinerary"] = itinerary
                    payload["gates"] = gates

            #diagnostics["duration_s"] = round(time.time() - start, 3)
            #payload["diagnostics"] = diagnostics
            #update_json_data(bucket, key, payload)
            return {
                "success": True,
                "itinerary": payload["itinerary"],
                "metrics": payload["metrics"],
                "gates": gates,
                #"explanation": explanation,
                "transport_result": transport_result,
                "diagnostics": diagnostics
            }

        except Exception as e:
            logger.exception("PlannerAgent.run failed: %s", e)
            ITINERARIES_CREATED.labels(status="failure").inc()
            diagnostics["error"] = str(e)
            diagnostics["traceback"] = traceback.format_exc()
            return {"success": False, "error": str(e), "diagnostics": diagnostics}
        finally:
            duration = time.time() - start
            LAST_RUN_DURATION.observe(duration)
            diagnostics["duration_s"] = round(duration, 3)
            LAST_RUN_GAUGE.set(time.time())
            update_json_data(bucket, key, payload)  # ensure metrics persisted with diagnostics