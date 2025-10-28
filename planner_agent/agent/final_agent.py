# final_agent.py
# ---------- NEW (AGENTIC) ----------
# FinalAgent formats the final itinerary for user consumption and can optionally call CrewAI to generate
# human-friendly narrative, timeline, or email-ready content.
import os
from typing import Dict, Any, Optional
import json

from planner_agent.tools.helper import safe_item_name
# final_agent.py
# ---------- AGENTIC final agent to polish + send itinerary ----------
# Usage:
#   crew_adapter = CrewAIAdapterForFinal(max_retries=2, timeout_seconds=30, verbose=True)
#   sender = ConsoleSender() OR FileSender("/tmp/itinerary.json") OR your own SenderAdapter
#   agent = FinalAgent(crew_adapter=crew_adapter, sender=sender)
#   result = agent.run_and_send(itinerary, metrics, explanation, gates, payload, recipient="user@example.com")
#
# The function returns a dict with send status and the final_payload that was sent.

import json
import time
from typing import Dict, Any, Optional

# Try import Crew SDK; if not available, adapter will fallback to deterministic behavior
try:
    from crewai import Crew, Agent, Task
except Exception:
    Crew = None

LLM_MODEL = os.getenv("CREW_LLM_MODEL", "gpt-4o-mini")

# ------------------------
# Crew adapter for final
# ------------------------
class CrewAIAdapterForFinal:
    """
    AGENTIC HOOK:
    Implement .run(prompt, context) to submit to CrewAI and receive a polished natural-language itinerary.
    This implementation expects a Crew SDK with a Crew class and kickoff() method like in the planner adapter.
    If Crew is not installed, the adapter returns None and the agent will use deterministic formatting.
    """

    def __init__(self, max_retries: int = 1, timeout_seconds: int = 30, verbose: bool = False):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose

    def _build_agent_spec(self, prompt: str, context: Dict[str, Any]):
        # Adjust to the actual Crew agent spec you use; this is a minimal convention.
        GOAL = (
            "Produce a polished, user-ready itinerary and concise human summary from the provided structured itinerary. "
            "Ensure the output highlights any warnings (budget, pace, coverage), clearly lists per-day timelines, travel times/costs, "
            "and gives actionable notes (what changed, what to verify). Return a JSON object containing at minimum: "
            "'human_summary' (string), 'itinerary' (structured), and optional 'attachments' (list)."
        )

        BACKSTORY = (
            "You are an expert travel designer and communicator who turns machine-generated plans into clear, trustworthy travel itineraries. "
            "You know how travellers read itineraries (quick skim then details), you prefer clarity over cleverness, and you always surface "
            "risks and verification points (e.g., 'prices estimated', 'opening hours may change'). "
            "When formatting, include a 3–5 sentence top-level summary, a per-day timeline with times, modes, and cost estimates, and a short "
            "section of 'Action items' for the user (e.g., verify tickets, check weather, book restaurants)."
        )

        agent = Agent(
            name="final_formatter_agent",
            role="formatter",
            goal=GOAL,  # "Repair itinerary to meet gates and produce JSON",
            backstory=BACKSTORY,
            allow_delegation=False,
            verbose=False,
            llm=LLM_MODEL,
        )

        task = Task(
            description=prompt,  # your natural-language instruction
            agent=agent,  # which agent performs it
            context=context,  # structured data you already build

        )
        return agent, task

    def _parse_crew_output(self, raw) -> Optional[Dict[str, Any]]:
        # Accept dict directly
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list):
            # find first dict-like element
            for e in raw:
                if isinstance(e, dict):
                    return e
                if isinstance(e, str):
                    try:
                        parsed = json.loads(e)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        continue
            return None
        if isinstance(raw, str):
            # try parse as JSON
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                # try extract first JSON substring
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(raw[start:end+1])
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        return None
        return None

    def run(self, prompt: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Run Crew to produce a polished itinerary. Returns parsed dict or None on failure.
        Expected structured response (recommended by prompt):
          {
            "human_summary": "... short text ...",
            "itinerary": { ... possibly augmented ... },
            "attachments": [ { "filename": "itinerary.html", "content": "<html>...</html>" } ],
            "metadata": {...}
          }
        """
        if Crew is None:
            if self.verbose:
                print("[CrewAIAdapterForFinal] Crew SDK not installed; skipping Crew step.")
            return None

        agent_spec = self._build_agent_spec(prompt, context)
        attempt = 0
        last_exc = None
        while attempt <= self.max_retries:
            attempt += 1
            try:
                if self.verbose:
                    print(f"[CrewAIAdapterForFinal] kickoff attempt {attempt}")
                crew = Crew(agents=[agent_spec], verbose=False)
                start = time.time()
                raw = crew.kickoff()
                elapsed = time.time() - start
                if self.verbose:
                    print(f"[CrewAIAdapterForFinal] Crew run finished ({elapsed:.1f}s). Raw type: {type(raw)}")
                parsed = self._parse_crew_output(raw)
                if parsed:
                    return parsed
                last_exc = RuntimeError("Crew returned unparsable output")
                time.sleep(0.2)
            except Exception as e:
                last_exc = e
                if self.verbose:
                    print(f"[CrewAIAdapterForFinal] exception: {e}")
                time.sleep(min(0.5 * attempt, 3.0))
                continue
        if self.verbose:
            print(f"[CrewAIAdapterForFinal] all attempts failed: {last_exc}")
        return None

# ------------------------
# Sender adapters (pluggable)
# ------------------------
class SenderAdapter:
    """
    Implement send(recipient, subject, body, html=None, attachments=None) -> dict
    Return a dict with at least: { "ok": bool, "message": str }
    """

    def send(self, recipient: str, subject: str, body: str, html: Optional[str] = None, attachments: Optional[list] = None) -> Dict[str, Any]:
        raise NotImplementedError("Implement send in subclass")

class ConsoleSender(SenderAdapter):
    """Simple sender that prints to console (useful in dev)."""
    def send(self, recipient: str, subject: str, body: str, html: Optional[str] = None, attachments: Optional[list] = None) -> Dict[str, Any]:
        print("=== Sending itinerary (ConsoleSender) ===")
        print("To:", recipient)
        print("Subject:", subject)
        print("Body:\n", body)
        if html:
            print("HTML content present (not shown).")
        if attachments:
            print(f"Attachments: {len(attachments)}")
        return {"ok": True, "message": "printed to console"}

class FileSender(SenderAdapter):
    """
    Save the payload to disk as JSON + HTML (attachments) for later retrieval.
    path_prefix: directory or filename prefix.
    """
    def __init__(self, path_prefix: str = "/tmp/itinerary"):
        self.path_prefix = path_prefix

    def send(self, recipient: str, subject: str, body: str, html: Optional[str] = None, attachments: Optional[list] = None) -> Dict[str, Any]:
        ts = int(time.time())
        json_path = f"{self.path_prefix}_{ts}.json"
        try:
            payload = {"recipient": recipient, "subject": subject, "body": body, "html": bool(html), "attachments": []}
            if attachments:
                # attachments expected as dicts with filename & content
                for a in attachments:
                    fname = a.get("filename")
                    content = a.get("content")
                    path = f"{self.path_prefix}_{ts}_{fname}"
                    try:
                        with open(path, "w", encoding="utf-8") as fh:
                            fh.write(content)
                        payload["attachments"].append(path)
                    except Exception:
                        payload["attachments"].append({"error_saving": fname})
            with open(json_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, indent=2))
            return {"ok": True, "message": f"written to {json_path}"}
        except Exception as e:
            return {"ok": False, "message": str(e)}

# ------------------------
# Utilities for formatting itinerary
# ------------------------
def safe_item_name(slot_obj):
    if not slot_obj or not isinstance(slot_obj, dict):
        return None
    item = slot_obj.get("item")
    if not item or not isinstance(item, dict):
        return None
    return item.get("name")

def build_plain_text_summary(itinerary: Dict[str, Any], metrics: Dict[str, Any], gates: Dict[str, Any]) -> str:
    lines = []
    lines.append("Your itinerary\n")
    # summary line
    total_spend = gates.get("expected_total_spend_sgd", metrics.get("estimated_adult_ticket_spend_sgd"))
    lines.append(f"Approx total spend (SGD): {total_spend}")
    if not gates.get("all_ok"):
        lines.append("WARNING: some planning gates failed. See notes below.")
    lines.append("")

    for date, plan in (itinerary.items() if isinstance(itinerary, dict) else []):
        lines.append(f"=== {date} ===")
        morning = safe_item_name(plan.get("morning"))
        lunch = safe_item_name(plan.get("lunch"))
        afternoon = safe_item_name(plan.get("afternoon"))
        travel_min = plan.get("metrics", {}).get("total_travel_min", 0)
        travel_cost = plan.get("metrics", {}).get("total_travel_cost_sgd", 0.0)
        lines.append(f"Morning: {morning or '—'}")
        lines.append(f"Lunch: {lunch or '—'}")
        lines.append(f"Afternoon: {afternoon or '—'}")
        lines.append(f"Travel time (min): {travel_min}, travel cost (SGD): {travel_cost}")
        lines.append("")
    # gates summary
    lines.append("Gates:")
    for k, v in gates.items():
        lines.append(f" - {k}: {v}")
    return "\n".join(lines)

def build_simple_html(itinerary: Dict[str, Any], metrics: Dict[str, Any], gates: Dict[str, Any]) -> str:
    # lightweight HTML summary; replace with templates later
    parts = ["<html><body><h1>Your Itinerary</h1>"]
    total_spend = gates.get("expected_total_spend_sgd", metrics.get("estimated_adult_ticket_spend_sgd"))
    parts.append(f"<p><strong>Approx total spend (SGD):</strong> {total_spend}</p>")
    if not gates.get("all_ok"):
        parts.append("<p style='color:darkred'><strong>Warning:</strong> Some planning gates failed. Review the notes below.</p>")
    parts.append("<div>")
    for date, plan in (itinerary.items() if isinstance(itinerary, dict) else []):
        parts.append(f"<h2>{date}</h2>")
        morning = safe_item_name(plan.get("morning")) or "&mdash;"
        lunch = safe_item_name(plan.get("lunch")) or "&mdash;"
        afternoon = safe_item_name(plan.get("afternoon")) or "&mdash;"
        travel_min = plan.get("metrics", {}).get("total_travel_min", 0)
        travel_cost = plan.get("metrics", {}).get("total_travel_cost_sgd", 0.0)
        parts.append(f"<p><strong>Morning:</strong> {morning}<br>")
        parts.append(f"<strong>Lunch:</strong> {lunch}<br>")
        parts.append(f"<strong>Afternoon:</strong> {afternoon}<br>")
        parts.append(f"<strong>Travel time (min):</strong> {travel_min}, <strong>Travel cost (SGD):</strong> {travel_cost}</p>")
    parts.append("</div>")
    parts.append("<h3>Gates</h3><ul>")
    for k, v in gates.items():
        parts.append(f"<li>{k}: {v}</li>")
    parts.append("</ul></body></html>")
    return "".join(parts)

# ------------------------
# FinalAgent (main class)
# ------------------------
class FinalAgent:
    """
    FinalAgent polishes and sends the final itinerary to the user.
    It is 'agentic' in that it optionally calls Crew to produce a nicer human summary.
    """

    def __init__(self, crew_adapter: Optional[CrewAIAdapterForFinal] = None, sender: Optional[SenderAdapter] = None, verbose: bool = False):
        self.crew_adapter = crew_adapter
        self.sender = sender or ConsoleSender()
        self.verbose = verbose

    def run_and_send(self,
                     itinerary: Dict[str, Any],
                     metrics: Dict[str, Any],
                     explanation: str,
                     gates: Dict[str, Any],
                     payload: dict,
                     recipient: str,
                     subject: Optional[str] = None) -> Dict[str, Any]:
        """
        Produce final content and send to recipient.
        Returns a dict: {
          "ok": bool, "send_result": {..}, "final_payload": {..}
        }
        """
        subject = subject or "Your itinerary"
        # 1) Optionally call Crew adapter to generate polished human_summary and attachments
        human_summary = None
        attachments = None
        final_itinerary = itinerary  # default: pass-through
        final_metrics = metrics

        if self.crew_adapter:
            prompt = "Polish and format the itinerary for the user. Output JSON with keys: human_summary (string), itinerary (optional), attachments (optional list of {filename,content})."
            context = {
                "itinerary": itinerary,
                "metrics": metrics,
                "gates": gates,
                "requirements": payload.get("requirements"),
                "explanation": explanation
            }
            if self.verbose:
                print("[FinalAgent] invoking crew adapter for polishing...")
            try:
                crew_resp = self.crew_adapter.run(prompt, context)
                if crew_resp:
                    # Accept crew edit if provided
                    human_summary = crew_resp.get("human_summary")
                    if crew_resp.get("itinerary"):
                        final_itinerary = crew_resp.get("itinerary")
                    if crew_resp.get("metrics"):
                        final_metrics = crew_resp.get("metrics")
                    attachments = crew_resp.get("attachments")
            except Exception as e:
                if self.verbose:
                    print("[FinalAgent] crew adapter error:", e)
                # fallback to deterministic formatting

        # If Crew didn't produce a human summary, build deterministic one
        if not human_summary:
            human_summary = build_plain_text_summary(final_itinerary, final_metrics, gates)

        # Build HTML for richer email if needed
        html = build_simple_html(final_itinerary, final_metrics, gates)

        # Build a main payload to send/store
        final_payload = {
            "itinerary": final_itinerary,
            "metrics": final_metrics,
            "gates": gates,
            "human_summary": human_summary,
            "explanation": explanation
        }

        # 3) Send using sender adapter
        try:
            send_result = self.sender.send(recipient=recipient, subject=subject, body=human_summary, html=html, attachments=attachments)
            ok = bool(send_result and send_result.get("ok"))
        except Exception as e:
            send_result = {"ok": False, "message": str(e)}
            ok = False

        if self.verbose:
            print("[FinalAgent] send_result:", send_result)

        return {"ok": ok, "send_result": send_result, "final_payload": final_payload}


class FinalAgent:
    def __init__(self, crew_adapter: Optional[CrewAIAdapterForFinal] = None):
        self.crew_adapter = crew_adapter

    def run(self, itinerary: Dict[str, Any], metrics: Dict[str, Any], explanation: str, gates: Dict[str, Any], payload: dict) -> Dict[str, Any]:
        """
        Compose final payload:
         - structured itinerary (with transport)
         - metrics, gates
         - human summary (either local or produced by CrewAI)
        """
        if self.crew_adapter:
            prompt = {
                "instruction": "Polish and format this itinerary for the user. Produce short intro, per-day timeline, and notes for any gates/warnings.",
                "itinerary": itinerary,
                "metrics": metrics,
                "gates": gates,
                "requirements": payload.get("requirements")
            }
            try:
                resp = self.crew_adapter.run(prompt=str(prompt), context=prompt)
                # Expect 'human_summary' and perhaps modified itinerary
                human_summary = resp.get("human_summary")
                polished_itinerary = resp.get("itinerary", itinerary)
                return {"itinerary": polished_itinerary, "metrics": metrics, "gates": gates, "human_summary": human_summary}
            except Exception:
                pass

        # Default deterministic formatting
        summary = {
            "days": len(itinerary),
            "approx_total_spend_sgd": gates.get("expected_total_spend_sgd", metrics.get("estimated_adult_ticket_spend_sgd")),
            "gates": gates,
            "notes": []
        }
        if not gates.get("all_ok"):
            summary["notes"].append("Plan did not pass all gates; planner agent was invoked.")
        # build per-day short timeline
        timeline = {}
        for date, plan in (itinerary.items() if isinstance(itinerary, dict) else []):
            # plan may be None or missing expected keys — guard carefully
            if not plan or not isinstance(plan, dict):
                timeline[date] = {
                    "morning": None,
                    "lunch": None,
                    "afternoon": None,
                    "travel_minutes": 0,
                    "travel_cost_sgd": 0.0
                }
                continue

            morning_name = safe_item_name(plan.get("morning"))
            lunch_name = safe_item_name(plan.get("lunch"))
            afternoon_name = safe_item_name(plan.get("afternoon"))
            travel_minutes = plan.get("metrics", {}).get("total_travel_min", 0)
            travel_cost = plan.get("metrics", {}).get("total_travel_cost_sgd", 0.0)

            timeline[date] = {
                "morning": morning_name,
                "lunch": lunch_name,
                "afternoon": afternoon_name,
                "travel_minutes": travel_minutes,
                "travel_cost_sgd": travel_cost
            }
        human_summary = {"summary": summary, "timeline": timeline}
        return {"itinerary": itinerary, "metrics": metrics, "gates": gates, "human_summary": human_summary}
