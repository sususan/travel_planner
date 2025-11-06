# final_agent.py
"""
FinalAgent: produce a polished human-readable summary (HTML preferred) from structured itinerary.
- FinalAgent.run(...) returns:
    {
      "human_summary": "<html>...</html>" OR "plain text string",
      "content_type": "html" | "text",
      "attachments": [ { "filename": "...", "content": "..." }, ... ]  # optional
    }
- Designed to be called as:
    crew_resp = crew_adapter.run(prompt=str(prompt), task_description=str(task_description))
  (This file mirrors the planner agent calling convention.)
- If Crew is not available or fails, a deterministic HTML fallback is returned.
"""
import logging
import os
import json
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try import Crew SDK (adapter will gracefully fallback if not present)
try:
    from crewai import Crew, Agent, Task
except Exception:
    Crew = None
    Agent = None
    Task = None

LLM_MODEL = os.getenv("CREW_LLM_MODEL", "gpt-4o-mini")


# ------------------------
# Helper for Fallback HTML (Simplified)
# ------------------------
def build_simple_html(itinerary: Dict[str, Any], metrics: Dict[str, Any], gates: Dict[str, Any]) -> str:
    """Builds a basic HTML summary from the itinerary data."""
    html = "<div style='font-family: sans-serif; max-width: 800px; margin: auto;'>"
    html += "<h2 style='border-bottom: 2px solid #333; padding-bottom: 5px;'>Itinerary Details</h2>"

    for date, plan in itinerary.items():
        html += f"<h3 style='color: #007bff;'>Day: {date}</h3>"
        for slot_name, slot_data in plan.items():
            if slot_data.get('item'):
                item = slot_data['item']
                item_name = item.get('name', 'Unknown Item')
                html += f"<p><strong>{slot_name.title()}</strong>: {item_name} (Duration: {item.get('duration', 'N/A')})</p>"

    # Add metrics and gate info (optional)
    html += "<h2 style='margin-top: 20px;'>Summary Metrics (Unformatted)</h2>"
    html += f"<pre>{json.dumps(metrics, indent=2)}</pre>"
    html += "</div>"
    return html


# ------------------------
# Crew adapter for final (compatible with planner usage)
# ------------------------
def _parse_crew_output(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Parses the final agent's output. The final agent returns a dict containing
    the "human_summary" (often a full HTML string) and metadata.
    """
    if raw is None:
        return None

    # Handle CrewOutput objects directly
    if hasattr(raw, 'raw') and isinstance(raw.raw, str):
        raw = raw.raw
    elif hasattr(raw, 'result') and isinstance(raw.result, str):
        raw = raw.result

    # If already a dict, return it
    if isinstance(raw, dict):
        return raw

    # The final agent is expected to return JSON containing the HTML string.
    if isinstance(raw, str):
        try:
            # First attempt: treat the whole string as JSON
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "human_summary" in parsed:
                return parsed
        except Exception:
            # Second attempt: try to find a JSON substring
            try:
                if '```json' in raw:
                    raw = raw.split('```json', 1)[-1].split('```', 1)[0].strip()
                elif '```' in raw:
                    # Generic code fence
                    raw = raw.split('```', 1)[-1].split('```', 1)[0].strip()

                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "human_summary" in parsed:
                    return parsed
            except Exception:
                logger.error("Failed final agent JSON parsing.")
                return None
    return None


class CrewAIAdapterForFinal:
    """
    Adapter that accepts the planner-style call:
        run(prompt=str(prompt), task_description=str(task_description))
    It constructs a Crew with one Agent/Task to produce the final, rich summary.
    """

    def __init__(self, verbose: bool = False, max_retries: int = 1):
        self.verbose = verbose
        self.max_retries = max_retries

    def _build_agent_spec(self, requirements: Dict[str, Any], itinerary: Dict[str, Any]):
        """
        Defines the Agent's role and the Task, enforcing the desired output structure.
        """
        GOAL = "Transform complex JSON travel itinerary data into a visually appealing, human-readable HTML summary document."
        BACKSTORY = (
            "You are a professional Travel Document Editor, known for clear communication, elegant formatting, "
            "and ensuring all key logistical details are present. Your output must be production-ready HTML."
        )

        LLM_CONFIG = {
            "temperature": 0.2,  # Use a slightly higher temperature for creative formatting
            "response_format": {"type": "json_object"},
            "max_tokens": 2048  # Allow for a larger HTML output
        }

        # Agent Definition
        if Agent is not None:
            agent = Agent(
                role="Travel Document Editor",
                goal=GOAL,
                backstory=BACKSTORY,
                allow_delegation=False,
                verbose=self.verbose,
                llm=LLM_MODEL,
                config=LLM_CONFIG
            )
        else:
            agent = {
                "name": "final_summary_agent",
                "role": "Travel Document Editor",
                "goal": GOAL,
                "backstory": BACKSTORY,
                "llm": LLM_MODEL,
                "config": LLM_CONFIG
            }

        # Expected Output Schema for the final JSON wrapper
        expected_schema = {
            "human_summary": "string (MUST be a complete HTML document including <head> and <body>, styled for mobile/desktop)",
            "content_type": "html",
            "attachments": "array (empty array if no attachments needed)"
        }
        schema_json = json.dumps(expected_schema, indent=2)

        # Task Description (Comprehensive Prompt)
        task_description = f"""
        --- FINAL SUMMARY TASK ---

        Your objective is to generate the final deliverable.

        DATA INPUT:
        USER REQUIREMENTS: {json.dumps(requirements, indent=2)}
        FINAL ITINERARY (Validated): {json.dumps(itinerary, indent=2)}

        --- STYLE AND FORMATTING RULES ---
        1. **Target Audience:** The end-user (traveler). Use a friendly, encouraging, and clear tone.
        2. **Layout:** The output MUST be a single, complete HTML document. Use inline CSS or `<style>` tags.
        3. **Content:**
            - Title/Header for the trip.
            - A brief, welcoming introduction paragraph (1-2 sentences).
            - A **Day-by-Day breakdown** of the itinerary, including times, place names, and brief descriptions.
            - A final section with 3-5 key **Action Items** (e.g., "Confirm ticket bookings," "Check weather," "Pack comfortable shoes").

        --- OUTPUT CONSTRAINTS ---
        Produce a JSON object matching the 'EXPECTED_RESPONSE_SCHEMA' below. Do NOT include any text outside the JSON block.

        EXPECTED_RESPONSE_SCHEMA:
        {schema_json}
        """

        return agent, task_description

    def run(self, itinerary: Dict[str, Any], metrics: Dict[str, Any], gates: Dict[str, Any],
            requirements: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Runs the single CrewAI agent to generate the final summary.
        """
        if Crew is not None:
            try:
                agent, task_description = self._build_agent_spec(requirements, itinerary)

                # The Task uses the comprehensive task_description as its prompt
                summary_task = Task(
                    description=task_description,
                    agent=agent,
                    expected_output="A single JSON object matching the provided schema, containing the full HTML summary.",
                    async_execution=False
                )

                crew = Crew(
                    agents=[agent],
                    tasks=[summary_task],
                    verbose=self.verbose,
                )

                logger.info("Invoking CrewAI summary agent.")
                raw = crew.kickoff()

                crew_resp = _parse_crew_output(raw)

                if crew_resp and crew_resp.get("human_summary"):
                    # Determine content type based on the presence of HTML tags
                    content_type = "html" if isinstance(crew_resp.get("human_summary"), str) and (
                                "<html" in crew_resp.get("human_summary") or "<body" in crew_resp.get(
                            "human_summary")) else "text"
                    return {
                        "human_summary": crew_resp.get("human_summary"),
                        "content_type": content_type,
                        "attachments": crew_resp.get("attachments", []) or []
                    }
            except Exception as e:
                logger.error("Crew adapter invocation failed: %s", e)
                # Fall through to deterministic fallback

        # Deterministic fallback (HTML) so downstream can render it to PDF
        html = build_simple_html(itinerary or {}, metrics or {}, gates or {})
        intro = "<p><em>This summary was auto-generated (Crew unavailable). Please verify times and bookings.</em></p>"
        actions = "<h3>Action items</h3><ul><li>Verify opening hours</li><li>Confirm transport tickets</li><li>Book timed-entry tickets if needed</li></ul>"
        full_html = f"<html><body><h1>Itinerary Summary</h1>{intro}{html}{actions}</body></html>"

        return {"human_summary": full_html, "content_type": "html", "attachments": []}