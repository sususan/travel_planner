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

from planner_agent.tools.config import LLM_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try import Crew SDK (adapter will gracefully fallback if not present)
try:
    from crewai import Crew, Agent, Task
except Exception:
    Crew = None
    Agent = None
    Task = None
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

    def __init__(self, verbose: bool = True, max_retries: int = 1):
        self.verbose = verbose
        self.max_retries = max_retries

    def _build_agent_spec(self, requirements: Dict[str, Any], itinerary: Dict[str, Any], transport_options: Dict[str, Any] = None):
        """
        Defines the Agent's role and the Task, enforcing the desired output structure.
        """
        GOAL = "Transform complex JSON travel itinerary data into a visually appealing, human-readable HTML summary document."
        BACKSTORY = (
            "You are a professional Travel Document Editor, known for clear communication, elegant formatting, "
            "and ensuring all key logistical details are present. Your output must be production-ready HTML."
        )
        LLM_CONFIG = {
            "api_key": OPENAI_API_KEY,
            "request_timeout": 60,
            "temperature": 0.2,
            # This is the standard way to force JSON output using LiteLLM/OpenAI config
            "response_format": {"type": "json_object"}
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
        TRANSPORT OPTIONS (Optional): {json.dumps(transport_options, indent=2)}

        --- STYLE AND FORMATTING RULES ---
        1. **Target Audience:** The end-user (traveler). Use a friendly, encouraging, and clear tone.
        2. **Layout:**  
           The output MUST be a **single, complete HTML document** containing:
           - A `<head>` section with `<meta charset="UTF-8">` and `<meta name="viewport" content="width=device-width, initial-scale=1.0">`
           - A `<style>` block or inline CSS for consistent formatting (cards, headings, icons, etc.)
           - A `<body>` section with clear visual hierarchy.
        3. **Content Requirements:**
           - **Title/Header:** Display the trip title and dates prominently.
           - **Introduction Paragraph (1–2 sentences):**  
             Warmly summarize the trip (destination, style, duration).
           - **Day-by-Day Breakdown:**  
             For each day:
             - Include time slots (Morning / Afternoon / Evening).
             - For each place:
               - Show the **place name**, **address**, and **short summary** (15–30 words).  
               - Add **tags or icons** (e.g., “Family-friendly”, “Stroller-friendly”, “Outdoor”).  
               - If available, include booking info or cost.
             - After each place (except the last of the day), include a Transport Section that summarizes the available transport options.
                Use data from the TRANSPORT_OPTIONS JSON where from_place_id == current_place.place_id and to_place_id == next_place.place_id.
             -Transport Selection Rules:
                1. Primary Mode: Always select and display the transport mode with the shortest duration.            
                2.Alternative Modes: Additionally, display up to two alternative modes only if they meet one of the following criteria:            
                    -Significantly Cheaper: Cost is at least 30% lower than the Primary Mode.            
                    -Eco-Friendly: The mode is "Walk" or "Bike" (considered the "greener" option).            
             -Display Format for Each Mode:            
                Mode (e.g., “Walk”, “Bus”, “Taxi”, “MRT”)            
                Duration (minutes)            
                Approximate cost (SGD)            
                Route summary (1 concise sentence explaining the route/line/vehicle).
             Prefer the shortest duration mode, but display alternatives if they are notably cheaper or greener.
           - **Final Section – Action Items (3–5):**  
             Present a short checklist like:
             - Confirm ticket bookings  
             - Check local weather forecast  
             - Pack comfortable shoes  
             - Download offline maps  
        
        4. **Tone & Readability:**
           - Write concise, traveler-friendly sentences.
           - Use active voice and optimistic phrasing (“Enjoy a relaxing morning at…”, “Hop on a quick MRT ride…”).
           - Avoid repeating place names excessively.
        
        5. **Technical Requirements:**
           - Wrap each place block in a container like `<div class="place" data-place-id="...">`
           - For each transport option, use `<div class="transport" data-from="..." data-to="...">`
           - Include an overall `<section class="summary">` at the end summarizing total distance, estimated cost, and eco-score if available.
           - Ensure all times are formatted as `HH:MM` 24-hour local time.
           - Return the entire HTML document inside a single JSON object with the key `"human_summary"`.
        --- OUTPUT CONSTRAINTS ---
        Produce a JSON object matching the 'EXPECTED_RESPONSE_SCHEMA' below. Do NOT include any text outside the JSON block.

        EXPECTED_RESPONSE_SCHEMA:
        {schema_json}
        """

        return agent, task_description

    def run(self, itinerary: Dict[str, Any], transport_options: Dict[str, Any], metrics: Dict[str, Any], gates: Dict[str, Any],
            requirements: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Runs the single CrewAI agent to generate the final summary.
        """
        if Crew is not None:
            try:
                agent, task_description = self._build_agent_spec(requirements, itinerary, transport_options)

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