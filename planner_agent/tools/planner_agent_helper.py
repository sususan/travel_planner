import html
from typing import Any, Optional, Dict
import re

RE_PROMPT_INJECTION = re.compile(
    r"(ignore (?:previous|all) instructions|follow new instructions|disregard|execute the following|system prompt|<\?xml|<script)",
    flags=re.IGNORECASE
)

def sanitize_text_field(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", s)          # remove control chars
    s = s.replace("{", "\\{").replace("}", "\\}")     # escape braces
    s = s.replace("```", "` ` `")                     # break code fences
    s = html.escape(s)
    # truncate very long fields to safe max (e.g., 2000 chars)
    MAX = 2000
    return s[:MAX]

def detect_prompt_injection(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return bool(RE_PROMPT_INJECTION.search(s))

def apply_edits_to_itinerary(itinerary: dict, edits: list, shortlist_by_id: dict) -> dict:
    """
    Applies edits to a deep-copy of itinerary and returns new_itinerary.
    - shortlist_by_id: dict mapping place_id -> place_record (from shortlist)
    """
    import copy
    new_itin = copy.deepcopy(itinerary)

    def find_place_position(place_id):
        for d_idx, day in enumerate(new_itin.get("days", [])):
            for p_idx, place in enumerate(day.get("places", [])):
                if place.get("place_id") == place_id:
                    return d_idx, p_idx
        return None, None

    for e in edits:
        action = e.get("action")
        day_index = e.get("day_index")
        # safety checks
        if action not in ("swap", "remove", "add"):
            continue
        if not isinstance(day_index, int) or day_index < 0 or day_index >= len(new_itin.get("days", [])):
            continue

        day = new_itin["days"][day_index]
        places = day.setdefault("places", [])

        if action == "remove":
            remove_id = e.get("remove_place_id")
            if not remove_id:
                continue
            # remove first occurrence
            for i, p in enumerate(places):
                if p.get("place_id") == remove_id:
                    places.pop(i)
                    break

        elif action == "swap":
            remove_id = e.get("remove_place_id")
            add_id = e.get("add_place_id")
            if not remove_id or not add_id:
                continue
            # find remove position
            pos = None
            for i, p in enumerate(places):
                if p.get("place_id") == remove_id:
                    pos = i
                    break
            # if remove found and add exists in shortlist, replace in same slot; else skip
            add_record = shortlist_by_id.get(add_id)
            if pos is not None and add_record:
                # convert shortlist record into itinerary-place shape if needed
                new_place = {
                    "place_id": add_record["place_id"],
                    "name": add_record.get("name"),
                    "category": add_record.get("category"),
                    "price_estimate": add_record.get("price_estimate"),
                    # include any other fields you require
                }
                places[pos] = new_place

        elif action == "add":
            add_id = e.get("add_place_id")
            insert_after = e.get("insert_after_place_id")
            add_record = shortlist_by_id.get(add_id)
            if not add_record:
                continue
            new_place = {
                "place_id": add_record["place_id"],
                "name": add_record.get("name"),
                "category": add_record.get("category"),
                "price_estimate": add_record.get("price_estimate"),
            }
            if insert_after:
                # find index of insert_after
                idx = None
                for i, p in enumerate(places):
                    if p.get("place_id") == insert_after:
                        idx = i
                        break
                if idx is None:
                    places.append(new_place)
                else:
                    places.insert(idx+1, new_place)
            else:
                # append to day
                places.append(new_place)

        new_itin["days"][day_index]=places

    return new_itin


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

def _extract_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    """
    Try to find the first JSON object in a string and parse it.
    Returns dict or None.
    """
    if not isinstance(s, str):
        return None
    start = s.find("{")
    if start == -1:
        return None
    # try progressively larger substrings until valid JSON parsed or fail
    for i in range(len(s), start - 1, -1):
        candidate = s[start:i]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    # last resort: try to parse the whole string
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None

import json
import re
import ast
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def parse_planner_repair_response(resp: Any) -> Dict[str, Any]:
    """
    Convert planner_repair tool response (which may be dict or str) into a dict.
    Tries multiple strategies and returns a safe dict on failure.
    """
    # 1) If already a dict, normalize and return
    if isinstance(resp, dict):
        return resp

    # 2) If it's bytes, decode
    if isinstance(resp, (bytes, bytearray)):
        try:
            resp = resp.decode()
        except Exception:
            resp = str(resp)

    # 3) If it's not a string now, coerce to str
    if not isinstance(resp, str):
        resp = str(resp)

    raw = resp.strip()

    # 4) Try JSON parse directly
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 5) Try to evaluate as Python literal (safe): handles single quotes / None / True etc.
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 6) If string contains an _raw JSON block (common), extract first {...} and json.loads it
    # This regex tries to match the largest JSON-like object (greedy)
    m = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            # Try Python literal on extracted candidate
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    # 7) Try to find a triple-backtick JSON block like ```json\n{...}\n```
    m2 = re.search(r"```(?:json)?\s*(\{(?:.|\n)*?\})\s*```", raw, flags=re.IGNORECASE)
    if m2:
        candidate = m2.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    # 8) As a last resort, try to salvage by converting single quotes to double quotes (best-effort)
    try:
        candidate = raw.replace("'", '"')
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 9) Give up: return a safe fallback with the raw string for audit
    logger.warning("parse_planner_repair_response: failed to parse response; returning fallback")
    return {
        "edits": [],
        "explain_summary": "parse_error_or_unrecognized_format",
        "confidence": "low",
        "_raw": raw
    }
