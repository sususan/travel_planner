import json
from typing import Any, Dict, Optional, Iterable

def _try_parse_json(s: str) -> Optional[Dict]:
    """Try to parse a JSON string; if text contains JSON blob, attempt to extract it."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    # try direct parse
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # fallback: find first {...} block and try parse
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(s[start:end+1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None

def _walk_find_itinerary(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Recursively search object (dict/list/str) for a dict that contains 'itinerary'
    or at least both 'itinerary'/'metrics'. Returns the dict found or None.
    """
    if obj is None:
        return None

    # If already a dict that looks like final output, accept it
    if isinstance(obj, dict):
        # direct hit
        if "itinerary" in obj:
            return obj
        # some agents may use 'final_itinerary' or 'result' naming — normalize common variants
        for alt in ("final_itinerary", "plan", "result"):
            if alt in obj and isinstance(obj[alt], dict) and "itinerary" in obj[alt]:
                return obj[alt]
        # search common fields that may contain stringified JSON or nested dicts/lists
        for k, v in obj.items():
            # skip huge context fields if necessary (but we still search)
            found = _walk_find_itinerary(v)
            if found:
                return found
        return None

    # If a list, check each element
    if isinstance(obj, list):
        for elem in obj:
            found = _walk_find_itinerary(elem)
            if found:
                return found
        return None

    # If a string, maybe it's JSON or contains JSON
    if isinstance(obj, str):
        parsed = _try_parse_json(obj)
        if parsed:
            # parsed is a dict — try to find itinerary inside it too
            found = _walk_find_itinerary(parsed)
            if found:
                return found
            # else if parsed itself contains itinerary, return it
            if "itinerary" in parsed:
                return parsed
        return None

    # Not a recognized type
    return None

def _parse_crew_output(raw) -> Optional[Dict[str, Any]]:
    """
    Robust parser for Crew run outputs.

    Returns:
      - dict containing at least 'itinerary' (and optionally 'metrics') if found
      - otherwise None
    """
    # quick guard
    if raw is None:
        return None

    # 1) If raw is a dict and directly contains itinerary -> return it immediately
    if isinstance(raw, dict) and "itinerary" in raw:
        return raw

    # 2) Recursively search raw for a dict with 'itinerary'
    found = _walk_find_itinerary(raw)
    if found:
        return found

    # 3) As last resort, if raw is a string try parsing any JSON blob and return that dict
    if isinstance(raw, str):
        parsed = _try_parse_json(raw)
        if parsed:
            # prefer dict containing 'itinerary' if present
            if "itinerary" in parsed:
                return parsed
            # otherwise return parsed dict (caller may tolerate)
            return parsed

    # Nothing usable found
    return None
