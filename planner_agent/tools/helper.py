# helper.py
# ---------- UPDATED ----------
# Helpers: imputation, aggregation, data quality penalty, transport proxy, haversine.
# NOTE: This file was updated to return per-age price tiers and compute budget aggregations.
import re
from collections import deque, defaultdict
from datetime import datetime
from difflib import get_close_matches
from typing import Dict, Any, Tuple, List, Iterable, Optional, Set
from math import radians, cos, sin, asin, sqrt

child_multiplier = 0.5  # Default value as fallback
senior_multiplier =0.7

#---------- Short List helper ---------------#

def pick_by_cluster_roundrobin(entries: List[Dict[str, Any]],
                               days: int) -> List[Dict[str, Any]]:
    """
    entries: list of dicts, each dict must have at least "cluster" and "score"
    days: number of items to pick
    place_id_of: function to retrieve a stable place id from an entry
    """
    # 1) Group entries by cluster and sort each group by score desc
    cluster_to_deque: Dict[str, deque] = {}
    for cluster, group in defaultdict(list, ((e["cluster"], []) for e in entries)).items():
        pass  # keep for clarity (we build below)

    # Helper: robust place id
    def place_id_of(entry):
        item = entry.get("item") or {}
        return item.get("place_id") or item.get("id") or item.get("name")

    tmp = defaultdict(list)
    for e in entries:
        tmp[e["cluster"]].append(e)

    for cluster, group in tmp.items():
        group_sorted = sorted(group, key=lambda x: x.get("score", 0), reverse=True)
        cluster_to_deque[cluster] = deque(group_sorted)

    if not cluster_to_deque:
        return []

    # 2) Order clusters by their best entry score (descending)
    clusters_ordered = sorted(
        cluster_to_deque.keys(),
        key=lambda c: (cluster_to_deque[c][0].get("score", 0) if cluster_to_deque[c] else -float("inf")),
        reverse=True
    )

    selected: List[Dict[str, Any]] = []
    used_place_ids: Set[Any] = set()

    # 3) Round-robin across clusters (in clusters_ordered) until we have enough or nothing left
    # We continue rounds until filled or all deques empty
    while len(selected) < days:
        any_popped_this_round = False
        for c in clusters_ordered:
            if len(selected) >= days:
                break
            dq = cluster_to_deque.get(c)
            # pop until we find an unused place_id or dq empties
            while dq:
                candidate = dq[0]  # peek
                pid = place_id_of(candidate)
                if pid in used_place_ids:
                    # duplicate id — remove and continue
                    dq.popleft()
                    continue
                # accept this candidate
                selected.append(candidate)
                used_place_ids.add(pid)
                dq.popleft()
                any_popped_this_round = True
                break
        if not any_popped_this_round:
            # nothing left to pick (all deques empty or items were duplicates)
            break

    return selected


# reuse your haversine from core if available; reimplement minimal here to avoid circular imports
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    try:
        R = 6371.0
        dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return 2 * R * asin(sqrt(a))
    except Exception:
        return 0.0

def impute_price(candidate: dict, medians_by_type: dict = None) -> dict:
    medians_by_type = medians_by_type or {}
    tp = candidate.get("ticket_price_sgd") or {}
    adult_v = 0.0

    if isinstance(tp, dict) and tp.get("adults") not in (None, ""):
        try:
            adult_v = float(tp.get("adults") or 0.0)
            child_v = float(tp.get("children")) if tp.get("children") not in (None, "") else round(adult_v * child_multiplier,2)
            senior_v = float(tp.get("senior")) if tp.get("senior") not in (None, "") else round(
                adult_v * senior_multiplier, 2)
            return {"adults": adult_v, "children": child_v, "senior": senior_v, "method": "known_tiers",
                    "uncertainty_pct": 0.0}
        except Exception:
            pass

    if candidate.get("cost_sgd"):
        try:
            adult_v = candidate.get("cost_sgd")
            return {"adults": adult_v, "children": adult_v, "senior": adult_v, "method": "known_tiers",
                    "uncertainty_pct": 0.0}
        except Exception:
            pass

    it_type = (candidate.get("type") or "").lower()
    it_tags = set(token_to_canonical(x) for x in (candidate.get("tags") or []))
    it_terms = set(token_to_canonical(x) for x in (candidate.get("terms") or []))
    # unify candidate vocabulary
    cand_vocab = set(it_terms) | {it_type} | set(it_tags)

    # 2) median_by_type (expected structure: medians_by_type[type] = {"adults":..,"children":..,"senior":..} or scalar adult)
    for t in cand_vocab:
        if medians_by_type and t in medians_by_type:
            m = medians_by_type[t]
            if isinstance(m, dict):
                adult_v = float(m.get("adults", 0.0))
                child_v = float(m.get("children", adult_v * child_multiplier))
                senior_v = float(m.get("senior", adult_v * senior_multiplier))
            else:
                adult_v = float(m)
                child_v = round(adult_v * child_multiplier, 2)
                senior_v = round(adult_v * senior_multiplier, 2)
            return {"adults": adult_v, "children": child_v, "senior": senior_v, "method": "median_by_type",
                    "uncertainty_pct": 0.0}

    # 3) fallback adult
    fallback_table = {
        "park": 0.0,
        "garden": 0.0,
        "museum": 20.0,
        "zoo": 40.0,
        "theme_park": 75.0,
        "temple": 0.0,
        "landmark": 35.0,
        "dining": 20.0,
        "food": 20.0,
        "restaurant": 35.0
    }
    adult_v = 15.0
    for t in cand_vocab:
        tmp = float(fallback_table.get(t, 15.0))
        if tmp > adult_v:
            adult_v = tmp

    return {
        "adults": adult_v,
        "children": round(adult_v * child_multiplier, 2),
        "senior": round(adult_v * senior_multiplier, 2),
        "method": "fallback_table",
        "uncertainty_pct": 0.20
    }

def dq_penalty_for_item(candidate: dict) -> float:
    """
    UPDATED: data-quality penalty to nudge planner/human review when metadata missing.
    Caps at 0.5.
    """
    penalty = 0.0
    # check known price presence
    tp = candidate.get("ticket_price_sgd")
    if not tp:
        penalty += 0.15
    if candidate.get("low_carbon_score") in (None, 0):
        penalty += 0.05
    if not candidate.get("opening_hours"):
        penalty += 0.10
    return min(0.5, penalty)

def aggregate_budget_range(scheduled_items: List[dict], medians_by_type: dict = None, group_counts: dict = None) -> Dict[str, Any]:
    """
    UPDATED: Accepts a list of scheduled items (each is a dict with key 'item'),
    computes expected/min/max totals for the group using impute_price and group_counts.
    group_counts: {"adults":1,"children":0,"seniors":0}
    Returns dict with expected/min/max, unknown_frac and uncertainty_ratio.
    """
    medians_by_type = medians_by_type or {}
    group_counts = group_counts or {"adults":1,"children":0,"seniors":0}
    adults = group_counts.get("adults", 1)
    children = group_counts.get("children", 0)
    seniors = group_counts.get("senior", 0)

    expected = 0.0
    min_t = 0.0
    max_t = 0.0
    unknown_count = 0
    n = max(1, len(scheduled_items))
    for s in scheduled_items:
        it = s.get("item") if isinstance(s, dict) else s
        price = impute_price(it, medians_by_type)
        # compute expected total for this item for group
        expected_item = price["adults"] * adults + price["children"] * children + price["senior"] * seniors
        expected += expected_item
        # min/max by uncertainty_pct
        u = price.get("uncertainty_pct", 0.0)
        min_t += expected_item * (1 - u)
        max_t += expected_item * (1 + u)
        if price.get("method") != "known_tiers" and price.get("method") != "known":
            unknown_count += 1
    unknown_frac = unknown_count / n
    uncertainty_ratio = (max_t - expected) / max(1e-6, expected)
    return {"expected": expected, "min": min_t, "max": max_t, "unknown_frac": unknown_frac, "uncertainty_ratio": uncertainty_ratio}

def transport_proxy_options(from_item: dict, to_item: dict) -> List[dict]:
    """
    Simple transport options generator used when real transport API is not yet connected.
    Returns plausible transport modes with estimated duration_min, cost_sgd, co2_kg.
    """
    from_geo = from_item.get("geo") or {}
    to_geo = to_item.get("geo") or {}
    try:
        f_lat, f_lon = float(from_geo.get("latitude")), float(from_geo.get("longitude"))
        t_lat, t_lon = float(to_geo.get("latitude")), float(to_geo.get("longitude"))
        dist = _haversine_km(f_lat, f_lon, t_lat, t_lon)
    except Exception:
        dist = 2.0

    opts = []
    if dist <= 1.5:
        opts.append({"mode": "walk", "duration_min": int(dist * 20 + 5), "co2_kg": 0.0, "cost_sgd": 0.0})
    if dist <= 10:
        opts.append({"mode": "metro", "duration_min": max(8, int(dist * 5 + 5)), "co2_kg": round(dist * 0.06, 3), "cost_sgd": 1.5})
    opts.append({"mode": "taxi", "duration_min": max(5, int(dist * 3 + 5)), "co2_kg": round(dist * 0.2, 3), "cost_sgd": round(dist * 1.5 + 3, 2)})
    return opts

def _ids_match(conn: dict, a: str, b: str) -> bool:
    """Return True if connection matches a->b considering undirected flag."""
    if not conn:
        return False
    f = conn.get("from_id")
    t = conn.get("to_id")
    if f == a and t == b:
        return True
    # If undirected (or missing flag, assume undirected), reversed also matches
    if conn.get("undirected", True) and f == b and t == a:
        return True
    return False

import json
from typing import List, Dict, Optional, Tuple, Any

# --- Normalization helper ---
def normalize_connections(connections: Any) -> List[Dict]:
    """
    Normalize input to a list of connection dicts.
    Accepts:
      - List[dict]
      - dict with key "connections": [...]
      - single connection dict
      - JSON string of any of the above
    Raises TypeError if input cannot be normalized.
    """
    if isinstance(connections, str):
        # try parsing JSON string
        try:
            parsed = json.loads(connections)
        except json.JSONDecodeError:
            raise TypeError("String connections must be valid JSON")
        return normalize_connections(parsed)

    if isinstance(connections, dict):
        # common pattern: {"connections": [ ... ]}
        if "connections" in connections and isinstance(connections["connections"], list):
            return connections["connections"]
        # single connection dict -> wrap in list
        return [connections]

    if isinstance(connections, list):
        return connections

    raise TypeError(f"Unsupported connections type: {type(connections)}")


# --- ID extraction helper (tolerant to different key names) ---
def _extract_ids(conn: Dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Try multiple canonical keys for from/to ids:
      - from_id, to_id
      - from_place_id, to_place_id
      - from_place_name, to_place_name (last resort)
      - from, to
    Returns tuple (from_id, to_id) (either may be None).
    """
    candidates_from = [
        "from_id", "from_place_id", "fromPlaceId", "from", "from_place_name", "from_place"
    ]
    candidates_to = [
        "to_id", "to_place_id", "toPlaceId", "to", "to_place_name", "to_place"
    ]
    def find_key(d, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return str(d[k])
        return None

    return find_key(conn, candidates_from), find_key(conn, candidates_to)


def _ids_match(conn: Dict, a: str, b: str) -> bool:
    """
    Returns True if conn represents a link between a and b.
    Considers undirected edges when conn.get("undirected") is truthy.
    Comparison is string equality after str() to be tolerant.
    """
    from_id, to_id = _extract_ids(conn)

    # if either id is missing, try matching by place_name fields as fallback
    if from_id is None or to_id is None:
        # try the place_name keys
        from_name = conn.get("from_place_name") or conn.get("from_name") or None
        to_name = conn.get("to_place_name") or conn.get("to_name") or None
        if from_name and to_name:
            from_id, to_id = str(from_name), str(to_name)

    # if still missing, fail fast
    if from_id is None or to_id is None:
        return False

    # canonicalize inputs to strings
    a_s, b_s = str(a), str(b)

    undirected = bool(conn.get("undirected") or conn.get("is_undirected"))

    if undirected:
        return (from_id == a_s and to_id == b_s) or (from_id == b_s and to_id == a_s)
    else:
        return (from_id == a_s and to_id == b_s)


# --- Finder functions ---
def find_connection_index(connections: Any, a: str, b: str) -> Optional[int]:
    """
    Return index of connection in normalized list that matches a and b (considering undirected),
    or None if not found.
    """
    conns = normalize_connections(connections)
    for idx, conn in enumerate(conns):
        if _ids_match(conn, a, b):
            return idx
    return None

def get_connection(connections: Any, a: str, b: str) -> Optional[Tuple[int, Dict]]:
    """
    Return (index, connection_dict) or None
    """
    conns = normalize_connections(connections)
    for idx, conn in enumerate(conns):
        if _ids_match(conn, a, b):
            return idx, conn
    return None

def compute_daily_travel_summary(day_plan: dict) -> Dict[str, Any]:
    """
    Compute total travel minutes, cost, co2 for a day from 'transport_options' list per hop.
    """
    total_min = 0
    total_cost = 0.0
    total_co2 = 0.0
    for hop in day_plan.get("transport_options", []):
        options = hop.get("options") or []
        # pick preferred option; default: metro if available, else first
        opt = next((o for o in options if o.get("mode") == "metro"), options[0] if options else None)
        if opt:
            total_min += int(opt.get("duration_min", 0))
            total_cost += float(opt.get("cost_sgd", 0.0))
            total_co2 += float(opt.get("co2_kg", 0.0))
    return {"total_travel_min": total_min, "total_travel_cost_sgd": round(total_cost, 2), "total_travel_co2_kg": round(total_co2, 3)}

def safe_item_name(slot_obj):
    """
    Robustly extract item name from a slot which may be None, or a dict possibly missing keys.
    Returns None if name not available.
    """
    if not slot_obj or not isinstance(slot_obj, dict):
        return None
    item = slot_obj.get("item")
    if not item or not isinstance(item, dict):
        return None
    return item.get("name")

def normalize_token(tok: str) -> str:
    """
    Normalize a single token:
      - lowercase
      - replace hyphens with underscore
      - remove common suffixes like '-friendly'
      - remove non-alpha chars at ends
      - naive singularization for trailing 's' (only when length>3)
    """
    if not tok:
        return ""
    s = tok.lower().strip()
    s = s.replace("-", "_")
    # remove '-friendly' style tokens: family_friendly -> family
    s = re.sub(r'(_?friendly)$', '', s)
    # remove punctuation
    s = re.sub(r'[^a-z0-9_ ]+', '', s)
    s = s.strip()
    # simple plural handling: museums -> museum (only naive)
    if len(s) > 3 and s.endswith("s"):
        s_sing = s[:-1]
        # guard against turning 'gas'->'ga'
        if len(s_sing) >= 3:
            s = s_sing
    return s

# small synonyms mapping — extendable
SYNONYMS = {
    "museum": {"museum", "museums", "gallery"},
    "park": {"park", "parks", "garden"},
    "theme_park": {"theme_park", "themepark", "theme", "amusement_park"},
    "family": {"family", "family_friendly", "familyfriendly", "family-friendly"},
    "photo_spot": {"photo_spot", "photo_spot", "photo", "photo-spot"},
    "attraction": {"attraction", "attractions"},
    "interactive": {"interactive", "hands_on"},
    "wheelchair_friendly": {"wheelchair_friendly", "accessible"},
    # add more domain-specific synonyms here
}

def build_synonym_map(synonyms: dict = None):
    """Invert SYNONYMS to map token -> canonical key"""
    synonyms = synonyms or SYNONYMS
    inv = {}
    for canon, variants in synonyms.items():
        for v in variants:
            inv[normalize_token(v)] = canon
    return inv

_SYNONYM_INV = build_synonym_map()

def token_to_canonical(tok: str) -> str:
    """
    Normalizes token and maps synonyms to canonical token if available.
    """
    n = normalize_token(tok)
    return _SYNONYM_INV.get(n, n)

# ---------- Interest scoring function ----------
def compute_interest_score(user_interests: Iterable[str], candidate_terms: Iterable[str], candidate_type: str, candidate_tags: Iterable[str]) -> float:
    """
    Compute a weighted interest score in [0..1].
    Strategy:
     - Normalize all inputs
     - Exact canonical matches -> weight 1.0
     - Synonym canonical matches (via synonyms map) -> weight 0.95 (treated as exact by canon)
     - Partial token overlap (substring or token prefix) -> weight 0.6
     - Fuzzy matches via difflib (cutoff) -> weight 0.5
    Returns average match fraction scaled by these weights vs number of user interests.
    """
    # build normalized canonical sets
    ui = set(token_to_canonical(x) for x in (user_interests or []))
    # defensive: if user provided a single dict set, handle it
    ui = {x for x in ui if x}
    it_terms = set(token_to_canonical(x) for x in (candidate_terms or []))
    it_type = token_to_canonical(candidate_type or "")
    it_tags = set(token_to_canonical(x) for x in (candidate_tags or []))
    # unify candidate vocabulary
    cand_vocab = set(it_terms) | {it_type} | set(it_tags)

    if not ui:
        return 0.6  # neutral when user provided no explicit interests

    score_sum = 0.0
    for interest in ui:
        if not interest:
            continue
        # Exact canonical match
        if interest in cand_vocab:
            score_sum += 1.0
            continue
        # Partial substring / token prefix match (e.g., 'museum' vs 'museum_exhibit')
        found_partial = False
        for cv in cand_vocab:
            if cv and (interest in cv or cv in interest):
                score_sum += 0.6
                found_partial = True
                break
        if found_partial:
            continue
        # Fuzzy match fallback with difflib
        # get_close_matches returns similar strings; cutoff tuned to 0.8 for high similarity
        candidates_list = [c for c in cand_vocab if c]
        close = get_close_matches(interest, candidates_list, n=2, cutoff=0.8)
        if close:
            score_sum += 0.5
            continue
        # no match
        score_sum += 0.0

    # normalized by number of interests (so more interests makes it harder to reach 1.0)
    return float(score_sum) / max(1, len(ui))

def eco_score_from_low_carbon(low_carbon_score):
    # treat None/empty like 0 if you prefer a default; here we coerce to float safely
    try:
        s = float(low_carbon_score)
    except (TypeError, ValueError):
        s = 0.0

    # clamp to 0..100 then invert to 1..0
    s = max(0.0, min(100.0, s))
    return 1.0 - (s / 100.0)

# Score

def _safe_get(d: dict, path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _to_lower_set(seq) -> set:
    if not seq:
        return set()
    return {str(x).strip().lower() for x in seq if x is not None}

def _tokenize(s: str) -> List[str]:
    if not s:
        return []
    return re.findall(r"[a-z0-9]+", s.lower())



def _grid_cluster(lat: Optional[float], lon: Optional[float], size_deg: float = 0.05) -> str:
    """Fallback cluster if geo_cluster_id missing. ~5.5km grid at equator for size_deg=0.05."""
    if lat is None or lon is None:
        return "cluster_unknown"
    gx = int(lat / size_deg)
    gy = int(lon / size_deg)
    return f"cluster_{gx}_{gy}"

def _normalize(x: float, lo: float, hi: float, invert: bool=False) -> float:
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    v = 1.0 - v if invert else v
    if v < 0.0: v = 0.0
    if v > 1.0: v = 1.0
    return v

def _parse_open_hours_cell(cell) -> float:
    """
    Parse standardized opening-hours cell and return total open hours as float.

    Supported input formats (examples):
      - "15:00-01:30"                   -> interval crossing midnight handled
      - "Closed"                        -> returns 0.0
      - "12:00-15:00,06:30-22:30"       -> multiple intervals separated by commas
      - also accepts a list of strings, will sum them

    Rules:
      - times must be HH:MM with two-digit minutes.
      - intervals separated by commas are summed.
      - if end <= start, we treat it as end on the next day (add 24 hours).
      - invalid/malformed intervals are skipped (do not raise).
    """
    if not cell:
        return 0.0

    # If list, sum recursively
    if isinstance(cell, list):
        return sum(_parse_open_hours_cell(c) for c in cell)

    s = str(cell).strip()
    if not s:
        return 0.0

    # "Closed" handling (case-insensitive)
    if s.lower() == "closed" or s.lower() == "close":
        return 0.0

    total_hours = 0.0
    # split by commas to allow multiple intervals
    parts = [p.strip() for p in s.split(",") if p.strip()]

    # regex for HH:MM-HH:MM (24-hour-ish)
    interval_re = re.compile(r'^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$')

    for part in parts:
        m = interval_re.match(part)
        if not m:
            # skip malformed parts gracefully
            continue
        sh, sm, eh, em = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        # normalize to fractional hours
        start = float(sh) + float(sm)/60.0
        end = float(eh) + float(em)/60.0
        # if end <= start assume crossing midnight -> add 24h to end
        if end <= start:
            end += 24.0
        duration = end - start
        if duration > 0:
            total_hours += duration

    return max(total_hours, 0.0)

def _get_open_hours_for_weekday(item: dict, weekday_name_lower: str) -> float:
    opening = item.get("opening_hours") or {}
    # handle e.g. {"monday": "...", "tue": "..."} or mixed cases
    for k, v in opening.items():
        if k.lower().startswith(weekday_name_lower[:3]):  # "mon" matches "monday"
            return _parse_open_hours_cell(v)
    return 0.0


def _parse_open_hours_cellbalbak(cell) -> float:
    if not cell:
        return 0.0
    if isinstance(cell, list):
        return sum(_parse_open_hours_cell(c) for c in cell)
    s = str(cell)
    if "all day" in s.lower() or "24" in s:
        return 24.0
    total = 0.0
    for part in str(s).split(","):
        part = part.strip()
        m = re.match(r"^\s*(\d{1,2}):?(\d{2})?-(\d{1,2}):?(\d{2})?\s*$", part)
        if m:
           sh, sm, eh, em = int(m.group(1)), int(m.group(2) or 0), int(m.group(3)), int(m.group(4) or 0)
           total += (eh + em / 60) - (sh + sm / 60)
           return max(total, 0.0)
        m_dash = re.search(r'^\s*(.+?)\s*-\s*(.+?)\s*$', part)
        if m:
            sh, sm, eh, em = int(m.group(1)), int(m.group(2) or 0), int(m.group(3)), int(m.group(4) or 0)
            total += (eh + em / 60) - (sh + sm / 60)
            return max(total, 0.0)

def _candidate_interest_terms(item: dict) -> set:
    """
    Generic interest terms from type, tags, and name—so it adapts to arbitrary user interests.
    """
    terms = set()
    typ = (item.get("type") or "")
    terms.update(_tokenize(typ))
    for t in item.get("tags", []) or []:
        terms.update(_tokenize(str(t)))
    terms.update(_tokenize(item.get("name", "")))
    return {t for t in terms if t}

def _is_dining(item: dict) -> bool:
    typ = str(item.get("type") or "").lower()
    if typ in {"dining", "restaurant", "food", "eatery", "cafe"}:
        return True
    tags = _to_lower_set(item.get("tags", []))
    return any(t in tags for t in {"restaurant", "food", "eat", "cafe", "lunch", "dining"})

def _diet_fit_score(item: dict, diet_pref: Optional[str]) -> float:
    """
    diet_pref can be "vegan", "vegetarian", "halal", "kosher", etc.
    We apply a soft match using tags/name. Non-dining items return neutral 0.7.
    """
    if not _is_dining(item):
        return 0.7
    if not diet_pref:
        return 0.8
    tags = _to_lower_set(item.get("tags", []))
    name_tokens = _candidate_interest_terms(item)
    key = diet_pref.lower()
    # allow common synonyms
    synonyms = {
        "vegan": {"vegan", "plantbased", "plant", "veg"},
        "vegetarian": {"vegetarian", "veg", "ovo", "lacto"},
        "halal": {"halal"},
        "kosher": {"kosher"},
        "glutenfree": {"glutenfree", "gluten-free"},
    }
    keys = synonyms.get(key, {key})
    hit = bool(keys & (tags | name_tokens))
    return 1.0 if hit else 0.5

def _is_accessible(item: dict) -> bool:
    tags = _to_lower_set(item.get("tags", []))
    name_tokens = _candidate_interest_terms(item)
    return any(x in tags or x in name_tokens for x in {"wheelchair", "accessible", "barrierfree", "barrier-free"})

def _get_geo(item: dict) -> Tuple[Optional[float], Optional[float]]:
    g = item.get("geo") or {}
    lat = g.get("latitude")
    lon = g.get("longitude")
    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
    except Exception:
        lat, lon = None, None
    return lat, lon

def _cluster_id(item: dict) -> str:
    cid = item.get("geo_cluster_id")
    if cid:
        return str(cid).lower()
    lat, lon = _get_geo(item)
    return _grid_cluster(lat, lon)

def _date_span_days(req: dict) -> int:
    # 1) explicit duration_days
    d = _safe_get(req, ["duration_days"])
    if d:
        try:
            return int(d)
        except Exception:
            pass
    # 2) compute from trip_dates
    start = _safe_get(req, ["trip_dates", "start_date"])
    end = _safe_get(req, ["trip_dates", "end_date"])
    if start and end:
        try:
            s = datetime.fromisoformat(start).date()
            e = datetime.fromisoformat(end).date()
            return max((e - s).days + 1, 1)
        except Exception:
            pass
    return 3  # fallback

def _start_date(req: dict) -> datetime:
    start = _safe_get(req, ["trip_dates", "start_date"])
    if start:
        try:
            return datetime.fromisoformat(start)
        except Exception:
            pass
    return datetime.now()

def _slot_times(req: dict) -> Dict[str, str]:
    # allow overrides like {"morning":"09:30","lunch":"12:30","afternoon":"15:00"}
    slot_times = _safe_get(req, ["optional", "slot_times"], {}) or {}
    return {
        "morning": slot_times.get("morning", "10:00"),
        "lunch": slot_times.get("lunch", "12:30"),
        "afternoon": slot_times.get("afternoon", "15:00"),
    }

def _accom_latlon(req: dict) -> Tuple[Optional[float], Optional[float]]:
    acc = _safe_get(req, ["optional", "accommodation_location"], {}) or {}
    lat = acc.get("lat")
    lon = acc.get("lng")
    try:
        return (float(lat), float(lon))
    except Exception:
        return (None, None)

"""
pace_mapping = {
            "slow": 2,
            "relaxed": 3,
            "moderate": 4,
            "active": 5,
            "fast":6
            }
"""
def _pace_minutes(pace: str) -> int:
    pace = (pace or "standard").lower()
    if pace in ("slow","relaxed"):
        return 360  # 6h
    if pace in ("moderate"):
        return 420 # 7h
    if pace in ("active"):
        return 540  # 9h
    if pace in ("fast"):
        return 600  # 10h
    return 420      # 7h default

def _minutes_for_item(x: dict) -> int:
    """
    Robustly infer a visit duration for a candidate.
    Checks several common fields; falls back by type.
    """
    item = x.get("item", {})
    # Common fields various retrievers may provide
    for k in "duration_recommended_minutes":
        v = item.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)

    # Type-based fallback (tunable)
    it_type = (x.get("type") or "").lower()
    it_tags = set(token_to_canonical(x) for x in (x.get("tags") or []))
    it_terms = set(token_to_canonical(x) for x in (x.get("terms") or []))
    # unify candidate vocabulary
    cand_vocab = set(it_terms) | {it_type} | set(it_tags)
    for t in cand_vocab:
        if t in {"theme", "theme_park", "zoo"}:
            return 210 #3.5hr
        if t in {"museum", "neighborhood_walk"}:
            return 90
        if t in {"park", "garden", "nature"}:
            return 90
        if t in {"shopping", "shopping_mall"}:
            return 90
        if t in {"attraction", "tourist_attraction"}:
            return 60
        if t in {"landmark", "temple", "viewpoint", "photo_spot", "tourist-spot"}:
            return 60

    # Dining will be handled separately; default non-dining duration:
    return 90

def _lunch_minutes(x: dict) -> int:
    """Dining stop time assumption (prefer item hints if any)."""
    item = x.get("item", {})
    for k in "duration_recommended_minutes":
        v = item.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    # default lunch block
    return 90

from typing import Dict, Any, List, Optional
import math

# ---------- Scoring helper ---------------------------------------------------
def _compute_item_score(entry: Dict[str, Any],
                        requirements: Dict[str, Any],
                        metrics: Dict[str, Any],
                        weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute a composite score in [0,1] for a candidate itinerary item.

    Factors (default weights):
      - relevance (tags vs user interests)       w_relevance = 0.35
      - proximity (same cluster as accommodation) w_proximity  = 0.20
      - eco friendliness                          w_eco       = 0.20
      - accessibility                             w_access    = 0.10
      - popularity / quality                      w_pop      = 0.15
      - cost_penalty applied as a multiplier (reduces score)

    The function is robust if some fields are missing.
    """
    # default weights (tweakable)
    if weights is None:
        weights = {
            "relevance": 0.35,
            "proximity": 0.20,
            "eco": 0.20,
            "access": 0.10,
            "pop": 0.15,
        }

    # --- relevance: overlap between entry tags and user interests (0..1) ---
    interests = set([s.lower() for s in (requirements.get("interests") or [])])
    tags = set([s.lower() for s in (entry.get("tags") or [])])
    relevance = 0.0
    if interests and tags:
        relevance = len(interests & tags) / max(1, len(interests))
        relevance = min(1.0, relevance)

    # --- proximity: 1 if same cluster as accommodation, else 0 (or distance-based if available) ---
    proximity = 0.0
    acc_cluster = metrics.get("accommodation_cluster")
    item_cluster = entry.get("geo_cluster_id") or entry.get("cluster")
    if acc_cluster and item_cluster:
        proximity = 1.0 if acc_cluster == item_cluster else 0.0
    else:
        # fallback: small score if distance_km exists and <= 1 km
        dist = entry.get("distance_km")
        if isinstance(dist, (int, float)):
            proximity = max(0.0, 1 - min(dist / 10.0, 1.0))  # 0..1, linear up to 10 km

    # --- eco friendliness: binary or numeric (preferred_for_eco True/False or eco_score 0..1) ---
    eco = 0.0
    if entry.get("preferred_for_eco") is True:
        eco = 1.0
    elif isinstance(entry.get("eco_score"), (int, float)):
        eco = max(0.0, min(1.0, float(entry["eco_score"])))

    # --- accessibility: boolean ---
    access = 1.0 if entry.get("accessible") or entry.get("is_accessible") or _is_accessible(entry) else 0.0

    # --- popularity / quality: normalize popularity if given (assume 0..100) ---
    pop = 0.5  # neutral default
    if isinstance(entry.get("popularity"), (int, float)):
        pop = max(0.0, min(1.0, float(entry["popularity"]) / 100.0))
    elif isinstance(entry.get("rating"), (int, float)):
        # rating 1..5 => scale to 0..1
        pop = max(0.0, min(1.0, (float(entry["rating"]) - 1) / 4.0))

    # --- base composite score ---
    composite = (
        weights["relevance"] * relevance +
        weights["proximity"] * proximity +
        weights["eco"] * eco +
        weights["access"] * access +
        weights["pop"] * pop
    )

    # --- cost penalty: if item too expensive relative to remaining budget, penalize ---
    cost_penalty = 1.0
    try:
        price = impute_price(entry)  # expects dict with keys adult/child/senior
        # estimate single-adult price if available
        candidate_price = price.get("adults") or price.get("price") or 0.0
        remaining = metrics.get("estimated_remaining_budget_sgd")
        if remaining is not None and remaining >= 0:
            # ratio candidate_price / remaining
            ratio = (candidate_price / max(1.0, remaining))
            # apply a soft penalty (exponential) so extremes are punished more
            cost_penalty = 1.0 / (1.0 + math.pow(max(0.0, ratio), 1.5))
            cost_penalty = max(0.15, min(1.0, cost_penalty))  # clamp
    except Exception:
        cost_penalty = 1.0

    score = float(composite) * float(cost_penalty)
    # normalize to [0,1]
    score = max(0.0, min(1.0, score))
    return score


# ---------- Alternatives generator ------------------------------------------
def _generate_alternatives(entry: Dict[str, Any],
                           requirements: Dict[str, Any],
                           metrics: Dict[str, Any],
                           top_n: int = 2) -> List[Dict[str, Any]]:
    """
    Generate up to `top_n` alternative items for a given `entry`.
    Preferred sources (in order):
      - entry.get("nearby_candidates"): a list of candidate dicts
      - metrics.get("catalog"): a global list of candidate dicts (falls back)
    Each alternative will include name, score, and minimal provenance.
    If no candidates are found, returns an empty list.

    IMPORTANT: Keep alternatives lightweight (no heavy data).
    """
    candidates = []
    # first try explicit nearby candidates on the entry (recommended)
    if isinstance(entry.get("nearby_candidates"), list) and entry["nearby_candidates"]:
        candidates = entry["nearby_candidates"]
    else:
        # fallback to a catalog in metrics (this is optional)
        catalog = metrics.get("catalog") or metrics.get("attraction_catalog") or []
        # naive proximity: pick candidates from same cluster if possible
        item_cluster = entry.get("geo_cluster_id") or entry.get("cluster")
        if item_cluster:
            candidates = [c for c in catalog if (c.get("geo_cluster_id") or c.get("cluster")) == item_cluster]
        # if still empty, use top N from catalog
        if not candidates and catalog:
            candidates = catalog[:50]  # limit scanning work

    # compute score for each candidate and pick top_n excluding the original item (by name or id)
    scored = []
    orig_name = (entry.get("name") or "").strip().lower()
    for c in candidates:
        if (c.get("name") or "").strip().lower() == orig_name:
            continue
        try:
            s = _compute_item_score(c, requirements, metrics)
        except Exception:
            s = 0.0
        scored.append((s, c))

    # sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    alternatives = []
    for s, c in scored[:top_n]:
        alternatives.append({
            "name": c.get("name"),
            "type": c.get("type"),
            "cluster": c.get("geo_cluster_id") or c.get("cluster"),
            "score": round(float(s), 3),
            "provenance": c.get("source") or "catalog"
        })
    return alternatives


# ---------- Default gate fixes ---------------------------------------------
def _default_fix_for_gate(gname: str) -> str:
    """
    Provide a short, actionable default fix suggestion for common gates.
    Keep text concise — UI can display it verbatim or use it to auto-replan.
    """
    mapping = {
        "budget_gate_v1": "Replace one paid attraction with a free alternative (park/museum) or reduce ticket budget by SGD 50-150 per person.",
        "carbon_gate_v1": "Substitute taxi legs with MRT/walking where feasible; cluster activities to reduce transit; remove one far-flung stop.",
        "coverage_gate_v1": "Add attractions from underrepresented clusters (e.g., heartlands) to improve geographic coverage.",
        "accessibility_gate_v1": "Swap inaccessible stops for nearby accessible equivalents; mark required accessibility features in requirements.",
        "time_gate_v1": "Rebalance by moving one attraction to another day or shorten visit durations by ~30 minutes.",
        # fallback
        "default": "Consider relaxing one constraint (budget/pace/coverage) or allow the planner to propose 1-2 swaps."
    }
    return mapping.get(gname, mapping["default"])
